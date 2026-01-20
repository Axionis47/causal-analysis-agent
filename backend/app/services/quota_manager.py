"""Quota management service for tracking user usage and enforcing limits."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.analysis import Analysis, AnalysisStatus
from app.services.rate_limiter import rate_limiter

logger = get_logger(__name__)


@dataclass
class QuotaReservationResult:
    """Result of a quota reservation attempt."""
    allowed: bool
    error_message: str
    exceeded_limit_type: Optional[str]  # "concurrent", "hourly", "daily", or None
    concurrent_count: Optional[int]
    hourly_remaining: Optional[int]
    daily_remaining: Optional[int]
    hourly_reset: Optional[int]
    daily_reset: Optional[int]

# Model pricing per 1K tokens (USD)
MODEL_PRICING = {
    "gpt-4": 0.03,
    "gpt-4o": 0.015,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0005,
    "claude-3-opus": 0.015,
    "claude-3-5-sonnet": 0.003,
    "claude-3-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "gemini-1.5-pro": 0.00125,
    "gemini-1.0-pro": 0.000125,
    "default": 0.01,  # Default pricing if model unknown
}


class QuotaManager:
    """
    Manages user quotas and usage tracking.

    Provides methods to check concurrent analysis limits, track LLM usage,
    and aggregate usage statistics for admin visibility.
    """

    async def get_running_analyses_count(
        self, db: AsyncSession, user_id: UUID
    ) -> int:
        """
        Count running analyses for a user.

        Args:
            db: Database session
            user_id: The user's UUID

        Returns:
            Number of currently running analyses
        """
        result = await db.execute(
            select(func.count())
            .select_from(Analysis)
            .where(
                Analysis.user_id == user_id,
                Analysis.status == AnalysisStatus.RUNNING,
            )
        )
        return result.scalar_one()

    async def can_start_analysis(
        self, db: AsyncSession, user_id: UUID
    ) -> tuple[bool, str]:
        """
        Check if user can start a new analysis based on all limits.

        Args:
            db: Database session
            user_id: The user's UUID

        Returns:
            Tuple of (can_start, error_message)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, ""

        # Check concurrent analysis limit
        running_count = await self.get_running_analyses_count(db, user_id)
        if running_count >= settings.RATE_LIMIT_CONCURRENT_ANALYSES:
            return False, (
                f"Maximum {settings.RATE_LIMIT_CONCURRENT_ANALYSES} concurrent analyses allowed. "
                f"You currently have {running_count} running analyses."
            )

        # Check hourly rate limit
        hourly_ok = await rate_limiter.check_rate_limit(user_id, "hourly")
        if not hourly_ok:
            remaining = await rate_limiter.get_remaining(user_id, "hourly")
            return False, (
                f"Hourly rate limit exceeded. "
                f"Limit: {settings.RATE_LIMIT_ANALYSES_PER_HOUR}/hour. "
                f"Remaining: {remaining}."
            )

        # Check daily rate limit
        daily_ok = await rate_limiter.check_rate_limit(user_id, "daily")
        if not daily_ok:
            remaining = await rate_limiter.get_remaining(user_id, "daily")
            return False, (
                f"Daily rate limit exceeded. "
                f"Limit: {settings.RATE_LIMIT_ANALYSES_PER_DAY}/day. "
                f"Remaining: {remaining}."
            )

        return True, ""

    async def record_analysis_started(self, user_id: UUID) -> None:
        """
        Record that a user started an analysis (increment rate limit counters).

        DEPRECATED: Use reserve_analysis_quota() instead which handles this atomically.

        Args:
            user_id: The user's UUID
        """
        await rate_limiter.increment_counter(user_id, "hourly")
        await rate_limiter.increment_counter(user_id, "daily")

    async def reserve_analysis_quota(
        self, db: AsyncSession, user_id: UUID
    ) -> QuotaReservationResult:
        """
        Atomically check and reserve quota for a new analysis.

        This method:
        1. Checks concurrent analysis limits (from database)
        2. Atomically checks and reserves hourly/daily rate limits (via Redis Lua script)

        The atomic reservation prevents race conditions where concurrent requests
        could exceed the quota despite the checks passing.

        Args:
            db: Database session
            user_id: The user's UUID

        Returns:
            QuotaReservationResult with allowed status and detailed limit information
        """
        if not settings.RATE_LIMIT_ENABLED:
            return QuotaReservationResult(
                allowed=True,
                error_message="",
                exceeded_limit_type=None,
                concurrent_count=None,
                hourly_remaining=settings.RATE_LIMIT_ANALYSES_PER_HOUR,
                daily_remaining=settings.RATE_LIMIT_ANALYSES_PER_DAY,
                hourly_reset=None,
                daily_reset=None,
            )

        # Check concurrent analysis limit first (from database)
        running_count = await self.get_running_analyses_count(db, user_id)
        if running_count >= settings.RATE_LIMIT_CONCURRENT_ANALYSES:
            logger.warning(
                "Quota exceeded",
                user_id=str(user_id),
                quota_type="concurrent",
                current_usage=running_count,
                limit=settings.RATE_LIMIT_CONCURRENT_ANALYSES,
            )
            return QuotaReservationResult(
                allowed=False,
                error_message=(
                    f"Maximum {settings.RATE_LIMIT_CONCURRENT_ANALYSES} concurrent analyses allowed. "
                    f"You currently have {running_count} running analyses."
                ),
                exceeded_limit_type="concurrent",
                concurrent_count=running_count,
                hourly_remaining=None,
                daily_remaining=None,
                hourly_reset=None,
                daily_reset=None,
            )

        # Atomically check and reserve hourly/daily quota
        reserve_result = await rate_limiter.atomic_reserve(user_id)

        if not reserve_result.allowed:
            if reserve_result.exceeded_limit_type == "hourly":
                current_usage = None
                if reserve_result.hourly_remaining is not None:
                    current_usage = settings.RATE_LIMIT_ANALYSES_PER_HOUR - reserve_result.hourly_remaining
                logger.warning(
                    "Quota exceeded",
                    user_id=str(user_id),
                    quota_type="hourly",
                    current_usage=current_usage,
                    limit=settings.RATE_LIMIT_ANALYSES_PER_HOUR,
                )
                error_message = (
                    f"Hourly rate limit exceeded. "
                    f"Limit: {settings.RATE_LIMIT_ANALYSES_PER_HOUR}/hour. "
                    f"Remaining: {reserve_result.hourly_remaining}."
                )
            else:  # daily
                current_usage = None
                if reserve_result.daily_remaining is not None:
                    current_usage = settings.RATE_LIMIT_ANALYSES_PER_DAY - reserve_result.daily_remaining
                logger.warning(
                    "Quota exceeded",
                    user_id=str(user_id),
                    quota_type="daily",
                    current_usage=current_usage,
                    limit=settings.RATE_LIMIT_ANALYSES_PER_DAY,
                )
                error_message = (
                    f"Daily rate limit exceeded. "
                    f"Limit: {settings.RATE_LIMIT_ANALYSES_PER_DAY}/day. "
                    f"Remaining: {reserve_result.daily_remaining}."
                )

            return QuotaReservationResult(
                allowed=False,
                error_message=error_message,
                exceeded_limit_type=reserve_result.exceeded_limit_type,
                concurrent_count=running_count,
                hourly_remaining=reserve_result.hourly_remaining,
                daily_remaining=reserve_result.daily_remaining,
                hourly_reset=reserve_result.hourly_reset,
                daily_reset=reserve_result.daily_reset,
            )

        return QuotaReservationResult(
            allowed=True,
            error_message="",
            exceeded_limit_type=None,
            concurrent_count=running_count,
            hourly_remaining=reserve_result.hourly_remaining,
            daily_remaining=reserve_result.daily_remaining,
            hourly_reset=reserve_result.hourly_reset,
            daily_reset=reserve_result.daily_reset,
        )

    async def get_user_usage_stats(
        self, db: AsyncSession, user_id: UUID, days: int = 30
    ) -> dict[str, Any]:
        """
        Get aggregate usage statistics for a user.

        Args:
            db: Database session
            user_id: The user's UUID
            days: Number of days to aggregate (default 30)

        Returns:
            Dictionary with usage statistics
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get total analyses and costs in period
        result = await db.execute(
            select(
                func.count().label("total_analyses"),
                func.coalesce(func.sum(Analysis.llm_tokens_used), 0).label("total_tokens"),
                func.coalesce(func.sum(Analysis.estimated_cost), Decimal("0")).label("total_cost"),
            )
            .select_from(Analysis)
            .where(
                Analysis.user_id == user_id,
                Analysis.created_at >= since,
            )
        )
        row = result.one()

        # Get running analyses count
        running_count = await self.get_running_analyses_count(db, user_id)

        # Get remaining quotas
        hourly_remaining = await rate_limiter.get_remaining(user_id, "hourly")
        daily_remaining = await rate_limiter.get_remaining(user_id, "daily")

        # Get status breakdown
        status_result = await db.execute(
            select(
                Analysis.status,
                func.count().label("count"),
            )
            .select_from(Analysis)
            .where(
                Analysis.user_id == user_id,
                Analysis.created_at >= since,
            )
            .group_by(Analysis.status)
        )
        status_breakdown = {
            status.value: count for status, count in status_result.all()
        }

        return {
            "period_days": days,
            "total_analyses": row.total_analyses,
            "total_tokens_used": int(row.total_tokens),
            "total_cost": float(row.total_cost),
            "running_analyses": running_count,
            "status_breakdown": status_breakdown,
            "rate_limits": {
                "hourly": {
                    "limit": settings.RATE_LIMIT_ANALYSES_PER_HOUR,
                    "remaining": hourly_remaining,
                    "reset_time": await rate_limiter.get_reset_time(user_id, "hourly"),
                },
                "daily": {
                    "limit": settings.RATE_LIMIT_ANALYSES_PER_DAY,
                    "remaining": daily_remaining,
                    "reset_time": await rate_limiter.get_reset_time(user_id, "daily"),
                },
                "concurrent": {
                    "limit": settings.RATE_LIMIT_CONCURRENT_ANALYSES,
                    "current": running_count,
                },
            },
        }

    async def track_llm_usage(
        self,
        db: AsyncSession,
        analysis_id: UUID,
        tokens: int,
        model: str,
    ) -> None:
        """
        Track LLM token usage and update estimated cost for an analysis.

        Args:
            db: Database session
            analysis_id: The analysis UUID
            tokens: Number of tokens used
            model: Model name for pricing calculation
        """
        # Calculate cost based on model pricing
        price_per_1k = MODEL_PRICING.get(model.lower(), MODEL_PRICING["default"])
        cost = (tokens / 1000) * price_per_1k

        # Update analysis with accumulated values
        result = await db.execute(
            select(Analysis).where(Analysis.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()

        if analysis:
            analysis.llm_tokens_used = (analysis.llm_tokens_used or 0) + tokens
            analysis.estimated_cost = float(
                Decimal(str(analysis.estimated_cost or 0)) + Decimal(str(cost))
            )
            await db.flush()

            logger.debug(
                "LLM usage tracked",
                analysis_id=str(analysis_id),
                tokens=tokens,
                model=model,
                cost=cost,
                total_tokens=analysis.llm_tokens_used,
                total_cost=float(analysis.estimated_cost),
            )

    def calculate_cost(self, tokens: int, model: str) -> float:
        """
        Calculate cost for a given number of tokens and model.

        Args:
            tokens: Number of tokens
            model: Model name

        Returns:
            Estimated cost in USD
        """
        price_per_1k = MODEL_PRICING.get(model.lower(), MODEL_PRICING["default"])
        return (tokens / 1000) * price_per_1k

    def estimate_tokens_from_text(self, text: str) -> int:
        """
        Estimate token count from text (approximate).

        Uses a simple heuristic of ~4 characters per token.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4


# Singleton instance
quota_manager = QuotaManager()
