"""API dependencies for authentication, authorization, and rate limiting."""

import uuid
from typing import Annotated

import sentry_sdk
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user as core_get_current_user
from app.core.config import settings
from app.core.logging import bind_contextvars
from app.db.database import get_async_session
from app.services.quota_manager import quota_manager
from app.services.rate_limiter import rate_limiter


async def get_current_user(
    request: Request,
    user_id: Annotated[uuid.UUID, Depends(core_get_current_user)],
) -> uuid.UUID:
    """Bind user context after authentication."""
    request.state.user_id = user_id
    bind_contextvars(user_id=str(user_id))
    sentry_payload = {"id": str(user_id)}
    user_email = getattr(request.state, "user_email", None)
    if user_email:
        sentry_payload["email"] = str(user_email)
    sentry_sdk.set_user(sentry_payload)
    return user_id


async def check_analysis_quota(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: Annotated[uuid.UUID, Depends(get_current_user)],
) -> uuid.UUID:
    """
    Dependency to check if user can start a new analysis.

    Sets request.state.user_id for per-user rate limiting with SlowAPI.

    Checks:
    - Concurrent analysis limits
    - Hourly rate limits
    - Daily rate limits

    Args:
        request: FastAPI request object
        db: Database session
        user_id: Current user's UUID

    Returns:
        The user_id if checks pass

    Raises:
        HTTPException: 429 if any quota is exceeded
    """
    # Set user_id on request state for SlowAPI per-user rate limiting
    request.state.user_id = user_id

    result = await quota_manager.reserve_analysis_quota(db, user_id)

    if not result.allowed:
        # Build headers based on the exceeded limit type
        headers = _build_rate_limit_error_headers(result)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=result.error_message,
            headers=headers,
        )

    return user_id


def _build_rate_limit_error_headers(result) -> dict[str, str]:
    """Build rate limit headers based on the exceeded limit type."""
    import time

    current_time = int(time.time())
    headers = {}

    if result.exceeded_limit_type == "concurrent":
        headers["X-RateLimit-Limit-Concurrent"] = str(settings.RATE_LIMIT_CONCURRENT_ANALYSES)
        headers["X-RateLimit-Current-Concurrent"] = str(result.concurrent_count or 0)
        # For concurrent limits, suggest retrying after a reasonable delay
        headers["Retry-After"] = "60"
    elif result.exceeded_limit_type == "hourly":
        headers["X-RateLimit-Limit-Hourly"] = str(settings.RATE_LIMIT_ANALYSES_PER_HOUR)
        headers["X-RateLimit-Remaining-Hourly"] = str(result.hourly_remaining or 0)
        headers["X-RateLimit-Reset-Hourly"] = str(result.hourly_reset or current_time + 3600)
        headers["Retry-After"] = str(max(0, (result.hourly_reset or current_time + 3600) - current_time))
    elif result.exceeded_limit_type == "daily":
        headers["X-RateLimit-Limit-Daily"] = str(settings.RATE_LIMIT_ANALYSES_PER_DAY)
        headers["X-RateLimit-Remaining-Daily"] = str(result.daily_remaining or 0)
        headers["X-RateLimit-Reset-Daily"] = str(result.daily_reset or current_time + 86400)
        headers["Retry-After"] = str(max(0, (result.daily_reset or current_time + 86400) - current_time))
    else:
        # Generic fallback
        headers["Retry-After"] = "3600"

    return headers


async def get_current_admin_user(
    user_id: Annotated[uuid.UUID, Depends(get_current_user)],
) -> uuid.UUID:
    """
    Dependency to verify the current user is an admin.

    Args:
        user_id: Current user's UUID

    Returns:
        The user_id if user is admin

    Raises:
        HTTPException: 403 if user is not an admin
    """
    if str(user_id) not in settings.ADMIN_USER_IDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user_id


async def get_rate_limit_headers(user_id: uuid.UUID) -> dict[str, str]:
    """
    Get rate limit headers for successful response.

    Uses standardized header names:
    - X-RateLimit-Limit-Hourly: Maximum analyses per hour
    - X-RateLimit-Remaining-Hourly: Remaining analyses this hour
    - X-RateLimit-Reset-Hourly: Unix timestamp when hourly limit resets
    - X-RateLimit-Limit-Daily: Maximum analyses per day
    - X-RateLimit-Remaining-Daily: Remaining analyses today
    - X-RateLimit-Reset-Daily: Unix timestamp when daily limit resets

    Args:
        user_id: Current user's UUID

    Returns:
        Dictionary of rate limit headers
    """
    hourly_remaining = await rate_limiter.get_remaining(user_id, "hourly")
    hourly_reset = await rate_limiter.get_reset_time(user_id, "hourly")
    daily_remaining = await rate_limiter.get_remaining(user_id, "daily")
    daily_reset = await rate_limiter.get_reset_time(user_id, "daily")

    return {
        "X-RateLimit-Limit-Hourly": str(settings.RATE_LIMIT_ANALYSES_PER_HOUR),
        "X-RateLimit-Remaining-Hourly": str(hourly_remaining),
        "X-RateLimit-Reset-Hourly": str(hourly_reset),
        "X-RateLimit-Limit-Daily": str(settings.RATE_LIMIT_ANALYSES_PER_DAY),
        "X-RateLimit-Remaining-Daily": str(daily_remaining),
        "X-RateLimit-Reset-Daily": str(daily_reset),
    }
