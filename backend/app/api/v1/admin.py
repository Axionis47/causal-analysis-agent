"""Admin API routes for quota management and user oversight."""

from datetime import datetime
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_admin_user
from app.core.config import settings
from app.crud.analysis import analysis_crud
from app.db.database import get_async_session
from app.models.analysis import Analysis, AnalysisStatus
from app.models.llm_log import LLMLog
from app.models.user import User
from app.schemas.quota import (
    AdminStats,
    LLMCacheInvalidateResponse,
    LLMCacheStats,
    LLMLogListResponse,
    QuotaResetResponse,
    QuotaUsage,
    UserAnalysisListResponse,
    UserListResponse,
    UserUsageStats,
)
from app.services.llm_cache import llm_cache_service
from app.services.quota_manager import quota_manager
from app.services.rate_limiter import rate_limiter

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get(
    "/users/{user_id}/quota",
    response_model=QuotaUsage,
    summary="Get user quota usage",
    responses={
        200: {
            "description": "User's current quota usage and limits",
            "content": {
                "application/json": {
                    "example": {
                        "analyses_this_hour": 3,
                        "analyses_this_day": 15,
                        "running_analyses": 1,
                        "total_tokens_used": 150000,
                        "total_cost": 2.50,
                        "limits": {
                            "hourly": 10,
                            "daily": 50,
                            "concurrent": 3,
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        404: {"description": "User not found"},
    },
)
async def get_user_quota(
    user_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
) -> QuotaUsage:
    """
    Get quota usage for a specific user.

    **Admin-only endpoint** - requires admin authentication.

    Returns current usage counts and remaining quotas for rate limiting:

    **Usage Metrics:**
    - `analyses_this_hour`: Analyses created in the current hour
    - `analyses_this_day`: Analyses created today
    - `running_analyses`: Currently executing analyses
    - `total_tokens_used`: LLM tokens consumed
    - `total_cost`: Estimated cost in USD

    **Limits:**
    - `hourly`: Maximum analyses per hour (configurable)
    - `daily`: Maximum analyses per day (configurable)
    - `concurrent`: Maximum simultaneous running analyses

    **Rate Limit Headers:**
    When users hit rate limits, they receive these response headers:
    - `X-RateLimit-Limit-Hourly`
    - `X-RateLimit-Remaining-Hourly`
    - `X-RateLimit-Reset-Hourly` (Unix timestamp)
    """
    # Get usage stats
    stats = await quota_manager.get_user_usage_stats(db, user_id, days=1)

    # Get hourly analyses count
    hourly_remaining = await rate_limiter.get_remaining(user_id, "hourly")
    daily_remaining = await rate_limiter.get_remaining(user_id, "daily")
    running_count = await quota_manager.get_running_analyses_count(db, user_id)

    return QuotaUsage(
        analyses_this_hour=settings.RATE_LIMIT_ANALYSES_PER_HOUR - hourly_remaining,
        analyses_this_day=settings.RATE_LIMIT_ANALYSES_PER_DAY - daily_remaining,
        running_analyses=running_count,
        total_tokens_used=stats["total_tokens_used"],
        total_cost=stats["total_cost"],
        limits={
            "hourly": settings.RATE_LIMIT_ANALYSES_PER_HOUR,
            "daily": settings.RATE_LIMIT_ANALYSES_PER_DAY,
            "concurrent": settings.RATE_LIMIT_CONCURRENT_ANALYSES,
        },
    )


@router.get(
    "/users/{user_id}/usage",
    response_model=UserUsageStats,
    summary="Get user usage statistics",
    responses={
        200: {
            "description": "Detailed usage statistics for the specified period",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "period_days": 30,
                        "total_analyses": 45,
                        "total_tokens_used": 500000,
                        "total_cost": 12.50,
                        "running_analyses": 1,
                        "status_breakdown": {
                            "completed": 40,
                            "failed": 3,
                            "cancelled": 2,
                        },
                        "rate_limits": {
                            "hourly_remaining": 7,
                            "daily_remaining": 35,
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def get_user_usage_stats(
    user_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
) -> UserUsageStats:
    """
    Get detailed usage statistics for a user over a specified period.

    **Admin-only endpoint** - requires admin authentication.

    Returns comprehensive usage metrics for monitoring user activity
    and managing quotas.

    **Period Range:** 1-365 days (default: 30 days)

    **Statistics Included:**
    - Total analyses created in the period
    - LLM token consumption
    - Estimated cost breakdown
    - Analysis status breakdown (completed, failed, cancelled)
    - Current rate limit status
    """
    stats = await quota_manager.get_user_usage_stats(db, user_id, days=days)

    return UserUsageStats(
        user_id=user_id,
        period_days=stats["period_days"],
        total_analyses=stats["total_analyses"],
        total_tokens_used=stats["total_tokens_used"],
        total_cost=stats["total_cost"],
        running_analyses=stats["running_analyses"],
        status_breakdown=stats["status_breakdown"],
        rate_limits=stats["rate_limits"],
    )


@router.get(
    "/users/{user_id}/analyses",
    response_model=UserAnalysisListResponse,
    summary="List user's analyses",
    responses={
        200: {
            "description": "Paginated list of user's analyses with cost breakdown",
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
                                "status": "completed",
                                "llm_tokens_used": 12000,
                                "estimated_cost": 0.25,
                                "created_at": "2026-01-20T12:00:00Z",
                                "completed_at": "2026-01-20T12:15:00Z",
                            }
                        ],
                        "total": 45,
                        "skip": 0,
                        "limit": 100,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def list_user_analyses(
    user_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    status_filter: Optional[AnalysisStatus] = Query(None, description="Filter by status"),
) -> UserAnalysisListResponse:
    """
    List analyses for a specific user with optional status filtering.

    **Admin-only endpoint** - requires admin authentication.

    Returns paginated list of analyses with cost breakdown for each.
    Useful for monitoring user activity and investigating issues.

    **Filtering:**
    - Filter by status: `pending`, `running`, `completed`, `failed`, `cancelled`

    **Response includes:**
    - Analysis metadata (ID, URL, status, timestamps)
    - Token usage and estimated cost per analysis
    - Pagination info (total count, skip, limit)
    """
    # Build query
    query = select(Analysis).where(Analysis.user_id == user_id)
    count_query = select(func.count()).select_from(Analysis).where(Analysis.user_id == user_id)

    if status_filter:
        query = query.where(Analysis.status == status_filter)
        count_query = count_query.where(Analysis.status == status_filter)

    query = query.order_by(Analysis.created_at.desc()).offset(skip).limit(limit)

    # Execute queries
    result = await db.execute(query)
    analyses = result.scalars().all()

    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Build response with cost info
    items = [
        {
            "id": str(a.id),
            "kaggle_url": a.kaggle_url,
            "status": a.status.value,
            "llm_tokens_used": a.llm_tokens_used,
            "estimated_cost": float(a.estimated_cost),
            "created_at": a.created_at.isoformat(),
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
        }
        for a in analyses
    ]

    return UserAnalysisListResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.post(
    "/users/{user_id}/quota/reset",
    response_model=QuotaResetResponse,
    summary="Reset user rate limits",
    responses={
        200: {
            "description": "Rate limits reset successfully",
            "content": {
                "application/json": {
                    "example": {
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "reset_types": ["hourly", "daily"],
                        "message": "Successfully reset hourly, daily rate limits for user",
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def reset_user_quota(
    user_id: uuid.UUID,
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
    reset_hourly: bool = Query(True, description="Reset hourly rate limit counter"),
    reset_daily: bool = Query(True, description="Reset daily rate limit counter"),
) -> QuotaResetResponse:
    """
    Reset rate limits for a specific user.

    **Admin-only endpoint** - requires admin authentication.

    Immediately resets the specified rate limit counters, allowing
    the user to create new analyses without waiting for automatic reset.

    **Reset Options:**
    - `reset_hourly`: Reset the hourly analysis limit counter
    - `reset_daily`: Reset the daily analysis limit counter

    **Note:** This does not affect the concurrent analysis count,
    which is based on actually running analyses. To free up concurrent
    slots, the running analyses must complete or be cancelled.

    **Use Cases:**
    - User hit rate limit due to failed analyses
    - Priority user needs immediate access
    - Testing and development
    """
    reset_types = []

    if reset_hourly:
        await rate_limiter.reset_counter(user_id, "hourly")
        reset_types.append("hourly")

    if reset_daily:
        await rate_limiter.reset_counter(user_id, "daily")
        reset_types.append("daily")

    return QuotaResetResponse(
        user_id=user_id,
        reset_types=reset_types,
        message=f"Successfully reset {', '.join(reset_types)} rate limits for user",
    )


@router.get(
    "/stats",
    response_model=AdminStats,
    summary="Get system-wide statistics",
    responses={
        200: {
            "description": "Aggregate statistics across all users",
            "content": {
                "application/json": {
                    "example": {
                        "total_users": 150,
                        "total_analyses": 2500,
                        "total_tokens_used": 50000000,
                        "total_cost": 125.50,
                        "running_analyses": 5,
                        "status_breakdown": {
                            "completed": 2300,
                            "failed": 150,
                            "cancelled": 50,
                            "running": 5,
                            "pending": 0,
                        },
                        "llm_cache_hit_rate": 0.35,
                        "llm_costs_by_model": {
                            "gpt-4": 80.25,
                            "gpt-3.5-turbo": 25.00,
                            "claude-3-opus": 20.25,
                        },
                        "llm_avg_latency_ms_by_provider": {
                            "openai": 1500,
                            "anthropic": 2000,
                            "vertex": 1800,
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def get_admin_stats(
    db: Annotated[AsyncSession, Depends(get_async_session)],
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
) -> AdminStats:
    """
    Get aggregate statistics across all users.

    **Admin-only endpoint** - requires admin authentication.

    Returns system-wide metrics for monitoring platform health and costs.

    **User Metrics:**
    - Total registered users
    - Total analyses created
    - Currently running analyses

    **Cost Metrics:**
    - Total LLM tokens used
    - Total estimated cost (USD)
    - Cost breakdown by LLM model

    **Performance Metrics:**
    - LLM cache hit rate (higher is better, saves costs)
    - Average latency by LLM provider
    - Analysis status breakdown

    **Monitoring Use Cases:**
    - Track platform usage growth
    - Monitor LLM costs and optimize model selection
    - Identify performance issues by provider
    - Assess cache effectiveness
    """
    # Total users
    user_count_result = await db.execute(select(func.count()).select_from(User))
    total_users = user_count_result.scalar_one()

    # Total analyses and costs
    analysis_stats_result = await db.execute(
        select(
            func.count().label("total_analyses"),
            func.coalesce(func.sum(Analysis.llm_tokens_used), 0).label("total_tokens"),
            func.coalesce(func.sum(Analysis.estimated_cost), 0).label("total_cost"),
        ).select_from(Analysis)
    )
    analysis_stats = analysis_stats_result.one()

    # Status breakdown
    status_result = await db.execute(
        select(Analysis.status, func.count().label("count"))
        .select_from(Analysis)
        .group_by(Analysis.status)
    )
    status_breakdown = {status.value: count for status, count in status_result.all()}

    # Currently running analyses
    running_result = await db.execute(
        select(func.count())
        .select_from(Analysis)
        .where(Analysis.status == AnalysisStatus.RUNNING)
    )
    running_analyses = running_result.scalar_one()

    # LLM cache hit rate
    cache_stats = await llm_cache_service.get_cache_stats()
    cache_hits = cache_stats.get("hits", 0)
    cache_misses = cache_stats.get("misses", 0)
    cache_total = cache_hits + cache_misses
    llm_cache_hit_rate = float(cache_hits / cache_total) if cache_total else 0.0

    # LLM cost breakdown by model
    llm_costs_result = await db.execute(
        select(
            LLMLog.model,
            func.coalesce(func.sum(LLMLog.estimated_cost), 0).label("total_cost"),
        ).group_by(LLMLog.model)
    )
    llm_costs_by_model = {model: float(total_cost) for model, total_cost in llm_costs_result.all()}

    # LLM average latency by provider
    llm_latency_result = await db.execute(
        select(
            LLMLog.provider,
            func.avg(LLMLog.latency_ms).label("avg_latency"),
        ).group_by(LLMLog.provider)
    )
    llm_avg_latency_ms_by_provider = {
        provider: float(avg_latency) if avg_latency is not None else 0.0
        for provider, avg_latency in llm_latency_result.all()
    }

    return AdminStats(
        total_users=total_users,
        total_analyses=analysis_stats.total_analyses,
        total_tokens_used=int(analysis_stats.total_tokens),
        total_cost=float(analysis_stats.total_cost),
        running_analyses=running_analyses,
        status_breakdown=status_breakdown,
        llm_cache_hit_rate=llm_cache_hit_rate,
        llm_costs_by_model=llm_costs_by_model,
        llm_avg_latency_ms_by_provider=llm_avg_latency_ms_by_provider,
    )


@router.get(
    "/llm-cache/stats",
    response_model=LLMCacheStats,
    summary="Get LLM cache statistics",
    responses={
        200: {
            "description": "LLM cache hit/miss statistics",
            "content": {
                "application/json": {
                    "example": {
                        "hits": 15000,
                        "misses": 10000,
                        "hit_rate": 0.6,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def get_llm_cache_stats(
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
) -> LLMCacheStats:
    """
    Get LLM cache hit/miss statistics.

    **Admin-only endpoint** - requires admin authentication.

    The LLM cache stores responses for repeated prompts, reducing
    costs and latency for similar analyses.

    **Metrics:**
    - `hits`: Number of requests served from cache
    - `misses`: Number of requests that required LLM API call
    - `hit_rate`: Ratio of hits to total requests (0.0-1.0)

    **Target Hit Rate:**
    - < 0.2: Low efficiency, consider cache tuning
    - 0.2-0.4: Normal for diverse workloads
    - > 0.4: Good efficiency, significant cost savings

    **Cost Savings Estimate:**
    Cache hits typically save 90%+ of the cost compared to API calls.
    """
    stats = await llm_cache_service.get_cache_stats()
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    total = hits + misses
    hit_rate = float(hits / total) if total else 0.0
    return LLMCacheStats(hits=hits, misses=misses, hit_rate=hit_rate)


@router.delete(
    "/llm-cache",
    response_model=LLMCacheInvalidateResponse,
    summary="Invalidate LLM cache",
    responses={
        200: {
            "description": "Cache entries invalidated",
            "content": {
                "application/json": {
                    "examples": {
                        "all": {"value": {"deleted": 15000, "pattern": None}},
                        "pattern": {"value": {"deleted": 500, "pattern": "eda:*"}},
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def invalidate_llm_cache(
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
    pattern: Optional[str] = Query(None, description="Redis key pattern (e.g., 'eda:*')"),
) -> LLMCacheInvalidateResponse:
    """
    Invalidate LLM cache entries by pattern.

    **Admin-only endpoint** - requires admin authentication.

    Clears cached LLM responses, forcing fresh API calls for subsequent
    requests. Use this when:

    - LLM prompts have been updated
    - Cached responses are stale or incorrect
    - Testing new LLM configurations

    **Pattern Matching:**
    - `None` (no pattern): Clears ALL cache entries
    - `eda:*`: Clears EDA-related cache entries
    - `discovery:*`: Clears causal discovery cache
    - `*:gpt-4:*`: Clears all GPT-4 model responses

    **Warning:** Invalidating the entire cache will increase costs and
    latency until the cache is repopulated.

    **Example Patterns:**
    ```
    ?pattern=eda:*           # All EDA agent caches
    ?pattern=*:gpt-4:*       # All GPT-4 responses
    ?pattern=treatment:*     # Treatment effect caches
    ```
    """
    deleted = await llm_cache_service.invalidate_cache(pattern)
    return LLMCacheInvalidateResponse(deleted=deleted, pattern=pattern)


@router.get(
    "/llm-logs",
    response_model=LLMLogListResponse,
    summary="List LLM API call logs",
    responses={
        200: {
            "description": "Paginated LLM logs with cost aggregation",
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                                "provider": "openai",
                                "model": "gpt-4",
                                "prompt": "Analyze the following...",
                                "response": "Based on the data...",
                                "prompt_tokens": 500,
                                "completion_tokens": 200,
                                "total_tokens": 700,
                                "estimated_cost": 0.021,
                                "cache_hit": False,
                                "latency_ms": 1500,
                                "error": None,
                                "created_at": "2026-01-20T12:00:00Z",
                            }
                        ],
                        "total": 5000,
                        "skip": 0,
                        "limit": 100,
                        "total_cost": 125.50,
                        "costs_by_model": {
                            "gpt-4": 80.25,
                            "gpt-3.5-turbo": 25.00,
                            "claude-3-opus": 20.25,
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def list_llm_logs(
    db: Annotated[AsyncSession, Depends(get_async_session)],
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
    analysis_id: Optional[uuid.UUID] = Query(None, description="Filter by analysis ID"),
    provider: Optional[str] = Query(None, description="Filter by provider (openai, anthropic, vertex)"),
    start_date: Optional[datetime] = Query(None, description="Filter logs after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter logs before this date"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
) -> LLMLogListResponse:
    """
    List LLM API call logs with filtering and cost aggregation.

    **Admin-only endpoint** - requires admin authentication.

    Returns detailed logs of all LLM API calls, useful for:
    - Debugging analysis issues
    - Monitoring costs
    - Optimizing prompt efficiency
    - Auditing LLM usage

    **Filtering Options:**
    - `analysis_id`: View logs for specific analysis
    - `provider`: Filter by LLM provider (openai, anthropic, vertex)
    - `start_date` / `end_date`: Date range filter

    **Response includes:**
    - Individual log entries with full prompt/response
    - Token counts and costs
    - Cache hit status
    - Latency metrics
    - Error information if any
    - Aggregated costs by model

    **Cost Tracking:**
    The `costs_by_model` field shows total spending per model across
    the filtered results, helping identify cost optimization opportunities.
    """
    filters = []
    if analysis_id:
        filters.append(LLMLog.analysis_id == analysis_id)
    if provider:
        filters.append(LLMLog.provider == provider)
    if start_date:
        filters.append(LLMLog.created_at >= start_date)
    if end_date:
        filters.append(LLMLog.created_at <= end_date)

    query = select(LLMLog)
    if filters:
        query = query.where(*filters)
    query = query.order_by(LLMLog.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()

    count_query = select(func.count()).select_from(LLMLog)
    if filters:
        count_query = count_query.where(*filters)
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    cost_query = select(
        LLMLog.model,
        func.coalesce(func.sum(LLMLog.estimated_cost), 0).label("total_cost"),
    ).select_from(LLMLog)
    if filters:
        cost_query = cost_query.where(*filters)
    cost_query = cost_query.group_by(LLMLog.model)
    costs_result = await db.execute(cost_query)
    costs_by_model = {model: float(total_cost) for model, total_cost in costs_result.all()}

    total_cost_result = await db.execute(
        select(func.coalesce(func.sum(LLMLog.estimated_cost), 0)).select_from(LLMLog).where(*filters)
        if filters
        else select(func.coalesce(func.sum(LLMLog.estimated_cost), 0)).select_from(LLMLog)
    )
    total_cost = float(total_cost_result.scalar_one() or 0)

    items = [
        {
            "id": str(log.id),
            "analysis_id": str(log.analysis_id) if log.analysis_id else None,
            "provider": log.provider,
            "model": log.model,
            "prompt": log.prompt,
            "response": log.response,
            "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens,
            "total_tokens": log.total_tokens,
            "estimated_cost": float(log.estimated_cost),
            "cache_hit": log.cache_hit,
            "latency_ms": log.latency_ms,
            "error": log.error,
            "metadata": log.metadata,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]

    return LLMLogListResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
        total_cost=total_cost,
        costs_by_model=costs_by_model,
    )


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List all users",
    responses={
        200: {
            "description": "Paginated list of users with usage summaries",
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "email": "user@example.com",
                                "created_at": "2026-01-15T10:00:00Z",
                                "analysis_count": 45,
                                "total_tokens_used": 500000,
                                "total_cost": 12.50,
                            }
                        ],
                        "total": 150,
                        "skip": 0,
                        "limit": 50,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
    },
)
async def list_users(
    db: Annotated[AsyncSession, Depends(get_async_session)],
    _admin_id: Annotated[uuid.UUID, Depends(get_current_admin_user)],
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum records to return"),
) -> UserListResponse:
    """
    List all users with usage summaries.

    **Admin-only endpoint** - requires admin authentication.

    Returns paginated list of all registered users with aggregated
    usage statistics for each user.

    **User Information:**
    - User ID and email
    - Account creation date
    - Total analysis count
    - Total LLM tokens used
    - Total estimated cost

    **Sorting:** Users are ordered by creation date (newest first).

    **Use Cases:**
    - Monitor user onboarding
    - Identify high-usage users
    - Track platform growth
    - Cost attribution by user
    """
    # Get users with aggregated stats
    query = (
        select(
            User.id,
            User.email,
            User.created_at,
            func.count(Analysis.id).label("analysis_count"),
            func.coalesce(func.sum(Analysis.llm_tokens_used), 0).label("total_tokens"),
            func.coalesce(func.sum(Analysis.estimated_cost), 0).label("total_cost"),
        )
        .outerjoin(Analysis, User.id == Analysis.user_id)
        .group_by(User.id, User.email, User.created_at)
        .order_by(User.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    result = await db.execute(query)
    users = result.all()

    # Get total count
    count_result = await db.execute(select(func.count()).select_from(User))
    total = count_result.scalar_one()

    items = [
        {
            "id": str(u.id),
            "email": u.email,
            "created_at": u.created_at.isoformat(),
            "analysis_count": u.analysis_count,
            "total_tokens_used": int(u.total_tokens),
            "total_cost": float(u.total_cost),
        }
        for u in users
    ]

    return UserListResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
    )
