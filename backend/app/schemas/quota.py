"""Pydantic schemas for quota and rate limiting responses."""

import uuid
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class QuotaUsage(BaseModel):
    """Current quota usage for a user."""

    model_config = ConfigDict(from_attributes=True)

    analyses_this_hour: int
    analyses_this_day: int
    running_analyses: int
    total_tokens_used: int
    total_cost: float
    limits: dict[str, int]


class RateLimitInfo(BaseModel):
    """Rate limit information for a specific window."""

    limit: int
    remaining: int
    reset_time: int


class ConcurrentLimitInfo(BaseModel):
    """Concurrent analysis limit information."""

    limit: int
    current: int


class RateLimits(BaseModel):
    """All rate limit information."""

    hourly: RateLimitInfo
    daily: RateLimitInfo
    concurrent: ConcurrentLimitInfo


class UserUsageStats(BaseModel):
    """Detailed usage statistics for a user."""

    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    period_days: int
    total_analyses: int
    total_tokens_used: int
    total_cost: float
    running_analyses: int
    status_breakdown: dict[str, int]
    rate_limits: dict[str, Any]


class AnalysisCostSummary(BaseModel):
    """Cost summary for an analysis."""

    id: str
    kaggle_url: str
    status: str
    llm_tokens_used: int
    estimated_cost: float
    created_at: str
    completed_at: Optional[str] = None


class UserAnalysisListResponse(BaseModel):
    """Paginated list of analyses with cost information."""

    items: list[dict[str, Any]]
    total: int
    skip: int
    limit: int


class QuotaResetResponse(BaseModel):
    """Response for quota reset operation."""

    user_id: uuid.UUID
    reset_types: list[str]
    message: str


class AdminStats(BaseModel):
    """Aggregate statistics for admin dashboard."""

    total_users: int
    total_analyses: int
    total_tokens_used: int
    total_cost: float
    running_analyses: int
    status_breakdown: dict[str, int]
    llm_cache_hit_rate: float
    llm_costs_by_model: dict[str, float]
    llm_avg_latency_ms_by_provider: dict[str, float]


class UserSummary(BaseModel):
    """Summary information for a user."""

    id: str
    email: str
    created_at: str
    analysis_count: int
    total_tokens_used: int
    total_cost: float


class UserListResponse(BaseModel):
    """Paginated list of users with usage summaries."""

    items: list[dict[str, Any]]
    total: int
    skip: int
    limit: int


class LLMCacheStats(BaseModel):
    """LLM cache statistics."""

    hits: int
    misses: int
    hit_rate: float


class LLMCacheInvalidateResponse(BaseModel):
    """Response for cache invalidation."""

    deleted: int
    pattern: Optional[str] = None


class LLMLogItem(BaseModel):
    """Single LLM log entry."""

    id: str
    analysis_id: Optional[str]
    provider: str
    model: str
    prompt: str
    response: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    estimated_cost: float
    cache_hit: bool
    latency_ms: Optional[int]
    error: Optional[str]
    metadata: dict[str, Any]
    created_at: str


class LLMLogListResponse(BaseModel):
    """Paginated list of LLM logs with aggregations."""

    items: list[LLMLogItem]
    total: int
    skip: int
    limit: int
    total_cost: float
    costs_by_model: dict[str, float]


class RateLimitHeaders(BaseModel):
    """Rate limit header values."""

    x_ratelimit_limit_hourly: int
    x_ratelimit_remaining_hourly: int
    x_ratelimit_limit_daily: int
    x_ratelimit_remaining_daily: int
    x_ratelimit_reset: int
