"""Rate limiting service using Redis for state management."""

import logging
import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)

# Time windows in seconds
TIME_WINDOWS = {
    "hourly": 3600,
    "daily": 86400,
}

# Lua script for atomic check-and-increment operation
# Returns: [hourly_allowed, daily_allowed, hourly_count, daily_count, hourly_oldest, daily_oldest]
ATOMIC_RESERVE_SCRIPT = """
local hourly_key = KEYS[1]
local daily_key = KEYS[2]
local now = tonumber(ARGV[1])
local hourly_window = tonumber(ARGV[2])
local daily_window = tonumber(ARGV[3])
local hourly_limit = tonumber(ARGV[4])
local daily_limit = tonumber(ARGV[5])
local member = ARGV[6]

-- Clean up expired entries for hourly window
local hourly_cutoff = now - hourly_window
redis.call('ZREMRANGEBYSCORE', hourly_key, 0, hourly_cutoff)

-- Clean up expired entries for daily window
local daily_cutoff = now - daily_window
redis.call('ZREMRANGEBYSCORE', daily_key, 0, daily_cutoff)

-- Get current counts
local hourly_count = redis.call('ZCARD', hourly_key)
local daily_count = redis.call('ZCARD', daily_key)

-- Get oldest entries for reset time calculation
local hourly_oldest = redis.call('ZRANGE', hourly_key, 0, 0, 'WITHSCORES')
local daily_oldest = redis.call('ZRANGE', daily_key, 0, 0, 'WITHSCORES')

local hourly_oldest_ts = 0
local daily_oldest_ts = 0
if #hourly_oldest >= 2 then
    hourly_oldest_ts = tonumber(hourly_oldest[2])
end
if #daily_oldest >= 2 then
    daily_oldest_ts = tonumber(daily_oldest[2])
end

-- Check if both limits allow the request
local hourly_allowed = hourly_count < hourly_limit
local daily_allowed = daily_count < daily_limit

-- Only increment if both limits allow
if hourly_allowed and daily_allowed then
    redis.call('ZADD', hourly_key, now, member .. ':hourly')
    redis.call('ZADD', daily_key, now, member .. ':daily')
    redis.call('EXPIRE', hourly_key, hourly_window + 60)
    redis.call('EXPIRE', daily_key, daily_window + 60)
    hourly_count = hourly_count + 1
    daily_count = daily_count + 1
end

return {
    hourly_allowed and 1 or 0,
    daily_allowed and 1 or 0,
    hourly_count,
    daily_count,
    hourly_oldest_ts,
    daily_oldest_ts
}
"""


@dataclass
class ReserveResult:
    """Result of an atomic rate limit reservation."""
    allowed: bool
    exceeded_limit_type: Optional[str]  # "hourly", "daily", or None if allowed
    hourly_remaining: int
    daily_remaining: int
    hourly_reset: int
    daily_reset: int


class RateLimiter:
    """
    Rate limiter using Redis with sliding window algorithm.

    Uses Redis sorted sets to implement accurate sliding window rate limiting.
    Falls open (allows requests) on Redis failures to prevent service disruption.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis_url = redis_url or settings.RATE_LIMIT_STORAGE_URL
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def _get_key(self, user_id: UUID, limit_type: str) -> str:
        """Generate Redis key for rate limit tracking."""
        return f"rate_limit:{user_id}:{limit_type}"

    def _get_window_seconds(self, limit_type: str) -> int:
        """Get time window in seconds for limit type."""
        return TIME_WINDOWS.get(limit_type, 3600)

    def _get_limit(self, limit_type: str) -> int:
        """Get the limit value for a given limit type."""
        if limit_type == "hourly":
            return settings.RATE_LIMIT_ANALYSES_PER_HOUR
        elif limit_type == "daily":
            return settings.RATE_LIMIT_ANALYSES_PER_DAY
        return 10  # Default fallback

    async def check_rate_limit(self, user_id: UUID, limit_type: str) -> bool:
        """
        Check if user is within rate limits using sliding window algorithm.

        Args:
            user_id: The user's UUID
            limit_type: Type of limit ("hourly" or "daily")

        Returns:
            True if within limits, False if limit exceeded
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True

        try:
            redis_client = await self._get_redis()
            key = self._get_key(user_id, limit_type)
            window_seconds = self._get_window_seconds(limit_type)
            limit = self._get_limit(limit_type)

            now = time.time()
            window_start = now - window_seconds

            # Remove expired entries
            await redis_client.zremrangebyscore(key, 0, window_start)

            # Count current entries in window
            current_count = await redis_client.zcard(key)

            return current_count < limit

        except Exception as exc:
            logger.warning(
                "Redis rate limit check failed, allowing request",
                extra={"user_id": str(user_id), "limit_type": limit_type, "error": str(exc)},
            )
            # Fail open - allow request on Redis errors
            return True

    async def increment_counter(self, user_id: UUID, limit_type: str) -> None:
        """
        Increment the rate limit counter for a user.

        Args:
            user_id: The user's UUID
            limit_type: Type of limit ("hourly" or "daily")
        """
        if not settings.RATE_LIMIT_ENABLED:
            return

        try:
            redis_client = await self._get_redis()
            key = self._get_key(user_id, limit_type)
            window_seconds = self._get_window_seconds(limit_type)

            now = time.time()

            # Add current timestamp to sorted set
            # Use timestamp as both score and member (with random suffix for uniqueness)
            member = f"{now}:{user_id}"
            await redis_client.zadd(key, {member: now})

            # Set TTL to ensure cleanup
            await redis_client.expire(key, window_seconds + 60)

        except Exception as exc:
            logger.warning(
                "Redis rate limit increment failed",
                extra={"user_id": str(user_id), "limit_type": limit_type, "error": str(exc)},
            )

    async def get_remaining(self, user_id: UUID, limit_type: str) -> int:
        """
        Get remaining quota for a user.

        Args:
            user_id: The user's UUID
            limit_type: Type of limit ("hourly" or "daily")

        Returns:
            Number of remaining requests in the current window
        """
        if not settings.RATE_LIMIT_ENABLED:
            return self._get_limit(limit_type)

        try:
            redis_client = await self._get_redis()
            key = self._get_key(user_id, limit_type)
            window_seconds = self._get_window_seconds(limit_type)
            limit = self._get_limit(limit_type)

            now = time.time()
            window_start = now - window_seconds

            # Remove expired entries
            await redis_client.zremrangebyscore(key, 0, window_start)

            # Count current entries
            current_count = await redis_client.zcard(key)

            return max(0, limit - current_count)

        except Exception as exc:
            logger.warning(
                "Redis get_remaining failed",
                extra={"user_id": str(user_id), "limit_type": limit_type, "error": str(exc)},
            )
            return self._get_limit(limit_type)

    async def get_reset_time(self, user_id: UUID, limit_type: str) -> int:
        """
        Get timestamp when the rate limit window resets.

        Args:
            user_id: The user's UUID
            limit_type: Type of limit ("hourly" or "daily")

        Returns:
            Unix timestamp when the oldest entry expires
        """
        if not settings.RATE_LIMIT_ENABLED:
            return int(time.time() + self._get_window_seconds(limit_type))

        try:
            redis_client = await self._get_redis()
            key = self._get_key(user_id, limit_type)
            window_seconds = self._get_window_seconds(limit_type)

            # Get oldest entry in the window
            oldest = await redis_client.zrange(key, 0, 0, withscores=True)

            if oldest:
                oldest_timestamp = oldest[0][1]
                return int(oldest_timestamp + window_seconds)

            return int(time.time() + window_seconds)

        except Exception as exc:
            logger.warning(
                "Redis get_reset_time failed",
                extra={"user_id": str(user_id), "limit_type": limit_type, "error": str(exc)},
            )
            return int(time.time() + self._get_window_seconds(limit_type))

    async def reset_counter(self, user_id: UUID, limit_type: str) -> None:
        """
        Reset rate limits for a user (admin function).

        Args:
            user_id: The user's UUID
            limit_type: Type of limit ("hourly" or "daily")
        """
        try:
            redis_client = await self._get_redis()
            key = self._get_key(user_id, limit_type)
            await redis_client.delete(key)
            logger.info(
                "Rate limit reset",
                extra={"user_id": str(user_id), "limit_type": limit_type},
            )
        except Exception as exc:
            logger.error(
                "Redis rate limit reset failed",
                extra={"user_id": str(user_id), "limit_type": limit_type, "error": str(exc)},
            )

    async def reset_all_counters(self, user_id: UUID) -> None:
        """
        Reset all rate limits for a user (admin function).

        Args:
            user_id: The user's UUID
        """
        for limit_type in TIME_WINDOWS.keys():
            await self.reset_counter(user_id, limit_type)

    async def atomic_reserve(self, user_id: UUID) -> ReserveResult:
        """
        Atomically check and reserve quota for both hourly and daily limits.

        This method uses a Lua script to ensure the check and increment happen
        atomically, preventing race conditions where concurrent requests could
        exceed the quota.

        Args:
            user_id: The user's UUID

        Returns:
            ReserveResult with allowed status, exceeded limit type, and remaining quotas
        """
        if not settings.RATE_LIMIT_ENABLED:
            return ReserveResult(
                allowed=True,
                exceeded_limit_type=None,
                hourly_remaining=self._get_limit("hourly"),
                daily_remaining=self._get_limit("daily"),
                hourly_reset=int(time.time() + TIME_WINDOWS["hourly"]),
                daily_reset=int(time.time() + TIME_WINDOWS["daily"]),
            )

        try:
            redis_client = await self._get_redis()

            hourly_key = self._get_key(user_id, "hourly")
            daily_key = self._get_key(user_id, "daily")
            now = time.time()
            hourly_limit = self._get_limit("hourly")
            daily_limit = self._get_limit("daily")
            member = f"{now}:{user_id}"

            # Execute atomic Lua script
            result = await redis_client.eval(
                ATOMIC_RESERVE_SCRIPT,
                2,  # Number of keys
                hourly_key,
                daily_key,
                str(now),
                str(TIME_WINDOWS["hourly"]),
                str(TIME_WINDOWS["daily"]),
                str(hourly_limit),
                str(daily_limit),
                member,
            )

            hourly_allowed = bool(result[0])
            daily_allowed = bool(result[1])
            hourly_count = int(result[2])
            daily_count = int(result[3])
            hourly_oldest_ts = float(result[4]) if result[4] else 0
            daily_oldest_ts = float(result[5]) if result[5] else 0

            # Calculate reset times
            if hourly_oldest_ts > 0:
                hourly_reset = int(hourly_oldest_ts + TIME_WINDOWS["hourly"])
            else:
                hourly_reset = int(now + TIME_WINDOWS["hourly"])

            if daily_oldest_ts > 0:
                daily_reset = int(daily_oldest_ts + TIME_WINDOWS["daily"])
            else:
                daily_reset = int(now + TIME_WINDOWS["daily"])

            # Determine which limit was exceeded (if any)
            allowed = hourly_allowed and daily_allowed
            exceeded_limit_type = None
            if not allowed:
                # Prioritize hourly limit in error messages since it resets sooner
                if not hourly_allowed:
                    exceeded_limit_type = "hourly"
                elif not daily_allowed:
                    exceeded_limit_type = "daily"

            return ReserveResult(
                allowed=allowed,
                exceeded_limit_type=exceeded_limit_type,
                hourly_remaining=max(0, hourly_limit - hourly_count),
                daily_remaining=max(0, daily_limit - daily_count),
                hourly_reset=hourly_reset,
                daily_reset=daily_reset,
            )

        except Exception as exc:
            logger.warning(
                "Redis atomic_reserve failed, allowing request",
                extra={"user_id": str(user_id), "error": str(exc)},
            )
            # Fail open - allow request on Redis errors
            return ReserveResult(
                allowed=True,
                exceeded_limit_type=None,
                hourly_remaining=self._get_limit("hourly"),
                daily_remaining=self._get_limit("daily"),
                hourly_reset=int(time.time() + TIME_WINDOWS["hourly"]),
                daily_reset=int(time.time() + TIME_WINDOWS["daily"]),
            )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


# Singleton instance
rate_limiter = RateLimiter()
