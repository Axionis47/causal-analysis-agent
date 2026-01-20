"""Tests for rate limiting service."""

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.rate_limiter import RateLimiter, TIME_WINDOWS


@pytest.fixture
def rate_limiter():
    """Create a rate limiter instance for testing."""
    return RateLimiter(redis_url="redis://localhost:6379/15")


@pytest.fixture
def user_id():
    """Generate a random user ID for testing."""
    return uuid.uuid4()


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_get_key_format(self, rate_limiter, user_id):
        """Test that Redis keys are formatted correctly."""
        key = rate_limiter._get_key(user_id, "hourly")
        assert key == f"rate_limit:{user_id}:hourly"

    def test_get_window_seconds(self, rate_limiter):
        """Test window duration lookup."""
        assert rate_limiter._get_window_seconds("hourly") == 3600
        assert rate_limiter._get_window_seconds("daily") == 86400
        assert rate_limiter._get_window_seconds("unknown") == 3600  # Default

    @pytest.mark.asyncio
    async def test_check_rate_limit_when_disabled(self, rate_limiter, user_id):
        """Test that rate limiting returns True when disabled."""
        with patch("app.services.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = False
            result = await rate_limiter.check_rate_limit(user_id, "hourly")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limits(self, rate_limiter, user_id):
        """Test rate limit check when user is within limits."""
        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zcard = AsyncMock(return_value=5)  # Under limit

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            with patch("app.services.rate_limiter.settings") as mock_settings:
                mock_settings.RATE_LIMIT_ENABLED = True
                mock_settings.RATE_LIMIT_ANALYSES_PER_HOUR = 10
                result = await rate_limiter.check_rate_limit(user_id, "hourly")
                assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter, user_id):
        """Test rate limit check when user exceeds limits."""
        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zcard = AsyncMock(return_value=10)  # At limit

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            with patch("app.services.rate_limiter.settings") as mock_settings:
                mock_settings.RATE_LIMIT_ENABLED = True
                mock_settings.RATE_LIMIT_ANALYSES_PER_HOUR = 10
                result = await rate_limiter.check_rate_limit(user_id, "hourly")
                assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_failure(self, rate_limiter, user_id):
        """Test that rate limiting fails open on Redis errors."""
        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock(side_effect=Exception("Redis error"))

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            with patch("app.services.rate_limiter.settings") as mock_settings:
                mock_settings.RATE_LIMIT_ENABLED = True
                result = await rate_limiter.check_rate_limit(user_id, "hourly")
                # Should fail open (allow request)
                assert result is True

    @pytest.mark.asyncio
    async def test_increment_counter(self, rate_limiter, user_id):
        """Test incrementing the rate limit counter."""
        mock_redis = AsyncMock()
        mock_redis.zadd = AsyncMock()
        mock_redis.expire = AsyncMock()

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            with patch("app.services.rate_limiter.settings") as mock_settings:
                mock_settings.RATE_LIMIT_ENABLED = True
                await rate_limiter.increment_counter(user_id, "hourly")

                # Verify zadd was called
                mock_redis.zadd.assert_called_once()
                mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_remaining(self, rate_limiter, user_id):
        """Test getting remaining quota."""
        mock_redis = AsyncMock()
        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zcard = AsyncMock(return_value=3)

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            with patch("app.services.rate_limiter.settings") as mock_settings:
                mock_settings.RATE_LIMIT_ENABLED = True
                mock_settings.RATE_LIMIT_ANALYSES_PER_HOUR = 10
                remaining = await rate_limiter.get_remaining(user_id, "hourly")
                assert remaining == 7  # 10 - 3

    @pytest.mark.asyncio
    async def test_get_reset_time(self, rate_limiter, user_id):
        """Test getting reset time."""
        now = time.time()
        mock_redis = AsyncMock()
        mock_redis.zrange = AsyncMock(return_value=[("entry", now - 1000)])

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            with patch("app.services.rate_limiter.settings") as mock_settings:
                mock_settings.RATE_LIMIT_ENABLED = True
                reset_time = await rate_limiter.get_reset_time(user_id, "hourly")
                # Reset time should be oldest timestamp + window
                expected = int((now - 1000) + 3600)
                assert reset_time == expected

    @pytest.mark.asyncio
    async def test_reset_counter(self, rate_limiter, user_id):
        """Test resetting rate limit counter."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            await rate_limiter.reset_counter(user_id, "hourly")
            mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_all_counters(self, rate_limiter, user_id):
        """Test resetting all rate limit counters for a user."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()

        with patch.object(rate_limiter, "_get_redis", return_value=mock_redis):
            await rate_limiter.reset_all_counters(user_id)
            # Should be called for both hourly and daily
            assert mock_redis.delete.call_count == 2


class TestTimeWindows:
    """Tests for time window constants."""

    def test_hourly_window(self):
        """Test hourly window is 1 hour."""
        assert TIME_WINDOWS["hourly"] == 3600

    def test_daily_window(self):
        """Test daily window is 24 hours."""
        assert TIME_WINDOWS["daily"] == 86400
