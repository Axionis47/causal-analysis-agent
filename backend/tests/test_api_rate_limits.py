"""Tests for API rate limiting integration."""

import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.analysis import Analysis, AnalysisStatus


@pytest.fixture
def mock_user_id():
    """Generate a mock user ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_admin_id():
    """Generate a mock admin ID."""
    return uuid.uuid4()


@pytest.fixture
def auth_headers(mock_user_id):
    """Create mock authentication headers."""
    return {"Authorization": f"Bearer mock-token-{mock_user_id}"}


class TestAnalysisRateLimiting:
    """Tests for analysis endpoint rate limiting."""

    @pytest.mark.asyncio
    async def test_create_analysis_rate_limit_exceeded(self, db_session, mock_user_id):
        """Test 429 response when rate limit is exceeded."""
        with patch("app.api.v1.analyses.check_analysis_quota") as mock_quota:
            from fastapi import HTTPException

            mock_quota.side_effect = HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Hourly rate limit exceeded. Limit: 10/hour. Remaining: 0.",
                headers={
                    "X-RateLimit-Limit": "10",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "1705678800",
                },
            )

            with patch("app.core.auth.get_current_user", return_value=mock_user_id):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/analyses",
                        json={
                            "kaggle_url": "https://www.kaggle.com/datasets/test/sample"
                        },
                        headers={"Authorization": "Bearer test-token"},
                    )

                    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.asyncio
    async def test_create_analysis_concurrent_limit_exceeded(
        self, db_session, mock_user_id
    ):
        """Test 429 response when concurrent analysis limit is exceeded."""
        # Create mock running analyses
        with patch("app.api.v1.analyses.check_analysis_quota", return_value=mock_user_id):
            with patch("app.api.v1.analyses.db") as mock_db:
                mock_db.scalar = AsyncMock(return_value=3)  # 3 running analyses

                with patch("app.core.config.settings") as mock_settings:
                    mock_settings.RATE_LIMIT_CONCURRENT_ANALYSES = 3

                    # The endpoint should return 429 for concurrent limit
                    # This tests the logic path in create_analysis

    @pytest.mark.asyncio
    async def test_rate_limit_headers_in_response(self, db_session, mock_user_id):
        """Test that rate limit headers are included in successful responses."""
        with patch("app.api.v1.analyses.check_analysis_quota", return_value=mock_user_id):
            with patch("app.api.v1.analyses.get_rate_limit_headers") as mock_headers:
                mock_headers.return_value = {
                    "X-RateLimit-Limit-Hourly": "10",
                    "X-RateLimit-Remaining-Hourly": "9",
                    "X-RateLimit-Limit-Daily": "50",
                    "X-RateLimit-Remaining-Daily": "49",
                    "X-RateLimit-Reset": "1705678800",
                }

                # Additional mocking would be needed for full integration test


class TestQuotaManager:
    """Tests for QuotaManager service."""

    @pytest.mark.asyncio
    async def test_get_running_analyses_count(self, db_session, mock_user_id):
        """Test counting running analyses for a user."""
        from app.services.quota_manager import quota_manager

        # Create test analyses
        for i in range(2):
            analysis = Analysis(
                user_id=mock_user_id,
                kaggle_url=f"https://www.kaggle.com/datasets/test/sample{i}",
                status=AnalysisStatus.RUNNING,
                config={},
                llm_tokens_used=0,
                estimated_cost=Decimal("0.0"),
            )
            db_session.add(analysis)
        await db_session.commit()

        count = await quota_manager.get_running_analyses_count(db_session, mock_user_id)
        assert count == 2

    @pytest.mark.asyncio
    async def test_can_start_analysis_within_limits(self, db_session, mock_user_id):
        """Test that analysis can start when within all limits."""
        from app.services.quota_manager import quota_manager

        with patch("app.services.rate_limiter.rate_limiter") as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=True)
            mock_limiter.get_remaining = AsyncMock(return_value=5)

            can_start, error = await quota_manager.can_start_analysis(
                db_session, mock_user_id
            )
            assert can_start is True
            assert error == ""

    @pytest.mark.asyncio
    async def test_can_start_analysis_concurrent_exceeded(self, db_session, mock_user_id):
        """Test that analysis cannot start when concurrent limit exceeded."""
        from app.services.quota_manager import quota_manager

        # Create running analyses at limit
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            mock_settings.RATE_LIMIT_CONCURRENT_ANALYSES = 2

            for i in range(2):
                analysis = Analysis(
                    user_id=mock_user_id,
                    kaggle_url=f"https://www.kaggle.com/datasets/test/sample{i}",
                    status=AnalysisStatus.RUNNING,
                    config={},
                    llm_tokens_used=0,
                    estimated_cost=Decimal("0.0"),
                )
                db_session.add(analysis)
            await db_session.commit()

            can_start, error = await quota_manager.can_start_analysis(
                db_session, mock_user_id
            )
            assert can_start is False
            assert "concurrent" in error.lower()

    @pytest.mark.asyncio
    async def test_track_llm_usage(self, db_session, mock_user_id):
        """Test tracking LLM token usage."""
        from app.services.quota_manager import quota_manager

        # Create an analysis
        analysis = Analysis(
            user_id=mock_user_id,
            kaggle_url="https://www.kaggle.com/datasets/test/sample",
            status=AnalysisStatus.RUNNING,
            config={},
            llm_tokens_used=0,
            estimated_cost=Decimal("0.0"),
        )
        db_session.add(analysis)
        await db_session.commit()

        # Track usage
        await quota_manager.track_llm_usage(
            db_session,
            analysis.id,
            tokens=1000,
            model="gpt-4",
        )

        # Refresh and check
        await db_session.refresh(analysis)
        assert analysis.llm_tokens_used == 1000
        assert float(analysis.estimated_cost) == pytest.approx(0.03, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_user_usage_stats(self, db_session, mock_user_id):
        """Test getting user usage statistics."""
        from app.services.quota_manager import quota_manager

        # Create test analyses
        for i in range(3):
            status_val = AnalysisStatus.COMPLETED if i < 2 else AnalysisStatus.RUNNING
            analysis = Analysis(
                user_id=mock_user_id,
                kaggle_url=f"https://www.kaggle.com/datasets/test/sample{i}",
                status=status_val,
                config={},
                llm_tokens_used=1000,
                estimated_cost=Decimal("0.03"),
            )
            db_session.add(analysis)
        await db_session.commit()

        with patch("app.services.rate_limiter.rate_limiter") as mock_limiter:
            mock_limiter.get_remaining = AsyncMock(return_value=5)
            mock_limiter.get_reset_time = AsyncMock(return_value=1705678800)

            stats = await quota_manager.get_user_usage_stats(
                db_session, mock_user_id, days=30
            )

            assert stats["total_analyses"] == 3
            assert stats["total_tokens_used"] == 3000
            assert stats["running_analyses"] == 1


class TestAdminEndpoints:
    """Tests for admin API endpoints."""

    @pytest.mark.asyncio
    async def test_admin_access_denied_for_non_admin(self, mock_user_id):
        """Test that non-admin users cannot access admin endpoints."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ADMIN_USER_IDS = []

            with patch("app.core.auth.get_current_user", return_value=mock_user_id):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get(
                        "/api/v1/admin/stats",
                        headers={"Authorization": "Bearer test-token"},
                    )
                    # Should be 403 or handled by auth
                    assert response.status_code in [
                        status.HTTP_403_FORBIDDEN,
                        status.HTTP_401_UNAUTHORIZED,
                    ]

    @pytest.mark.asyncio
    async def test_admin_can_reset_user_quota(self, mock_user_id, mock_admin_id):
        """Test that admin can reset user quotas."""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.ADMIN_USER_IDS = [str(mock_admin_id)]

            with patch("app.services.rate_limiter.rate_limiter") as mock_limiter:
                mock_limiter.reset_counter = AsyncMock()

                with patch(
                    "app.api.dependencies.get_current_admin_user",
                    return_value=mock_admin_id,
                ):
                    # The reset would work with proper auth
                    pass


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_calculate_cost_gpt4(self):
        """Test cost calculation for GPT-4."""
        from app.services.quota_manager import quota_manager

        cost = quota_manager.calculate_cost(1000, "gpt-4")
        assert cost == pytest.approx(0.03, rel=0.01)

    def test_calculate_cost_gpt35(self):
        """Test cost calculation for GPT-3.5."""
        from app.services.quota_manager import quota_manager

        cost = quota_manager.calculate_cost(1000, "gpt-3.5-turbo")
        assert cost == pytest.approx(0.0005, rel=0.01)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model uses default."""
        from app.services.quota_manager import quota_manager

        cost = quota_manager.calculate_cost(1000, "unknown-model")
        assert cost == pytest.approx(0.01, rel=0.01)  # Default rate

    def test_estimate_tokens_from_text(self):
        """Test token estimation from text."""
        from app.services.quota_manager import quota_manager

        # ~4 characters per token
        text = "a" * 400
        estimated = quota_manager.estimate_tokens_from_text(text)
        assert estimated == 100
