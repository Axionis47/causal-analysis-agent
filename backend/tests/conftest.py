"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.models import (
    AnalysisStatus,
    DiscoveryMethod,
    TreatmentMethod,
    ValidationType,
)
from tests.fixtures.agent_fixtures import (  # noqa: F401
    mock_circuit_breaker,
    mock_gcs_client,
    mock_kaggle_api,
    mock_llm_router,
    sample_dataframe,
    sample_state,
)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session for testing."""
    async_session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_factory() as session:
        yield session


@pytest.fixture
def sample_analysis_data() -> dict:
    """Sample data for creating an Analysis."""
    return {
        "kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset",
        "status": AnalysisStatus.PENDING,
        "config": {"treatment": "variable_a", "outcome": "variable_b"},
    }


@pytest.fixture
def sample_dataset_data() -> dict:
    """Sample data for creating a Dataset."""
    return {
        "kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset",
        "files": [{"name": "data.csv", "size": 1024, "format": "csv"}],
        "metadata": {"description": "Test dataset"},
        "characteristics": {"row_count": 1000, "column_count": 10},
    }


@pytest.fixture
def sample_causal_graph_data() -> dict:
    """Sample data for creating a CausalGraph."""
    return {
        "method": DiscoveryMethod.PC,
        "edges": [{"source": "A", "target": "B", "weight": 0.8}],
        "nodes": [{"id": "A", "label": "Variable A"}, {"id": "B", "label": "Variable B"}],
        "graph_data": {},
        "algorithm_params": {"alpha": 0.05},
        "confidence": 0.85,
    }


@pytest.fixture
def sample_treatment_effect_data() -> dict:
    """Sample data for creating a TreatmentEffect."""
    return {
        "treatment_variable": "treatment",
        "outcome_variable": "outcome",
        "method": TreatmentMethod.PROPENSITY_MATCHING,
        "ate": 0.25,
        "ate_ci_lower": 0.15,
        "ate_ci_upper": 0.35,
        "confidence_interval": {"lower": 0.15, "upper": 0.35},
        "confounders_adjusted": ["age", "gender"],
        "sample_size": {"treatment": 500, "control": 500},
        "assumptions_checked": {"overlap": True, "balance": True},
    }


@pytest.fixture
def sample_validation_result_data() -> dict:
    """Sample data for creating a ValidationResult."""
    return {
        "validation_type": ValidationType.REFUTATION,
        "method": "placebo_treatment",
        "passed": True,
        "confidence_score": 0.95,
        "details": {"p_value": 0.8, "effect_size": 0.01},
        "recommendations": [],
    }


# Rate limiting fixtures


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter for testing."""
    from unittest.mock import AsyncMock, MagicMock

    limiter = MagicMock()
    limiter.check_rate_limit = AsyncMock(return_value=True)
    limiter.increment_counter = AsyncMock()
    limiter.get_remaining = AsyncMock(return_value=10)
    limiter.get_reset_time = AsyncMock(return_value=1705678800)
    limiter.reset_counter = AsyncMock()
    return limiter


@pytest.fixture
def mock_quota_manager():
    """Create a mock quota manager for testing."""
    from unittest.mock import AsyncMock, MagicMock

    manager = MagicMock()
    manager.get_running_analyses_count = AsyncMock(return_value=0)
    manager.can_start_analysis = AsyncMock(return_value=(True, ""))
    manager.record_analysis_started = AsyncMock()
    manager.get_user_usage_stats = AsyncMock(
        return_value={
            "period_days": 30,
            "total_analyses": 10,
            "total_tokens_used": 50000,
            "total_cost": 1.50,
            "running_analyses": 1,
            "status_breakdown": {"completed": 8, "failed": 1, "running": 1},
            "rate_limits": {
                "hourly": {"limit": 10, "remaining": 5, "reset_time": 1705678800},
                "daily": {"limit": 50, "remaining": 40, "reset_time": 1705750800},
                "concurrent": {"limit": 3, "current": 1},
            },
        }
    )
    manager.track_llm_usage = AsyncMock()
    return manager


@pytest.fixture
def admin_user_id():
    """Generate an admin user ID for testing."""
    import uuid

    return uuid.uuid4()


@pytest.fixture
def regular_user_id():
    """Generate a regular (non-admin) user ID for testing."""
    import uuid

    return uuid.uuid4()


@pytest.fixture
def mock_admin_settings(admin_user_id):
    """Create mock settings with admin user configured."""
    from unittest.mock import patch

    with patch("app.core.config.settings") as mock_settings:
        mock_settings.ADMIN_USER_IDS = [str(admin_user_id)]
        mock_settings.RATE_LIMIT_ENABLED = True
        mock_settings.RATE_LIMIT_ANALYSES_PER_HOUR = 10
        mock_settings.RATE_LIMIT_ANALYSES_PER_DAY = 50
        mock_settings.RATE_LIMIT_CONCURRENT_ANALYSES = 3
        yield mock_settings
