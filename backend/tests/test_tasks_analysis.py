"""Tests for analysis Celery task behavior."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.crud.analysis import analysis_crud
from app.models.analysis import AnalysisStatus
from app.tasks.analysis import _run_analysis, run_analysis_task


def make_session_context(db_session):
    """Create a session context manager for patching."""

    @asynccontextmanager
    async def _context():
        yield db_session

    return _context


@pytest.mark.asyncio
async def test_run_analysis_task_success(db_session):
    """Task updates status from pending to completed."""
    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "kaggle_url": "https://www.kaggle.com/datasets/test/success",
            "config": {},
        },
    )

    with patch("app.tasks.analysis.get_session_context", make_session_context(db_session)):
        with patch("app.tasks.analysis.run_causal_analysis", new=AsyncMock()):
            await _run_analysis(str(analysis.id), analysis.kaggle_url)

    updated = await analysis_crud.get(db_session, analysis.id)
    assert updated is not None
    assert updated.status == AnalysisStatus.COMPLETED
    assert updated.started_at is not None
    assert updated.completed_at is not None


@pytest.mark.asyncio
async def test_run_analysis_task_failure(db_session):
    """Task updates status to failed when an exception occurs."""
    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "kaggle_url": "https://www.kaggle.com/datasets/test/failure",
            "config": {},
        },
    )

    with patch("app.tasks.analysis.get_session_context", make_session_context(db_session)):
        with patch("app.orchestrator.get_session_context", make_session_context(db_session)):
            with patch(
                "app.tasks.analysis.run_causal_analysis",
                side_effect=RuntimeError("boom"),
            ):
                with pytest.raises(RuntimeError):
                    await _run_analysis(str(analysis.id), analysis.kaggle_url)

    updated = await analysis_crud.get(db_session, analysis.id)
    assert updated is not None
    assert updated.status == AnalysisStatus.FAILED
    assert updated.error_message is not None
    assert "boom" in updated.error_message


def test_run_analysis_task_retry_config():
    """Task is configured with retry settings."""
    assert run_analysis_task.autoretry_for == (Exception,)
    assert run_analysis_task.retry_kwargs["max_retries"] == 3
    assert run_analysis_task.retry_backoff is True
    assert run_analysis_task.retry_backoff_max == 600
    assert run_analysis_task.retry_jitter is True
