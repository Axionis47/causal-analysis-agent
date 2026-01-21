"""Celery tasks for analysis execution."""

import asyncio
import uuid
from datetime import datetime, timezone

# ============================================================================
# NetworkX compatibility patch for DoWhy
# DoWhy 0.11.1 uses nx.algorithms.d_separated which was removed in NetworkX 3.x
# The function was renamed to is_d_separator. This patch restores compatibility.
# ============================================================================
import networkx as nx

if not hasattr(nx.algorithms, "d_separated"):
    nx.algorithms.d_separated = lambda G, x, y, z: nx.is_d_separator(G, x, y, z)
    # Also add to nx namespace for direct access (dowhy.gcm.falsify uses nx.d_separated)
    if not hasattr(nx, "d_separated"):
        nx.d_separated = lambda G, x, y, z: nx.is_d_separator(G, x, y, z)
# ============================================================================

from celery.exceptions import SoftTimeLimitExceeded

from app.crud.analysis import analysis_crud
from app.core.logging import bind_contextvars, capture_exception, get_logger
from app.db.database import get_session_context
from app.models.analysis import AnalysisStatus
from app.orchestrator import mark_analysis_failed, run_causal_analysis
from app.orchestrator.checkpointer import get_checkpointer
from app.worker import celery_app

logger = get_logger(__name__)


async def _run_analysis(analysis_id: str, kaggle_url: str) -> None:
    """Run analysis work and update status fields."""
    start_time = datetime.now(timezone.utc)
    bind_contextvars(analysis_id=analysis_id)
    logger.info(
        "Analysis task started",
        analysis_id=analysis_id,
        kaggle_url=kaggle_url,
        start_time=start_time.isoformat(),
    )
    success = False
    try:
        analysis_uuid = uuid.UUID(analysis_id)
        async with get_session_context() as session:
            analysis = await analysis_crud.get(session, analysis_uuid)
            if analysis is None:
                logger.warning("Analysis not found", analysis_id=analysis_id)
                return
            if analysis.user_id is not None:
                bind_contextvars(user_id=str(analysis.user_id))
            await analysis_crud.update(
                session,
                db_obj=analysis,
                obj_in={
                    "status": AnalysisStatus.RUNNING,
                    "started_at": datetime.now(timezone.utc),
                },
            )
        logger.info(
            "Analysis execution started",
            analysis_id=analysis_id,
            kaggle_url=kaggle_url,
        )
        checkpointer = get_checkpointer()
        await run_causal_analysis(analysis_id, kaggle_url, checkpointer=checkpointer)
        async with get_session_context() as session:
            analysis = await analysis_crud.get(session, analysis_uuid)
            if analysis is not None:
                await analysis_crud.update(
                    session,
                    db_obj=analysis,
                    obj_in={
                        "status": AnalysisStatus.COMPLETED,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
        success = True
        logger.info("Analysis execution completed", analysis_id=analysis_id)
    except Exception as exc:
        message = (
            "Soft time limit exceeded"
            if isinstance(exc, SoftTimeLimitExceeded)
            else str(exc)
        )
        await mark_analysis_failed(analysis_id, message)
        capture_exception(exc)
        logger.exception(
            "Analysis execution failed",
            analysis_id=analysis_id,
            error=message,
        )
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        duration_seconds = (end_time - start_time).total_seconds()
        logger.info(
            "Analysis task finished",
            analysis_id=analysis_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration_seconds,
            success=success,
        )


@celery_app.task(
    bind=True,
    name="run_analysis_task",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def run_analysis_task(self, analysis_id: str, kaggle_url: str) -> None:
    """Celery task wrapper for running analyses."""
    asyncio.run(_run_analysis(analysis_id, kaggle_url))
