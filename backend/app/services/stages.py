"""Helpers for tracking analysis stage progress."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis_stage import AnalysisStage, StageStatus, StageType
from app.services.progress import ProgressEvent, publish_progress


async def _publish_progress_best_effort(event: ProgressEvent) -> None:
    # Database write must complete before Redis publish to ensure fallback consistency.
    try:
        await publish_progress(event)
    except Exception:  # noqa: BLE001 - best-effort progress updates
        return


async def start_stage(
    session: AsyncSession,
    analysis_id: str,
    stage: StageType,
    agent_name: str,
    message: str,
) -> AnalysisStage:
    analysis_uuid = uuid.UUID(analysis_id)
    result = await session.execute(
        select(AnalysisStage).where(
            AnalysisStage.analysis_id == analysis_uuid,
            AnalysisStage.stage == stage,
        )
    )
    stage_row = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if stage_row is None:
        stage_row = AnalysisStage(
            analysis_id=analysis_uuid,
            stage=stage,
            status=StageStatus.RUNNING,
            started_at=now,
            agent_name=agent_name,
            progress_percent=0,
            current_step=message,
        )
        session.add(stage_row)
    else:
        stage_row.status = StageStatus.RUNNING
        stage_row.started_at = now
        stage_row.current_step = message
        stage_row.agent_name = agent_name
    await session.flush()
    await _publish_progress_best_effort(
        ProgressEvent(
            analysis_id=str(analysis_id),
            stage=stage.value,
            status=StageStatus.RUNNING.value,
            progress_percent=stage_row.progress_percent,
            message=message,
        )
    )
    return stage_row


async def update_stage_progress(
    session: AsyncSession,
    stage_id: uuid.UUID,
    progress_percent: int,
    message: str,
) -> None:
    stage_row = await session.get(AnalysisStage, stage_id)
    if stage_row is None:
        return
    stage_row.progress_percent = progress_percent
    stage_row.current_step = message
    await session.flush()
    await _publish_progress_best_effort(
        ProgressEvent(
            analysis_id=str(stage_row.analysis_id),
            stage=stage_row.stage.value,
            status=stage_row.status.value,
            progress_percent=progress_percent,
            message=message,
        )
    )


async def complete_stage(
    session: AsyncSession,
    stage_id: uuid.UUID,
    outputs: dict,
    message: str,
) -> None:
    stage_row = await session.get(AnalysisStage, stage_id)
    if stage_row is None:
        return
    stage_row.status = StageStatus.COMPLETED
    stage_row.completed_at = datetime.now(timezone.utc)
    stage_row.progress_percent = 100
    stage_row.outputs = outputs
    stage_row.current_step = message
    await session.flush()
    await _publish_progress_best_effort(
        ProgressEvent(
            analysis_id=str(stage_row.analysis_id),
            stage=stage_row.stage.value,
            status=StageStatus.COMPLETED.value,
            progress_percent=100,
            message=message,
        )
    )


async def fail_stage(
    session: AsyncSession,
    stage_id: uuid.UUID,
    error: str,
    message: str,
    outputs: dict | None = None,
) -> None:
    stage_row = await session.get(AnalysisStage, stage_id)
    if stage_row is None:
        return
    stage_row.status = StageStatus.FAILED
    stage_row.completed_at = datetime.now(timezone.utc)
    stage_row.errors = {"error": error}
    stage_row.outputs = outputs or {}
    stage_row.current_step = message
    await session.flush()
    await _publish_progress_best_effort(
        ProgressEvent(
            analysis_id=str(stage_row.analysis_id),
            stage=stage_row.stage.value,
            status=StageStatus.FAILED.value,
            progress_percent=stage_row.progress_percent,
            message=message,
            error=error,
        )
    )
