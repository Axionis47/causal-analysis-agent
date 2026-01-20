"""Publish and stream analysis progress updates."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import AsyncIterator

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_session_context
from app.models.analysis_stage import AnalysisStage, StageStatus, StageType

logger = get_logger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_stream_client_counts: Counter[str] = Counter()
_stream_client_lock = asyncio.Lock()


@dataclass
class ProgressEvent:
    analysis_id: str
    stage: str
    status: str
    progress_percent: int
    message: str
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


async def get_redis() -> Redis:
    return Redis.from_url(settings.REDIS_URL, decode_responses=True)


async def publish_progress(event: ProgressEvent) -> None:
    try:
        await _upsert_stage_progress(event)
    except Exception:  # noqa: BLE001 - best-effort progress updates
        return
    try:
        redis = await get_redis()
        channel = _channel(event.analysis_id)
        await redis.publish(channel, event.to_json())
        await redis.close()
    except Exception:  # noqa: BLE001 - best-effort progress updates
        return


async def stream_progress(analysis_id: str) -> AsyncIterator[ProgressEvent]:
    last_event_hash: int | None = None
    client_count = await _increment_stream_clients(analysis_id)
    logger.debug(
        "Progress stream client connected",
        analysis_id=analysis_id,
        client_count=client_count,
    )
    channel = _channel(analysis_id)
    redis = None
    pubsub = None
    stream_started_at = time.monotonic()
    stream_mode = "redis"
    try:
        try:
            redis = await get_redis()
            pubsub = redis.pubsub()
            await pubsub.subscribe(channel)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                if not data:
                    continue
                payload = json.loads(data)
                event = ProgressEvent(**payload)
                event_hash = _event_hash(event)
                if event_hash == last_event_hash:
                    continue
                last_event_hash = event_hash
                yield event
                if event.status in TERMINAL_STATUSES:
                    break
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning(
                "Redis progress stream unavailable, falling back to database",
                exc_info=exc,
                analysis_id=analysis_id,
                stream_mode="redis",
            )
            if not settings.PROGRESS_FALLBACK_TO_DB:
                raise
            stream_mode = "database"
            async for event in _stream_from_database(
                analysis_id=analysis_id,
                poll_interval=settings.PROGRESS_POLL_INTERVAL,
                max_poll_interval=settings.PROGRESS_MAX_POLL_INTERVAL,
                timeout_seconds=settings.PROGRESS_STREAM_TIMEOUT,
            ):
                event_hash = _event_hash(event)
                if event_hash == last_event_hash:
                    continue
                last_event_hash = event_hash
                yield event
    finally:
        stream_duration = time.monotonic() - stream_started_at
        logger.info(
            "Progress stream mode ended",
            analysis_id=analysis_id,
            stream_mode=stream_mode,
            duration_seconds=stream_duration,
        )
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:  # noqa: BLE001 - cleanup only
                pass
            try:
                await pubsub.close()
            except Exception:  # noqa: BLE001 - cleanup only
                pass
        if redis is not None:
            try:
                await redis.close()
            except Exception:  # noqa: BLE001 - cleanup only
                pass
        client_count = await _decrement_stream_clients(analysis_id)
        logger.debug(
            "Progress stream client disconnected",
            analysis_id=analysis_id,
            client_count=client_count,
        )


def _channel(analysis_id: str) -> str:
    return f"analysis:{analysis_id}:progress"


def _event_hash(event: ProgressEvent) -> int:
    return hash(
        (
            event.stage,
            event.status,
            event.progress_percent,
            event.message,
            event.error,
        )
    )


async def _upsert_stage_progress(event: ProgressEvent) -> None:
    analysis_uuid = uuid.UUID(event.analysis_id)
    try:
        stage_type = StageType(event.stage)
        stage_status = StageStatus(event.status)
    except ValueError as exc:
        logger.warning(
            "Progress event has invalid stage or status",
            exc_info=exc,
            analysis_id=event.analysis_id,
            stage=event.stage,
            status=event.status,
        )
        raise

    errors: dict[str, str] = {"error": event.error} if event.error else {}
    now = datetime.now(timezone.utc)
    async with get_session_context() as session:
        result = await session.execute(
            select(AnalysisStage)
            .where(
                AnalysisStage.analysis_id == analysis_uuid,
                AnalysisStage.stage == stage_type,
            )
            .order_by(AnalysisStage.updated_at.desc())
            .limit(1)
        )
        stage_row = result.scalar_one_or_none()
        if stage_row is None:
            stage_row = AnalysisStage(
                analysis_id=analysis_uuid,
                stage=stage_type,
                status=stage_status,
                progress_percent=event.progress_percent,
                current_step=event.message,
                errors=errors,
            )
            if event.status in TERMINAL_STATUSES:
                stage_row.completed_at = now
            session.add(stage_row)
        else:
            stage_row.status = stage_status
            stage_row.progress_percent = event.progress_percent
            stage_row.current_step = event.message
            stage_row.errors = errors
            if event.status in TERMINAL_STATUSES:
                stage_row.completed_at = now
        await session.flush()


def _event_from_stage(stage_row: AnalysisStage) -> ProgressEvent:
    error_message = None
    if stage_row.errors:
        error_message = stage_row.errors.get("error")
    return ProgressEvent(
        analysis_id=str(stage_row.analysis_id),
        stage=stage_row.stage.value,
        status=stage_row.status.value,
        progress_percent=stage_row.progress_percent,
        message=stage_row.current_step or "",
        error=error_message,
    )


async def _stream_from_database(
    analysis_id: str,
    poll_interval: float = 2.0,
    max_poll_interval: float = 10.0,
    timeout_seconds: int = 300,
) -> AsyncIterator[ProgressEvent]:
    analysis_uuid = uuid.UUID(analysis_id)
    start_time = time.monotonic()
    deadline = start_time + timeout_seconds
    interval = poll_interval
    last_seen_updated_at = None
    last_event_hash: int | None = None
    retry_attempts = 0

    logger.info(
        "Progress stream database polling started",
        analysis_id=analysis_id,
        stream_mode="database",
    )
    try:
        while True:
            if time.monotonic() >= deadline:
                logger.info(
                    "Progress stream database polling timed out",
                    analysis_id=analysis_id,
                    stream_mode="database",
                )
                break
            try:
                async with get_session_context() as session:
                    result = await session.execute(
                        select(AnalysisStage)
                        .where(AnalysisStage.analysis_id == analysis_uuid)
                        .order_by(AnalysisStage.updated_at.desc())
                        .limit(1)
                    )
                    stage_row = result.scalar_one_or_none()
                retry_attempts = 0
            except SQLAlchemyError as exc:
                retry_attempts += 1
                logger.warning(
                    "Progress stream database polling error",
                    exc_info=exc,
                    analysis_id=analysis_id,
                    stream_mode="database",
                    attempt=retry_attempts,
                )
                if retry_attempts >= 3:
                    logger.error(
                        "Progress stream database polling exceeded retry limit",
                        analysis_id=analysis_id,
                        stream_mode="database",
                    )
                    break
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_poll_interval)
                continue

            if stage_row is not None and stage_row.updated_at != last_seen_updated_at:
                last_seen_updated_at = stage_row.updated_at
                event = _event_from_stage(stage_row)
                event_hash = _event_hash(event)
                if event_hash != last_event_hash:
                    last_event_hash = event_hash
                    yield event
                    if event.status in TERMINAL_STATUSES:
                        break

            await asyncio.sleep(interval)
            interval = min(interval * 1.5, max_poll_interval)
    finally:
        duration = time.monotonic() - start_time
        logger.info(
            "Progress stream database polling stopped",
            analysis_id=analysis_id,
            stream_mode="database",
            duration_seconds=duration,
        )


async def _increment_stream_clients(analysis_id: str) -> int:
    async with _stream_client_lock:
        _stream_client_counts[analysis_id] += 1
        return _stream_client_counts[analysis_id]


async def _decrement_stream_clients(analysis_id: str) -> int:
    async with _stream_client_lock:
        _stream_client_counts[analysis_id] -= 1
        if _stream_client_counts[analysis_id] <= 0:
            _stream_client_counts.pop(analysis_id, None)
            return 0
        return _stream_client_counts[analysis_id]
