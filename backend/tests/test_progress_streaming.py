"""Tests for progress streaming fallback and SSE behavior."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.v1 import analyses as analyses_api
from app.models.analysis import Analysis, AnalysisStatus
from app.models.analysis_stage import AnalysisStage, StageStatus, StageType
from app.services import progress as progress_service


class FakePubSub:
    def __init__(self, messages, subscribe_error=None):
        self._messages = messages
        self._subscribe_error = subscribe_error

    async def subscribe(self, channel):  # noqa: ARG002 - test stub
        if self._subscribe_error:
            raise self._subscribe_error

    async def unsubscribe(self, channel):  # noqa: ARG002 - test stub
        return

    async def close(self):
        return

    async def listen(self):
        for message in self._messages:
            yield message


class FakeRedis:
    def __init__(self, pubsub):
        self._pubsub = pubsub

    def pubsub(self):
        return self._pubsub

    async def close(self):
        return


async def _create_analysis(db_session) -> Analysis:
    analysis = Analysis(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        kaggle_url="https://www.kaggle.com/datasets/test/sample-dataset",
        status=AnalysisStatus.RUNNING,
        config={},
    )
    db_session.add(analysis)
    await db_session.commit()
    return analysis


async def _create_stage(db_session, analysis_id, status, progress_percent, message):
    stage = AnalysisStage(
        analysis_id=analysis_id,
        stage=StageType.EDA,
        status=status,
        progress_percent=progress_percent,
        current_step=message,
    )
    db_session.add(stage)
    await db_session.commit()
    return stage


@pytest.mark.asyncio
async def test_stream_progress_redis_happy_path(monkeypatch):
    analysis_id = str(uuid.uuid4())
    running = progress_service.ProgressEvent(
        analysis_id=analysis_id,
        stage="eda",
        status="running",
        progress_percent=10,
        message="starting",
    )
    completed = progress_service.ProgressEvent(
        analysis_id=analysis_id,
        stage="eda",
        status="completed",
        progress_percent=100,
        message="done",
    )
    messages = [
        {"type": "message", "data": running.to_json()},
        {"type": "message", "data": completed.to_json()},
    ]
    fake_pubsub = FakePubSub(messages)
    fake_redis = FakeRedis(fake_pubsub)
    monkeypatch.setattr(progress_service, "get_redis", AsyncMock(return_value=fake_redis))

    events = [event async for event in progress_service.stream_progress(analysis_id)]

    assert [event.status for event in events] == ["running", "completed"]


@pytest.mark.asyncio
async def test_stream_progress_deduplicates_events(monkeypatch):
    analysis_id = str(uuid.uuid4())
    running = progress_service.ProgressEvent(
        analysis_id=analysis_id,
        stage="eda",
        status="running",
        progress_percent=10,
        message="starting",
    )
    completed = progress_service.ProgressEvent(
        analysis_id=analysis_id,
        stage="eda",
        status="completed",
        progress_percent=100,
        message="done",
    )
    messages = [
        {"type": "message", "data": running.to_json()},
        {"type": "message", "data": running.to_json()},
        {"type": "message", "data": completed.to_json()},
    ]
    fake_pubsub = FakePubSub(messages)
    fake_redis = FakeRedis(fake_pubsub)
    monkeypatch.setattr(progress_service, "get_redis", AsyncMock(return_value=fake_redis))

    events = [event async for event in progress_service.stream_progress(analysis_id)]

    assert len(events) == 2
    assert events[0].status == "running"
    assert events[1].status == "completed"


@pytest.mark.asyncio
async def test_stream_progress_fallback_to_database(db_session, monkeypatch):
    analysis = await _create_analysis(db_session)
    await _create_stage(
        db_session,
        analysis.id,
        StageStatus.COMPLETED,
        100,
        "done",
    )

    @asynccontextmanager
    async def session_context():
        yield db_session

    monkeypatch.setattr(
        progress_service,
        "get_redis",
        AsyncMock(side_effect=RedisConnectionError("down")),
    )
    monkeypatch.setattr(progress_service, "get_session_context", session_context)

    events = [event async for event in progress_service.stream_progress(str(analysis.id))]

    assert len(events) == 1
    assert events[0].status == "completed"
    assert events[0].progress_percent == 100


@pytest.mark.asyncio
async def test_stream_from_database_exponential_backoff(db_session, monkeypatch):
    sleep_calls = []
    current_time = {"value": 0.0}

    def fake_monotonic():
        return current_time["value"]

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        current_time["value"] += duration

    @asynccontextmanager
    async def session_context():
        yield db_session

    monkeypatch.setattr(progress_service.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(progress_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(progress_service, "get_session_context", session_context)

    events = [
        event
        async for event in progress_service._stream_from_database(
            analysis_id=str(uuid.uuid4()),
            poll_interval=1.0,
            max_poll_interval=3.0,
            timeout_seconds=4.0,
        )
    ]

    assert events == []
    assert sleep_calls == [1.0, 1.5, 2.25]


@pytest.mark.asyncio
async def test_stream_from_database_timeout(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr(progress_service.asyncio, "sleep", sleep_mock)
    monkeypatch.setattr(progress_service.time, "monotonic", lambda: 0.0)

    events = [
        event
        async for event in progress_service._stream_from_database(
            analysis_id=str(uuid.uuid4()),
            timeout_seconds=0,
        )
    ]

    assert events == []
    assert sleep_mock.await_count == 0


@pytest.mark.asyncio
async def test_analysis_event_generator_timeout_yields_disconnected(monkeypatch):
    async def stalled_stream_progress(analysis_id):  # noqa: ARG001 - test stub
        while True:
            await asyncio.sleep(3600)
            yield progress_service.ProgressEvent(
                analysis_id=str(uuid.uuid4()),
                stage="eda",
                status="running",
                progress_percent=10,
                message="waiting",
            )

    analysis = Analysis(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        kaggle_url="https://www.kaggle.com/datasets/test/sample-dataset",
        status=AnalysisStatus.RUNNING,
        config={},
    )

    monkeypatch.setattr(analyses_api, "stream_progress", stalled_stream_progress)
    monkeypatch.setattr(analyses_api, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(analyses_api.settings, "PROGRESS_STREAM_TIMEOUT", 0.05)

    events = []

    async def collect_events():
        async for event in analyses_api._analysis_event_generator(analysis):
            events.append(event)
            if event.get("event") == "disconnected":
                break

    await asyncio.wait_for(collect_events(), timeout=0.5)

    assert events[0]["event"] == "progress"
    assert any(event.get("comment") == "heartbeat" for event in events)
    disconnected = next(event for event in events if event.get("event") == "disconnected")
    payload = json.loads(disconnected["data"])
    assert payload["reason"] == "timeout"


@pytest.mark.asyncio
async def test_analysis_event_generator_handles_cancelled(monkeypatch):
    async def stalled_stream_progress(analysis_id):  # noqa: ARG001 - test stub
        while True:
            await asyncio.sleep(3600)
            yield progress_service.ProgressEvent(
                analysis_id=str(uuid.uuid4()),
                stage="eda",
                status="running",
                progress_percent=10,
                message="waiting",
            )

    analysis = Analysis(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        kaggle_url="https://www.kaggle.com/datasets/test/sample-dataset",
        status=AnalysisStatus.RUNNING,
        config={},
    )

    monkeypatch.setattr(analyses_api, "stream_progress", stalled_stream_progress)
    monkeypatch.setattr(analyses_api.settings, "PROGRESS_STREAM_TIMEOUT", 60)

    async def consume():
        async for _ in analyses_api._analysis_event_generator(analysis):
            await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
