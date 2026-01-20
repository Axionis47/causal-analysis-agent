"""Unit tests for BaseAgent behavior."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from app.agents import base
from app.agents.base import AgentFailure, AgentResult, BaseAgent, PARTIAL_OUTPUTS_KEY
from app.models.analysis_stage import AnalysisStage, StageType
from app.services import stages
from sqlalchemy import select
from tests.fixtures.agent_fixtures import create_analysis
from tests.fixtures.mock_helpers import make_session_context


class SuccessAgent(BaseAgent):
    name = "SuccessAgent"
    stage = StageType.EDA

    async def _run(self, state):
        state["result"] = "ok"
        return AgentResult(state=state, outputs={"status": "ok"}, message="done")


class FailingAgent(BaseAgent):
    name = "FailingAgent"
    stage = StageType.EDA

    async def _run(self, state):
        raise RuntimeError("boom")


class PartialAgent(BaseAgent):
    name = "PartialAgent"
    stage = StageType.EDA

    async def _run(self, state):
        raise AgentFailure("boom", partial_outputs={"partial": True})


@pytest.mark.asyncio
async def test_stage_lifecycle_success(monkeypatch):
    events = []

    class RecordingAgent(BaseAgent):
        name = "RecordingAgent"
        stage = StageType.EDA

        async def _run(self, state):
            events.append("run")
            return AgentResult(state=state, outputs={"status": "ok"}, message="done")

    async def fake_start(*_args, **_kwargs):
        events.append("start")
        return SimpleNamespace(id=uuid.uuid4())

    async def fake_complete(*_args, **_kwargs):
        events.append("complete")

    monkeypatch.setattr(base, "get_session_context", make_session_context(SimpleNamespace()))
    monkeypatch.setattr(base, "start_stage", fake_start)
    monkeypatch.setattr(base, "complete_stage", fake_complete)

    state = {"analysis_id": str(uuid.uuid4())}
    result = await RecordingAgent().run(state)

    assert result["analysis_id"]
    assert events == ["start", "run", "complete"]


@pytest.mark.asyncio
async def test_success_path_updates_outputs(monkeypatch):
    complete_mock = AsyncMock()

    async def fake_start(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(base, "get_session_context", make_session_context(SimpleNamespace()))
    monkeypatch.setattr(base, "start_stage", fake_start)
    monkeypatch.setattr(base, "complete_stage", complete_mock)

    state = {"analysis_id": str(uuid.uuid4())}
    result = await SuccessAgent().run(state)

    assert result["result"] == "ok"
    assert complete_mock.await_count == 1
    assert complete_mock.call_args.kwargs["outputs"] == {"status": "ok"}


@pytest.mark.asyncio
async def test_failure_path_records_error(monkeypatch):
    fail_mock = AsyncMock()

    async def fake_start(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(base, "get_session_context", make_session_context(SimpleNamespace()))
    monkeypatch.setattr(base, "start_stage", fake_start)
    monkeypatch.setattr(base, "fail_stage", fail_mock)

    state = {"analysis_id": str(uuid.uuid4())}
    result = await FailingAgent().run(state)

    assert "errors" in result
    assert "boom" in result["errors"][0]
    assert fail_mock.await_count == 1


@pytest.mark.asyncio
async def test_partial_outputs_on_failure(monkeypatch):
    fail_mock = AsyncMock()

    async def fake_start(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(base, "get_session_context", make_session_context(SimpleNamespace()))
    monkeypatch.setattr(base, "start_stage", fake_start)
    monkeypatch.setattr(base, "fail_stage", fail_mock)

    state = {"analysis_id": str(uuid.uuid4())}
    result = await PartialAgent().run(state)

    assert result[PARTIAL_OUTPUTS_KEY]["PartialAgent"] == {"partial": True}
    assert fail_mock.call_args.kwargs["outputs"] == {"partial": True}


@pytest.mark.asyncio
async def test_stage_tracking_creates_analysis_stage(db_session, monkeypatch):
    analysis = await create_analysis(db_session)

    monkeypatch.setattr(base, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(stages, "publish_progress", AsyncMock())

    state = {"analysis_id": str(analysis.id)}
    await SuccessAgent().run(state)

    await db_session.commit()
    rows = await db_session.execute(
        select(AnalysisStage).where(AnalysisStage.analysis_id == analysis.id)
    )
    stages_rows = rows.scalars().all()
    assert stages_rows
    stage = stages_rows[0]
    assert stage.analysis_id == analysis.id
    assert stage.stage == StageType.EDA
    assert stage.agent_name == SuccessAgent.name


@pytest.mark.asyncio
async def test_tracing_integration(monkeypatch):
    traced_calls = []

    def fake_traced(name):
        traced_calls.append(name)

        def wrapper(func):
            return func

        return wrapper

    async def fake_start(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(base, "get_session_context", make_session_context(SimpleNamespace()))
    monkeypatch.setattr(base, "start_stage", fake_start)
    monkeypatch.setattr(base, "complete_stage", AsyncMock())
    monkeypatch.setattr(base, "traced", fake_traced)

    state = {"analysis_id": str(uuid.uuid4())}
    await SuccessAgent().run(state)

    assert traced_calls == ["SuccessAgent"]
