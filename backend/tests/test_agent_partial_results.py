"""Tests for partial result preservation in agents."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from app.agents.base import (
    AgentFailure,
    AgentResult,
    BaseAgent,
    PARTIAL_OUTPUTS_KEY,
    _extract_partial_outputs,
)
from app.models.analysis_stage import StageType


class PartialResultAgent(BaseAgent):
    name = "PartialResultAgent"
    stage = StageType.EDA

    async def _run(self, state):
        partial_state = {**state, "extra": "value"}
        partial = AgentResult(state=partial_state, outputs={"partial": True}, message="partial")
        raise AgentFailure("boom", partial_result=partial)


class PartialOutputsAgent(BaseAgent):
    name = "PartialOutputsAgent"
    stage = StageType.EDA

    async def _run(self, state):
        raise AgentFailure("boom", partial_outputs={"checkpoint": 1})


@pytest.mark.asyncio
async def test_agentfailure_with_partial_result_updates_state(monkeypatch):
    agent = PartialResultAgent()
    state = {"analysis_id": str(uuid.uuid4()), "kaggle_url": "url"}

    @asynccontextmanager
    async def session_context():
        yield SimpleNamespace()

    monkeypatch.setattr("app.agents.base.get_session_context", session_context)
    monkeypatch.setattr(
        "app.agents.base.start_stage",
        AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
    )
    monkeypatch.setattr("app.agents.base.fail_stage", AsyncMock())

    updated = await agent.run(state)

    assert updated.get("extra") == "value"
    partials = updated.get(PARTIAL_OUTPUTS_KEY)
    assert partials[agent.name] == {"partial": True}


@pytest.mark.asyncio
async def test_agentfailure_with_partial_outputs_dict(monkeypatch):
    agent = PartialOutputsAgent()
    state = {"analysis_id": str(uuid.uuid4()), "kaggle_url": "url"}

    @asynccontextmanager
    async def session_context():
        yield SimpleNamespace()

    monkeypatch.setattr("app.agents.base.get_session_context", session_context)
    monkeypatch.setattr(
        "app.agents.base.start_stage",
        AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
    )
    monkeypatch.setattr("app.agents.base.fail_stage", AsyncMock())

    updated = await agent.run(state)

    partials = updated.get(PARTIAL_OUTPUTS_KEY)
    assert partials[agent.name] == {"checkpoint": 1}


def test_state_diff_extraction():
    snapshot = {"analysis_id": "id", "kaggle_url": "url", "value": 1}
    state = {"analysis_id": "id", "kaggle_url": "url", "value": 2, "new": "x"}

    partial = _extract_partial_outputs(state, RuntimeError("boom"), snapshot, "Agent")

    assert partial == {"value": 2, "new": "x"}


@pytest.mark.asyncio
async def test_partial_outputs_persisted_in_fail_stage(monkeypatch):
    agent = PartialOutputsAgent()
    state = {"analysis_id": str(uuid.uuid4()), "kaggle_url": "url"}
    fail_mock = AsyncMock()

    @asynccontextmanager
    async def session_context():
        yield SimpleNamespace()

    monkeypatch.setattr("app.agents.base.get_session_context", session_context)
    monkeypatch.setattr(
        "app.agents.base.start_stage",
        AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
    )
    monkeypatch.setattr("app.agents.base.fail_stage", fail_mock)

    await agent.run(state)

    assert fail_mock.await_count == 1
    assert fail_mock.call_args.kwargs["outputs"] == {"checkpoint": 1}


@pytest.mark.asyncio
async def test_multiple_agent_failures_accumulate(monkeypatch):
    agent_a = PartialOutputsAgent()
    agent_b = PartialOutputsAgent()
    agent_b.name = "SecondAgent"
    state = {"analysis_id": str(uuid.uuid4()), "kaggle_url": "url"}

    @asynccontextmanager
    async def session_context():
        yield SimpleNamespace()

    monkeypatch.setattr("app.agents.base.get_session_context", session_context)
    monkeypatch.setattr(
        "app.agents.base.start_stage",
        AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())),
    )
    monkeypatch.setattr("app.agents.base.fail_stage", AsyncMock())

    state = await agent_a.run(state)
    state = await agent_b.run(state)

    partials = state.get(PARTIAL_OUTPUTS_KEY)
    assert partials[agent_a.name]
    assert partials[agent_b.name]


def test_excluded_keys_not_in_partial_outputs():
    snapshot = {
        "analysis_id": "id",
        "kaggle_url": "url",
        "errors": [],
        "progress_percent": 10,
    }
    state = {
        "analysis_id": "id",
        "kaggle_url": "url",
        "errors": ["boom"],
        "progress_percent": 20,
    }

    partial = _extract_partial_outputs(state, RuntimeError("boom"), snapshot, "Agent")

    assert partial is None
