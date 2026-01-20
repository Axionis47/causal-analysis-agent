"""Unit tests for CausalDiscoveryAgent."""

from __future__ import annotations

import builtins
import sys
import types
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.agents import discovery
from app.agents.discovery import (
    CausalDiscoveryAgent,
    _prepare_matrix,
    _run_fci,
    _run_ges,
    _run_pc,
)
from app.models.agent_interaction import AgentInteraction
from app.models.causal_graph import CausalGraph
from sqlalchemy import select

from tests.fixtures.agent_fixtures import (
    MOCK_GES_GRAPH,
    MOCK_PC_GRAPH,
    create_analysis,
    create_dataset,
)
from tests.fixtures.mock_helpers import create_mock_dataframe, make_session_context


def _install_module(path: str, module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = path.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, path, module)


@pytest.mark.asyncio
async def test_discovery_cache_hit():
    state = {
        "analysis_id": str(uuid.uuid4()),
        "dataset_id": str(uuid.uuid4()),
        "causal_graph_ids": [str(uuid.uuid4())],
    }

    result = await CausalDiscoveryAgent()._run(state)

    assert result.outputs["cached"] is True


def test_pc_algorithm_success(monkeypatch):
    class DummyGraph:
        graph = np.array([[0, 1], [0, 0]])

    class DummyCG:
        G = DummyGraph()

    def fake_pc(_data, alpha=0.05):  # noqa: ARG001
        return DummyCG()

    module = types.ModuleType("causallearn.search.ConstraintBased.PC")
    module.pc = fake_pc
    _install_module("causallearn.search.ConstraintBased.PC", module, monkeypatch)

    payload = _run_pc(np.ones((2, 2)), ["A", "B"])

    assert payload["edges"]
    assert payload["nodes"]


def test_ges_algorithm_success(monkeypatch):
    class DummyGraph:
        graph = np.array([[0, 1], [0, 0]])

    def fake_ges(_data):  # noqa: ARG001
        return {"G": DummyGraph()}

    module = types.ModuleType("causallearn.search.ScoreBased.GES")
    module.ges = fake_ges
    _install_module("causallearn.search.ScoreBased.GES", module, monkeypatch)

    payload = _run_ges(np.ones((2, 2)), ["X", "Y"])

    assert payload["edges"]
    assert payload["nodes"]


def test_fci_algorithm_success(monkeypatch):
    class DummyGraph:
        graph = np.array([[0, 1], [0, 0]])

    def fake_fci(_data, alpha=0.05):  # noqa: ARG001
        return DummyGraph(), None

    module = types.ModuleType("causallearn.search.ConstraintBased.FCI")
    module.fci = fake_fci
    _install_module("causallearn.search.ConstraintBased.FCI", module, monkeypatch)

    payload = _run_fci(np.ones((2, 2)), ["M", "N"])

    assert payload["edges"]
    assert payload["nodes"]


def test_prepare_matrix_encodes_categorical_and_imputes_missing():
    df = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0] * 8,
            "category": ["a", "b", None, "a"] * 8,
            "text": [f"text_{idx}" for idx in range(32)],
        }
    )

    matrix, names = _prepare_matrix(df)

    assert "numeric" in names
    assert "category" in names
    assert "text" not in names
    assert not np.isnan(matrix).any()


def test_fallback_graph_on_import_failure(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "causallearn.search.ConstraintBased.PC":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    payload = _run_pc(np.ones((2, 2)), ["A", "B", "C"])

    assert payload["confidence"] == 0.4
    assert len(payload["edges"]) <= 3


@pytest.mark.asyncio
async def test_llm_interpretation_updates_confidence(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=50)

    async def fake_llm(*_args, **_kwargs):
        return {"confidence": 0.9}

    class DummyRegistry:
        def available_tools(self, *_args, **_kwargs):
            return [SimpleNamespace(name="pc")]

        def track_performance(self, *_args, **_kwargs):
            return None

    def fake_retry_sync(func, *args, **kwargs):
        kwargs.pop("retry_on", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(discovery, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(discovery, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(discovery, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(discovery, "_run_discovery", lambda *_args, **_kwargs: dict(MOCK_PC_GRAPH))
    monkeypatch.setattr(discovery, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(discovery, "ToolRegistry", lambda: DummyRegistry())

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    await CausalDiscoveryAgent()._run(state)

    graph = await db_session.get(CausalGraph, uuid.UUID(state["causal_graph_ids"][0]))
    assert graph is not None
    assert graph.confidence == 0.9


@pytest.mark.asyncio
async def test_multiple_tools_execution_and_tracking(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=30)
    tool_calls = []

    class DummyRegistry:
        def available_tools(self, *_args, **_kwargs):
            return [SimpleNamespace(name="pc"), SimpleNamespace(name="ges")]

        def track_performance(self, tool_name, success, elapsed):
            tool_calls.append((tool_name, success, elapsed))

    payloads = [dict(MOCK_PC_GRAPH), dict(MOCK_GES_GRAPH)]

    def fake_run_discovery(*_args, **_kwargs):
        return payloads.pop(0)

    def fake_retry_sync(func, *args, **kwargs):
        kwargs.pop("retry_on", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(discovery, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(discovery, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(discovery, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(discovery, "_run_discovery", fake_run_discovery)

    async def fake_llm(*_args, **_kwargs):
        return None

    monkeypatch.setattr(discovery, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(discovery, "ToolRegistry", lambda: DummyRegistry())

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    result = await CausalDiscoveryAgent()._run(state)

    assert len(state["causal_graph_ids"]) == 2
    assert len(result.outputs["tools"]) == 2
    assert len(tool_calls) == 2


@pytest.mark.asyncio
async def test_low_confidence_question_created(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=20)

    class DummyRegistry:
        def available_tools(self, *_args, **_kwargs):
            return [SimpleNamespace(name="pc")]

        def track_performance(self, *_args, **_kwargs):
            return None

    low_conf_graph = {
        "nodes": [{"name": "A", "node_type": "variable"}, {"name": "B", "node_type": "variable"}],
        "edges": [{"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.4}],
        "graph_data": {},
        "confidence": 0.4,
    }

    def fake_retry_sync(func, *args, **kwargs):
        kwargs.pop("retry_on", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(discovery, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(discovery, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(discovery, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(discovery, "_run_discovery", lambda *_args, **_kwargs: dict(low_conf_graph))

    async def fake_llm(*_args, **_kwargs):
        return None

    monkeypatch.setattr(discovery, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(discovery, "ToolRegistry", lambda: DummyRegistry())

    state = {
        "analysis_id": str(analysis.id),
        "dataset_id": str(dataset.id),
        "data_characteristics": {"treatment_candidates": ["treatment"]},
    }
    await CausalDiscoveryAgent()._run(state)

    assert state.get("questions")
    assert state["questions"][0]["confidence"] < 0.5
    result = await db_session.execute(
        select(AgentInteraction).where(AgentInteraction.analysis_id == analysis.id)
    )
    assert result.scalars().all()


@pytest.mark.asyncio
async def test_retry_on_tool_errors(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=20)
    attempts = {"count": 0}

    def flaky_run(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return dict(MOCK_PC_GRAPH)

    def fake_retry_sync(func, *args, **kwargs):
        last_exc = None
        for _ in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - retry loop
                last_exc = exc
                if kwargs.get("retry_on") and not kwargs["retry_on"](exc):
                    raise
        raise last_exc

    class DummyRegistry:
        def available_tools(self, *_args, **_kwargs):
            return [SimpleNamespace(name="pc")]

        def track_performance(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(discovery, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(discovery, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(discovery, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(discovery, "_run_discovery", flaky_run)

    async def fake_llm(*_args, **_kwargs):
        return None

    monkeypatch.setattr(discovery, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(discovery, "ToolRegistry", lambda: DummyRegistry())

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    await CausalDiscoveryAgent()._run(state)

    assert attempts["count"] == 2


def test_empty_data_matrix_raises():
    df = pd.DataFrame({"text": [f"val_{i}" for i in range(25)]})

    with pytest.raises(ValueError, match="No usable columns"):
        _prepare_matrix(df)
