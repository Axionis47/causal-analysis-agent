"""Unit tests for TreatmentEffectsAgent."""

from __future__ import annotations

import sys
import types
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.agents import treatment
from app.agents.treatment import (
    TreatmentEffectsAgent,
    _confounders_from_state,
    _consensus_summary,
    _estimate_doubly_robust,
    _estimate_heterogeneous,
    _estimate_iv,
    _estimate_mediation,
    _estimate_psm,
    _estimate_time_varying,
    _select_pairs,
)
from app.models.treatment_effect import TreatmentEffect
from tests.fixtures.agent_fixtures import create_analysis, create_dataset
from tests.fixtures.mock_helpers import create_mock_dataframe, make_session_context


def _install_module(path: str, module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = path.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, path, module)


@pytest.mark.asyncio
async def test_treatment_cache_hit():
    state = {
        "analysis_id": str(uuid.uuid4()),
        "dataset_id": str(uuid.uuid4()),
        "treatment_effect_ids": [str(uuid.uuid4())],
    }

    result = await TreatmentEffectsAgent()._run(state)

    assert result.outputs["cached"] is True


def test_psm_estimation(monkeypatch):
    class DummyEstimate:
        value = 0.25

        def get_confidence_intervals(self):
            return [(0.1, 0.4)]

    class DummyModel:
        def __init__(self, data, treatment, outcome, common_causes=None):  # noqa: ARG002
            return None

        def identify_effect(self):
            return "estimand"

        def estimate_effect(self, estimand, method_name=None):  # noqa: ARG002
            return DummyEstimate()

    module = types.ModuleType("dowhy")
    module.CausalModel = DummyModel
    _install_module("dowhy", module, monkeypatch)

    df = pd.DataFrame({"treatment": [0, 1], "outcome": [1, 2], "feature": [3, 4]})
    result = _estimate_psm(df, "treatment", "outcome", ["feature"])

    assert result
    assert result["ate"] == 0.25
    assert result["confidence_interval"]["lower"] == 0.1


def test_doubly_robust_estimation(monkeypatch):
    class DummyLinearDML:
        def __init__(self, model_y=None, model_t=None):  # noqa: ARG002
            return None

        def fit(self, y, t, X=None):  # noqa: ARG002,N802
            return None

        def ate(self, X=None):  # noqa: ARG002,N802
            return 0.15

    module = types.ModuleType("econml.dml")
    module.LinearDML = DummyLinearDML
    _install_module("econml.dml", module, monkeypatch)

    class DummyLinearRegression:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            return None

    class DummyLogisticRegression:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            return None

    sklearn_module = types.ModuleType("sklearn.linear_model")
    sklearn_module.LinearRegression = DummyLinearRegression
    sklearn_module.LogisticRegression = DummyLogisticRegression
    _install_module("sklearn.linear_model", sklearn_module, monkeypatch)

    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0],
            "outcome": [1.0, 2.0, 1.5],
            "conf": [0.2, 0.1, 0.3],
        }
    )
    result = _estimate_doubly_robust(df, "treatment", "outcome", ["conf"])

    assert result
    assert result["ate"] == 0.15


def test_iv_estimation(monkeypatch):
    class DummyEstimate:
        value = 0.4

    class DummyModel:
        def __init__(  # noqa: ARG002
            self,
            data,
            treatment,
            outcome,
            common_causes=None,
            instruments=None,
        ):
            self.instruments = instruments

        def identify_effect(self):
            return "estimand"

        def estimate_effect(self, estimand, method_name=None):  # noqa: ARG002
            return DummyEstimate()

    module = types.ModuleType("dowhy")
    module.CausalModel = DummyModel
    _install_module("dowhy", module, monkeypatch)

    monkeypatch.setattr(treatment, "_select_instrument", lambda *_args, **_kwargs: "instrument")

    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1],
            "outcome": [1.0, 1.1, 1.0, 1.2],
            "instrument": [0.2, 0.3, 0.21, 0.31],
        }
    )
    result = _estimate_iv(df, "treatment", "outcome", [])

    assert result
    assert result["ate"] == 0.4


def test_mediation_analysis(monkeypatch):
    class DummyEstimate:
        value = 0.33

    class DummyModel:
        def __init__(  # noqa: ARG002
            self,
            data,
            treatment,
            outcome,
            common_causes=None,
            mediators=None,
        ):
            self.mediators = mediators

        def identify_effect(self):
            return "estimand"

        def estimate_effect(self, estimand, method_name=None):  # noqa: ARG002
            return DummyEstimate()

    module = types.ModuleType("dowhy")
    module.CausalModel = DummyModel
    _install_module("dowhy", module, monkeypatch)

    df = pd.DataFrame(
        {
            "treatment": [0, 1],
            "outcome": [1.0, 2.0],
            "mediator": [0.5, 0.7],
        }
    )
    result = _estimate_mediation(df, "treatment", "outcome", [])

    assert result
    assert result["mediator"] == "mediator"


def test_heterogeneous_effects(monkeypatch):
    class DummyForest:
        def __init__(self, model_y=None, model_t=None):  # noqa: ARG002
            return None

        def fit(self, y, t, X=None):  # noqa: ARG002,N802
            return None

        def effect(self, X):  # noqa: N802
            return np.array([0.1] * len(X))

    module = types.ModuleType("econml.dml")
    module.CausalForestDML = DummyForest
    _install_module("econml.dml", module, monkeypatch)

    class DummyLinearRegression:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            return None

    class DummyLogisticRegression:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            return None

    sklearn_module = types.ModuleType("sklearn.linear_model")
    sklearn_module.LinearRegression = DummyLinearRegression
    sklearn_module.LogisticRegression = DummyLogisticRegression
    _install_module("sklearn.linear_model", sklearn_module, monkeypatch)

    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1],
            "outcome": [1.0, 1.2, 1.1, 1.3],
            "conf": [1, 2, 3, 4],
        }
    )
    result = _estimate_heterogeneous(df, "treatment", "outcome", ["conf"])

    assert result
    assert result["cate_mean"] == pytest.approx(0.1)


def test_time_varying_effects():
    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1] * 5,
            "outcome": np.arange(20, dtype=float),
            "event_time": np.arange(20),
        }
    )

    result = _estimate_time_varying(df, "treatment", "outcome")

    assert result
    assert result["time_bins"]


def test_select_pairs_from_state():
    df = pd.DataFrame({"treatment": [0, 1], "outcome": [1, 2], "other": [3, 4]})
    state = {
        "data_characteristics": {
            "treatment_candidates": ["treatment"],
            "outcome_candidates": ["outcome"],
        }
    }

    pairs = _select_pairs(df, state)

    assert pairs == [("treatment", "outcome")]


def test_select_pairs_fallback_numeric():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    pairs = _select_pairs(df, {})

    assert pairs == [("a", "b")]


def test_confounder_filtering():
    state = {"data_characteristics": {"confounder_candidates": ["treatment", "conf", "outcome"]}}

    confounders = _confounders_from_state(state, "treatment", "outcome")

    assert confounders == ["conf"]


def test_consensus_summary():
    results = [
        {"method": "psm", "ate": 0.2},
        {"method": "dr", "ate": 0.3},
        {"method": "iv", "ate": 0.1},
    ]

    summary = _consensus_summary(results)

    assert summary
    assert summary["mean_ate"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_tool_registry_integration_and_persistence(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=80, include_categorical=False)
    track_calls = []

    class DummyRegistry:
        def available_tools(self, *_args, **_kwargs):
            return [
                SimpleNamespace(name="doubly_robust"),
                SimpleNamespace(name="instrumental_variable"),
            ]

        def track_performance(self, tool_name, success, elapsed):
            track_calls.append((tool_name, success, elapsed))

    def fake_retry_sync(func, *args, **kwargs):
        kwargs.pop("retry_on", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(treatment, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(treatment, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(treatment, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(
        treatment,
        "_estimate_psm",
        lambda *_args, **_kwargs: {"ate": 0.2},
    )
    monkeypatch.setattr(
        treatment,
        "_estimate_doubly_robust",
        lambda *_args, **_kwargs: {"ate": 0.1},
    )
    monkeypatch.setattr(
        treatment,
        "_estimate_iv",
        lambda *_args, **_kwargs: {"ate": 0.3},
    )
    monkeypatch.setattr(treatment, "ToolRegistry", lambda: DummyRegistry())

    state = {
        "analysis_id": str(analysis.id),
        "dataset_id": str(dataset.id),
        "analysis_types": ["iv"],
        "data_characteristics": {
            "treatment_candidates": ["treatment"],
            "outcome_candidates": ["outcome"],
            "confounder_candidates": ["feature"],
        },
    }

    result = await TreatmentEffectsAgent()._run(state)

    assert result.outputs["treatment_effects"] == 3
    stored = await db_session.execute(
        TreatmentEffect.__table__.select().where(TreatmentEffect.analysis_id == analysis.id)
    )
    assert len(stored.fetchall()) == 3
    assert track_calls


@pytest.mark.asyncio
async def test_retry_on_estimation_failures(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=40, include_categorical=False)
    attempts = {"count": 0}

    def flaky_psm(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return {"ate": 0.2}

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
            return []

        def track_performance(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(treatment, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(treatment, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(treatment, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(treatment, "_estimate_psm", flaky_psm)
    monkeypatch.setattr(treatment, "ToolRegistry", lambda: DummyRegistry())

    state = {
        "analysis_id": str(analysis.id),
        "dataset_id": str(dataset.id),
        "data_characteristics": {
            "treatment_candidates": ["treatment"],
            "outcome_candidates": ["outcome"],
        },
    }
    await TreatmentEffectsAgent()._run(state)

    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_no_pairs_found(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = pd.DataFrame({"only": [1, 2, 3], "label": ["a", "b", "c"]})

    def fake_retry_sync(func, *args, **kwargs):
        kwargs.pop("retry_on", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(treatment, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(treatment, "retry_sync", fake_retry_sync)
    monkeypatch.setattr(treatment, "_load_dataframe", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(
        treatment,
        "ToolRegistry",
        lambda: SimpleNamespace(available_tools=lambda *_a, **_k: []),
    )

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    result = await TreatmentEffectsAgent()._run(state)

    assert result.message == "No treatment/outcome pairs found"
