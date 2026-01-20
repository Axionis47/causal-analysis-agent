"""Unit tests for ValidationAgent."""

from __future__ import annotations

import sys
import types
import uuid

import pandas as pd
import pytest
from sqlalchemy import select

from app.agents import validation
from app.agents.validation import ValidationAgent, _run_refutation
from app.models.validation_result import ValidationResult
from tests.fixtures.agent_fixtures import create_analysis, create_dataset, create_treatment_effect
from tests.fixtures.mock_helpers import make_session_context


def _install_module(path: str, module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = path.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, path, module)


@pytest.mark.asyncio
async def test_validation_cache_hit():
    state = {
        "analysis_id": str(uuid.uuid4()),
        "dataset_id": str(uuid.uuid4()),
        "validation_result_ids": [str(uuid.uuid4())],
    }

    result = await ValidationAgent()._run(state)

    assert result.outputs["cached"] is True


@pytest.mark.asyncio
async def test_validation_no_treatment_effects(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)

    monkeypatch.setattr(validation, "get_session_context", make_session_context(db_session))

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    result = await ValidationAgent()._run(state)

    assert result.message == "No treatment effects to validate"


def test_placebo_refuter_passed(monkeypatch):
    class DummyRefute:
        p_value = 0.8

        def __str__(self):
            return "refute"

    class DummyEstimate:
        value = 0.2

    class DummyModel:
        def __init__(self, data, treatment, outcome, common_causes=None):  # noqa: ARG002
            return None

        def identify_effect(self):
            return "estimand"

        def estimate_effect(self, estimand, method_name=None):  # noqa: ARG002
            return DummyEstimate()

        def refute_estimate(self, estimand, estimate, method_name=None):  # noqa: ARG002
            return DummyRefute()

    module = types.ModuleType("dowhy")
    module.CausalModel = DummyModel
    _install_module("dowhy", module, monkeypatch)

    df = pd.DataFrame({"treatment": [0, 1], "outcome": [1, 2], "conf": [0.1, 0.2]})
    result = _run_refutation(df, "treatment", "outcome", ["conf"], "placebo_treatment_refuter")

    assert result
    assert result["passed"] is True
    assert result["confidence_score"] == 0.7


def test_random_common_cause_failed(monkeypatch):
    class DummyRefute:
        p_value = 0.01

        def __str__(self):
            return "refute"

    class DummyEstimate:
        value = 0.2

    class DummyModel:
        def __init__(self, data, treatment, outcome, common_causes=None):  # noqa: ARG002
            return None

        def identify_effect(self):
            return "estimand"

        def estimate_effect(self, estimand, method_name=None):  # noqa: ARG002
            return DummyEstimate()

        def refute_estimate(self, estimand, estimate, method_name=None):  # noqa: ARG002
            return DummyRefute()

    module = types.ModuleType("dowhy")
    module.CausalModel = DummyModel
    _install_module("dowhy", module, monkeypatch)

    df = pd.DataFrame({"treatment": [0, 1], "outcome": [1, 2], "conf": [0.1, 0.2]})
    result = _run_refutation(df, "treatment", "outcome", ["conf"], "random_common_cause")

    assert result
    assert result["passed"] is False
    assert result["confidence_score"] == 0.3
    assert result["recommendations"]


@pytest.mark.asyncio
async def test_multiple_refutation_tests_persist_and_aggregate(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    await create_treatment_effect(db_session, analysis_id=analysis.id)

    def fake_run_refutation(*_args, **_kwargs):
        method = _kwargs.get("method") or _args[-1]
        if method == "placebo_treatment_refuter":
            return {
                "method": method,
                "passed": True,
                "confidence_score": 0.7,
                "details": {"p_value": 0.8},
                "recommendations": [],
            }
        return {
            "method": method,
            "passed": False,
            "confidence_score": 0.3,
            "details": {"p_value": 0.01},
            "recommendations": ["Review"],
        }

    monkeypatch.setattr(validation, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(validation, "_load_dataframe", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(validation, "_run_refutation", fake_run_refutation)

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    result = await ValidationAgent()._run(state)

    assert result.outputs["confidence_score"] == 0.5
    stored = await db_session.execute(
        select(ValidationResult).where(ValidationResult.analysis_id == analysis.id)
    )
    assert len(stored.scalars().all()) == 2


def test_refutation_error_handling(monkeypatch):
    class DummyModel:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("boom")

    module = types.ModuleType("dowhy")
    module.CausalModel = DummyModel
    _install_module("dowhy", module, monkeypatch)

    df = pd.DataFrame({"treatment": [0, 1], "outcome": [1, 2]})
    result = _run_refutation(df, "treatment", "outcome", [], "placebo_treatment_refuter")

    assert result is None
