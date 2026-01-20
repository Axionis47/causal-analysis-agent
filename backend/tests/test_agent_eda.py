"""Unit tests for EDAAgent."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from app.agents import eda
from app.agents.base import AgentFailure
from app.agents.eda import (
    EDAAgent,
    _augment_analysis_types,
    _columns_info,
    _heuristic_candidates,
    _preprocessing_recommendations,
    _quality_issues,
    _top_correlations,
)
from app.crud.dataset import dataset_crud
from app.services.data_quality import ValidationResult
from tests.fixtures.agent_fixtures import (
    create_analysis,
    create_data_understanding,
    create_dataset,
)
from tests.fixtures.mock_helpers import create_mock_dataframe, make_session_context


@pytest.mark.asyncio
async def test_eda_cache_from_state():
    state = {
        "analysis_id": str(uuid.uuid4()),
        "dataset_id": str(uuid.uuid4()),
        "data_characteristics": {"summary_stats": {"row_count": 10}},
    }

    result = await EDAAgent()._run(state)

    assert result.outputs["cached"] is True


@pytest.mark.asyncio
async def test_eda_cache_from_database(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    await create_data_understanding(
        db_session,
        dataset_id=dataset.id,
        columns=[{"name": "event_time", "dtype": "datetime"}],
    )

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))

    result = await EDAAgent()._run(state)

    assert result.outputs["cached"] is True
    assert "data_characteristics" in state
    assert "time_varying" in state.get("analysis_types", [])


@pytest.mark.asyncio
async def test_eda_fresh_execution_persists(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(
        rows=800,
        include_categorical=True,
        include_missing=True,
        include_outliers=True,
    )

    async def fake_retry_async(_func, *_args, **_kwargs):
        return df

    async def fake_llm(*_args, **_kwargs):
        return None

    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(eda, "retry_async", fake_retry_async)
    monkeypatch.setattr(eda, "call_llm_with_retry", fake_llm)

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    result = await EDAAgent()._run(state)

    assert result.outputs["row_count"] == len(df)
    assert state.get("data_characteristics")

    refreshed = await dataset_crud.get(db_session, dataset.id)
    assert refreshed is not None
    assert refreshed.characteristics


def test_columns_info_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4],
            "category": ["a", "b", "a", "b"],
            "flag": [True, False, True, False],
        }
    )

    info = _columns_info(df)
    numeric_entry = next(item for item in info if item["name"] == "numeric")
    categorical_entry = next(item for item in info if item["name"] == "category")
    binary_entry = next(item for item in info if item["name"] == "flag")

    assert numeric_entry["dtype"] == "numerical"
    assert "mean" in numeric_entry["distribution_summary"]
    assert categorical_entry["dtype"] == "categorical"
    assert "top_values" in categorical_entry["distribution_summary"]
    assert binary_entry["dtype"] == "binary"


def test_quality_issue_detection_missing_and_outliers():
    df = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 1000, 1001] * 10,
            "missing": [1, None, None, None, None] * 10,
        }
    )

    issues = _quality_issues(df)

    issue_types = {issue["type"] for issue in issues}
    assert "high_missing" in issue_types
    assert "outliers" in issue_types


def test_top_correlations_threshold():
    x = np.arange(100)
    df = pd.DataFrame({"a": x, "b": x * 2, "c": np.random.default_rng(1).normal(size=100)})

    correlations = _top_correlations(df)

    assert correlations
    top = correlations[0]
    assert top["value"] >= 0.6


def test_heuristic_candidates():
    df = pd.DataFrame(
        {
            "treat_flag": [0, 1, 0, 1],
            "target_value": [1, 2, 3, 4],
            "feature": [5, 6, 7, 8],
        }
    )

    treatments, outcomes, confounders = _heuristic_candidates(df)

    assert "treat_flag" in treatments
    assert "target_value" in outcomes
    assert "feature" in confounders


def test_preprocessing_recommendations():
    columns = [
        {"name": "category", "dtype": "categorical"},
        {"name": "numeric", "dtype": "numerical"},
    ]
    issues = [{"type": "high_missing"}]

    steps = _preprocessing_recommendations(columns, issues)

    assert "impute_missing" in steps
    assert "encode_categorical" in steps
    assert "scale_numeric" in steps


def test_analysis_type_augmentation():
    existing = ["time_varying"]
    columns = [
        {"name": "event_time", "dtype": "datetime"},
        {"name": "mediator_var", "dtype": "numerical"},
    ]

    updated = _augment_analysis_types(existing, columns)

    assert "time_varying" in updated
    assert "mediation" in updated


@pytest.mark.asyncio
async def test_llm_suggestions_override_heuristics(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=800)

    async def fake_retry_async(_func, *_args, **_kwargs):
        return df

    async def fake_llm(*_args, **_kwargs):
        return {
            "treatment_candidates": ["custom_treatment"],
            "outcome_candidates": ["custom_outcome"],
            "confounder_candidates": ["custom_confounder"],
            "data_quality_issues": [],
        }

    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(eda, "retry_async", fake_retry_async)
    monkeypatch.setattr(eda, "call_llm_with_retry", fake_llm)

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    await EDAAgent()._run(state)

    characteristics = state["data_characteristics"]
    assert characteristics["treatment_candidates"] == ["custom_treatment"]
    assert characteristics["outcome_candidates"] == ["custom_outcome"]
    assert characteristics["confounder_candidates"] == ["custom_confounder"]


@pytest.mark.asyncio
async def test_llm_retry_failure_falls_back_to_heuristics(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=800)

    async def fake_retry_async(_func, *_args, **_kwargs):
        return df

    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(eda, "retry_async", fake_retry_async)
    async def fake_llm(*_args, **_kwargs):
        return None

    monkeypatch.setattr(eda, "call_llm_with_retry", fake_llm)

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    await EDAAgent()._run(state)

    characteristics = state["data_characteristics"]
    assert characteristics["treatment_candidates"]
    assert characteristics["outcome_candidates"]
    assert characteristics["confounder_candidates"]


@pytest.mark.asyncio
async def test_eda_validation_blocks_bad_data(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=200)

    async def fake_retry_async(_func, *_args, **_kwargs):
        return df

    async def fake_llm(*_args, **_kwargs):
        return None

    class FakeValidator:
        async def validate_dataframe(self, _df):
            return ValidationResult(
                passed=False,
                warnings=[
                    {
                        "level": "error",
                        "message": "Too few rows",
                        "metric": "row_count",
                        "value": 50,
                        "threshold": 100,
                        "affected_columns": [],
                    }
                ],
                can_proceed_with_override=False,
            )

    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(eda, "retry_async", fake_retry_async)
    monkeypatch.setattr(eda, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(eda, "DataQualityValidator", FakeValidator)

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    with pytest.raises(AgentFailure):
        await EDAAgent()._run(state)


@pytest.mark.asyncio
async def test_eda_validation_warnings_require_override(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=800, include_missing=False)
    warnings = [
        {
            "level": "warning",
            "message": "Duplicate rows are high",
            "metric": "duplicates",
            "value": 0.4,
            "threshold": 0.3,
            "affected_columns": [],
        }
    ]

    async def fake_retry_async(_func, *_args, **_kwargs):
        return df

    async def fake_llm(*_args, **_kwargs):
        return None

    class FakeValidator:
        async def validate_dataframe(self, _df):
            return ValidationResult(
                passed=True,
                warnings=warnings,
                can_proceed_with_override=True,
            )

    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(eda, "retry_async", fake_retry_async)
    monkeypatch.setattr(eda, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(eda, "DataQualityValidator", FakeValidator)

    state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}
    with pytest.raises(AgentFailure) as exc_info:
        await EDAAgent()._run(state)

    assert exc_info.value.partial_outputs["quality_warnings"] == warnings


@pytest.mark.asyncio
async def test_eda_validation_override_allows_proceed(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    dataset = await create_dataset(db_session, analysis_id=analysis.id)
    df = create_mock_dataframe(rows=800, include_missing=False)
    warnings = [
        {
            "level": "warning",
            "message": "Duplicate rows are high",
            "metric": "duplicates",
            "value": 0.4,
            "threshold": 0.3,
            "affected_columns": [],
        }
    ]

    async def fake_retry_async(_func, *_args, **_kwargs):
        return df

    async def fake_llm(*_args, **_kwargs):
        return None

    class FakeValidator:
        async def validate_dataframe(self, _df):
            return ValidationResult(
                passed=True,
                warnings=warnings,
                can_proceed_with_override=True,
            )

    monkeypatch.setattr(eda, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(eda, "retry_async", fake_retry_async)
    monkeypatch.setattr(eda, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(eda, "DataQualityValidator", FakeValidator)

    state = {
        "analysis_id": str(analysis.id),
        "dataset_id": str(dataset.id),
        "config": {"override_quality_warnings": True},
    }
    await EDAAgent()._run(state)

    refreshed = await dataset_crud.get_with_understanding(db_session, dataset.id)
    assert refreshed is not None
    assert refreshed.data_understanding is not None
    assert refreshed.data_understanding.quality_warnings == warnings
