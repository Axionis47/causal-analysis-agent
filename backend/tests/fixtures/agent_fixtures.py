"""Shared fixtures and helpers for agent tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.crud.analysis import analysis_crud
from app.crud.dataset import dataset_crud
from app.models.analysis import AnalysisStatus
from app.models.causal_graph import CausalGraph, DiscoveryMethod
from app.models.data_understanding import DataUnderstanding
from app.models.generated_report import GeneratedReport, ReportFormat, ReportType
from app.models.treatment_effect import TreatmentEffect, TreatmentMethod
from app.models.validation_result import ValidationResult, ValidationType

from tests.fixtures.mock_helpers import (
    MockGCSClient,
    MockKaggleApi,
    MockLLMRouter,
    create_mock_dataframe,
    create_mock_state,
)

MOCK_PC_GRAPH = {
    "nodes": [{"name": "A", "node_type": "variable"}, {"name": "B", "node_type": "variable"}],
    "edges": [{"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.6}],
    "graph_data": {"nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]},
    "confidence": 0.6,
}
MOCK_GES_GRAPH = {
    "nodes": [{"name": "X", "node_type": "variable"}, {"name": "Y", "node_type": "variable"}],
    "edges": [{"source": "X", "target": "Y", "edge_type": "directed", "confidence": 0.7}],
    "graph_data": {"nodes": [{"id": "X"}, {"id": "Y"}], "links": [{"source": "X", "target": "Y"}]},
    "confidence": 0.7,
}
MOCK_FCI_GRAPH = {
    "nodes": [{"name": "M", "node_type": "variable"}, {"name": "N", "node_type": "variable"}],
    "edges": [{"source": "M", "target": "N", "edge_type": "directed", "confidence": 0.65}],
    "graph_data": {"nodes": [{"id": "M"}, {"id": "N"}], "links": [{"source": "M"}]},
    "confidence": 0.65,
}

MOCK_DOWHY_ESTIMATE = {
    "ate": 0.25,
    "ate_ci_lower": 0.1,
    "ate_ci_upper": 0.4,
    "confidence_interval": {"lower": 0.1, "upper": 0.4},
    "sample_size": {"total": 1000},
    "assumptions_checked": {"positivity": True},
}

MOCK_ECONML_ESTIMATE = {
    "ate": 0.12,
    "ate_ci_lower": None,
    "ate_ci_upper": None,
    "confidence_interval": {},
    "sample_size": {"total": 800},
    "assumptions_checked": {"robust": True},
}


@pytest.fixture
def sample_dataframe():
    return create_mock_dataframe(
        rows=250,
        include_categorical=True,
        include_missing=True,
        include_outliers=True,
    )


@pytest.fixture
def numeric_dataframe():
    return create_mock_dataframe(
        rows=200,
        include_categorical=False,
        include_missing=False,
        include_outliers=False,
        include_binary=False,
    )


@pytest.fixture
def mock_kaggle_responses():
    return {
        "test-owner/test-dataset": {"data.csv": "col\n1\n2\n"},
        "test-competition": {"competition.csv": "col\n3\n4\n"},
    }


@pytest.fixture
def mock_kaggle_api(mock_kaggle_responses):
    return MockKaggleApi(
        dataset_files=mock_kaggle_responses,
        competition_files=mock_kaggle_responses,
    )


@pytest.fixture
def mock_llm_router():
    responses = [
        {
            "treatment_candidates": ["treatment"],
            "outcome_candidates": ["outcome"],
            "confounder_candidates": ["feature"],
            "data_quality_issues": [],
        },
        {"confidence": 0.9},
        {"overview": "Summary", "key_findings": [], "recommendations": ["Review"]},
    ]
    return MockLLMRouter(responses=responses)


@pytest.fixture
def mock_gcs_client():
    return MockGCSClient()


@pytest.fixture
def sample_state():
    return create_mock_state()


@pytest.fixture
def mock_circuit_breaker():
    from app.services.circuit_breaker import CircuitBreaker

    return CircuitBreaker(threshold=1, timeout_seconds=60)


async def create_analysis(
    db_session,
    *,
    kaggle_url: str = "https://www.kaggle.com/datasets/test/sample-dataset",
    config: dict[str, Any] | None = None,
    status: AnalysisStatus = AnalysisStatus.PENDING,
):
    return await analysis_crud.create(
        db_session,
        obj_in={
            "kaggle_url": kaggle_url,
            "config": config or {},
            "status": status,
        },
    )


async def create_dataset(
    db_session,
    *,
    analysis_id: uuid.UUID,
    kaggle_url: str = "https://www.kaggle.com/datasets/test/sample-dataset",
    cache_key: str | None = None,
    metadata: dict[str, Any] | None = None,
    characteristics: dict[str, Any] | None = None,
):
    return await dataset_crud.create(
        db_session,
        obj_in={
            "analysis_id": analysis_id,
            "kaggle_url": kaggle_url,
            "kaggle_dataset_id": "test-owner/test-dataset",
            "files": [{"name": "data.csv", "size_bytes": 100, "format": "csv"}],
            "selected_file": "data.csv",
            "selection_reasoning": "test",
            "gcs_path": "gs://bucket/data.csv",
            "cache_key": cache_key,
            "cache_expires_at": (
                datetime.now(timezone.utc) + timedelta(days=1) if cache_key else None
            ),
            "downloaded_at": datetime.now(timezone.utc),
            "metadata": metadata or {"local_path": "/tmp/data.csv"},
            "characteristics": characteristics or {},
        },
    )


async def create_data_understanding(
    db_session,
    *,
    dataset_id: uuid.UUID,
    summary_stats: dict[str, Any] | None = None,
    columns: list[dict[str, Any]] | None = None,
    quality_issues: list[dict[str, Any]] | None = None,
    quality_warnings: list[dict[str, Any]] | None = None,
    recommended_preprocessing: list[str] | None = None,
    treatment_candidates: list[str] | None = None,
    outcome_candidates: list[str] | None = None,
    confounder_candidates: list[str] | None = None,
):
    understanding = DataUnderstanding(
        dataset_id=dataset_id,
        summary_stats=summary_stats or {"row_count": 10, "column_count": 3},
        columns=columns or [{"name": "treatment", "dtype": "binary"}],
        quality_issues=quality_issues or [],
        quality_warnings=quality_warnings or [],
        recommended_preprocessing=recommended_preprocessing or [],
        treatment_candidates=treatment_candidates or ["treatment"],
        outcome_candidates=outcome_candidates or ["outcome"],
        confounder_candidates=confounder_candidates or ["feature"],
    )
    db_session.add(understanding)
    await db_session.commit()
    await db_session.refresh(understanding)
    return understanding


async def create_causal_graph(
    db_session,
    *,
    analysis_id: uuid.UUID,
    method: DiscoveryMethod = DiscoveryMethod.PC,
    edges: list[dict[str, Any]] | None = None,
    nodes: list[dict[str, Any]] | None = None,
    confidence: float | None = 0.8,
):
    graph = CausalGraph(
        analysis_id=analysis_id,
        method=method,
        edges=edges
        or [
            {"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.8}
        ],
        nodes=nodes
        or [
            {"name": "A", "node_type": "variable"},
            {"name": "B", "node_type": "variable"},
        ],
        graph_data={},
        algorithm_params={"alpha": 0.05},
        confidence=confidence,
        execution_time_seconds=0.1,
    )
    db_session.add(graph)
    await db_session.commit()
    await db_session.refresh(graph)
    return graph


async def create_treatment_effect(
    db_session,
    *,
    analysis_id: uuid.UUID,
    method: TreatmentMethod = TreatmentMethod.PROPENSITY_MATCHING,
    ate: float | None = 0.2,
):
    effect = TreatmentEffect(
        analysis_id=analysis_id,
        treatment_variable="treatment",
        outcome_variable="outcome",
        method=method,
        ate=ate,
        ate_ci_lower=0.1,
        ate_ci_upper=0.3,
        confidence_interval={"lower": 0.1, "upper": 0.3},
        confounders_adjusted=["feature"],
        sample_size={"total": 100},
        assumptions_checked={"positivity": True},
    )
    db_session.add(effect)
    await db_session.commit()
    await db_session.refresh(effect)
    return effect


async def create_validation_result(
    db_session,
    *,
    analysis_id: uuid.UUID,
    method: str = "placebo_treatment_refuter",
    passed: bool = True,
    confidence_score: float = 0.7,
):
    result = ValidationResult(
        analysis_id=analysis_id,
        validation_type=ValidationType.REFUTATION,
        method=method,
        passed=passed,
        confidence_score=confidence_score,
        details={"p_value": 0.6},
        recommendations=[],
    )
    db_session.add(result)
    await db_session.commit()
    await db_session.refresh(result)
    return result


async def create_report(
    db_session,
    *,
    analysis_id: uuid.UUID,
    report_type: ReportType = ReportType.EXECUTIVE,
    content: str = "{}",
):
    report = GeneratedReport(
        analysis_id=analysis_id,
        report_type=report_type,
        format=ReportFormat.JSON,
        content=content,
        generated_at=datetime.now(timezone.utc),
        file_size_bytes=len(content),
    )
    db_session.add(report)
    await db_session.commit()
    await db_session.refresh(report)
    return report
