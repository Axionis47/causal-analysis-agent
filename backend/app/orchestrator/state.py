"""Shared state for the LangGraph orchestrator."""

from __future__ import annotations

from typing import Any, TypedDict


class AnalysisState(TypedDict, total=False):
    analysis_id: str
    kaggle_url: str
    dataset_id: str
    data_characteristics: dict[str, Any]
    causal_graph_ids: list[str]
    treatment_effect_ids: list[str]
    validation_result_ids: list[str]
    report_ids: list[str]
    questions: list[dict[str, Any]]
    analysis_types: list[str]
    progress_percent: int
    errors: list[str]
    extra_results: dict[str, Any]
    partial_outputs: dict[str, Any]
    config: dict[str, Any]
