"""LangGraph orchestrator for causal analysis pipeline."""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.orchestrator.state import AnalysisState

logger = logging.getLogger(__name__)


def _select_after_eda(state: AnalysisState) -> str:
    analysis_types = [t.lower() for t in state.get("analysis_types", [])]
    needs_treatment = any(
        t in analysis_types for t in ["treatment_effects", "mediation", "heterogeneous", "time_varying", "iv"]
    )
    if "causal_discovery" in analysis_types:
        return "discovery"
    if needs_treatment:
        return "treatment"
    return "report"


def _select_after_discovery(state: AnalysisState) -> str:
    analysis_types = [t.lower() for t in state.get("analysis_types", [])]
    needs_treatment = any(
        t in analysis_types for t in ["treatment_effects", "mediation", "heterogeneous", "time_varying", "iv"]
    )
    if needs_treatment:
        return "treatment"
    return "report"


def build_graph(
    data_agent: Callable[[AnalysisState], Any],
    eda_agent: Callable[[AnalysisState], Any],
    discovery_agent: Callable[[AnalysisState], Any],
    treatment_agent: Callable[[AnalysisState], Any],
    validation_agent: Callable[[AnalysisState], Any],
    report_agent: Callable[[AnalysisState], Any],
) -> StateGraph:
    graph = StateGraph(AnalysisState)
    graph.add_node("data", data_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("discovery", discovery_agent)
    graph.add_node("treatment", treatment_agent)
    graph.add_node("validation", validation_agent)
    graph.add_node("report", report_agent)

    graph.set_entry_point("data")
    graph.add_edge("data", "eda")
    graph.add_conditional_edges(
        "eda",
        _select_after_eda,
        {"discovery": "discovery", "treatment": "treatment", "report": "report"},
    )
    graph.add_conditional_edges(
        "discovery",
        _select_after_discovery,
        {"treatment": "treatment", "report": "report"},
    )
    graph.add_edge("treatment", "validation")
    graph.add_edge("validation", "report")
    graph.add_edge("report", END)
    return graph
