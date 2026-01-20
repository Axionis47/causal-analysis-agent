"""Prompt template for EDA interpretation."""

from __future__ import annotations

import json
from typing import Any

TEMPLATE = (
    "Analyze this dataset summary and suggest causal analysis hints.\n"
    "Summary: {summary_stats}\n"
    "Columns: {columns}\n"
    "Top correlations: {correlations}\n"
    "Quality issues: {quality_issues}\n"
    "Return JSON with treatment_candidates, outcome_candidates, confounder_candidates, data_quality_issues."
)


def build_eda_prompt(
    summary_stats: dict[str, Any],
    columns: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
) -> str:
    return TEMPLATE.format(
        summary_stats=json.dumps(summary_stats),
        columns=json.dumps(columns[:10]),
        correlations=json.dumps(correlations),
        quality_issues=json.dumps(quality_issues),
    )
