"""Prompt template for report summaries."""

from __future__ import annotations

import json
from typing import Any

TEMPLATE = (
    "Create an executive summary for this causal analysis in JSON with keys: "
    "overview, key_findings, recommendations. Data: {payload}"
)


def build_executive_summary_prompt(payload: dict[str, Any]) -> str:
    return TEMPLATE.format(payload=json.dumps(payload))
