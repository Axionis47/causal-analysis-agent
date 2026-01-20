"""Prompt template for causal graph interpretation."""

from __future__ import annotations

import json
from typing import Any

TEMPLATE = (
    "Interpret the causal graph and return JSON with confidence (0-1), "
    "treatment_candidates, outcome_candidates. "
    "Nodes: {nodes} "
    "Edges: {edges}"
)


def build_discovery_prompt(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    return TEMPLATE.format(nodes=json.dumps(nodes), edges=json.dumps(edges))
