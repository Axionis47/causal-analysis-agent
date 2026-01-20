"""Service for comparing multiple causal analyses."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_graph_similarity(graph1: dict[str, Any], graph2: dict[str, Any]) -> float:
    """
    Calculate Jaccard similarity of edge sets between two graphs.

    Args:
        graph1: First graph with 'edges' list
        graph2: Second graph with 'edges' list

    Returns:
        Jaccard index (0-1) measuring edge set overlap
    """
    def edge_key(edge: dict[str, Any]) -> str:
        return f"{edge.get('source', '')}→{edge.get('target', '')}"

    edges1 = set(edge_key(e) for e in graph1.get("edges", []))
    edges2 = set(edge_key(e) for e in graph2.get("edges", []))

    if not edges1 and not edges2:
        return 1.0  # Both empty = identical

    intersection = edges1 & edges2
    union = edges1 | edges2

    if not union:
        return 1.0

    return len(intersection) / len(union)


def compute_node_similarity(graph1: dict[str, Any], graph2: dict[str, Any]) -> float:
    """
    Calculate Jaccard similarity of node sets between two graphs.

    Args:
        graph1: First graph with 'nodes' list
        graph2: Second graph with 'nodes' list

    Returns:
        Jaccard index (0-1) measuring node set overlap
    """
    def node_key(node: dict[str, Any]) -> str:
        if isinstance(node, dict):
            return node.get("name", str(node))
        return str(node)

    nodes1 = set(node_key(n) for n in graph1.get("nodes", []))
    nodes2 = set(node_key(n) for n in graph2.get("nodes", []))

    if not nodes1 and not nodes2:
        return 1.0

    intersection = nodes1 & nodes2
    union = nodes1 | nodes2

    if not union:
        return 1.0

    return len(intersection) / len(union)


def compute_effect_difference(
    effect1: dict[str, Any],
    effect2: dict[str, Any]
) -> float | None:
    """
    Calculate percentage difference in ATE between two treatment effects.

    Args:
        effect1: First effect with 'ate' value
        effect2: Second effect with 'ate' value

    Returns:
        Percentage difference or None if ATEs are missing
    """
    ate1 = effect1.get("ate")
    ate2 = effect2.get("ate")

    if ate1 is None or ate2 is None:
        return None

    try:
        ate1_float = float(ate1)
        ate2_float = float(ate2)
    except (TypeError, ValueError):
        return None

    if ate1_float == 0 and ate2_float == 0:
        return 0.0

    # Use the average as base for percentage calculation
    avg_ate = (abs(ate1_float) + abs(ate2_float)) / 2
    if avg_ate == 0:
        return 0.0

    return abs(ate1_float - ate2_float) / avg_ate


def compute_ate_differences(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute ATE differences across multiple analyses for matching treatment-outcome pairs.

    Args:
        analyses: List of analysis data with treatment_effects

    Returns:
        List of comparison records with ATE values and differences
    """
    # Group effects by treatment-outcome pair
    effect_groups: dict[str, dict[str, Any]] = {}

    for analysis in analyses:
        analysis_id = analysis.get("id", "unknown")
        for effect in analysis.get("treatment_effects", []):
            treatment = effect.get("treatment") or effect.get("treatment_variable")
            outcome = effect.get("outcome") or effect.get("outcome_variable")

            if not treatment or not outcome:
                continue

            key = f"{treatment}→{outcome}"
            if key not in effect_groups:
                effect_groups[key] = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "values": {},
                }
            effect_groups[key]["values"][analysis_id] = effect.get("ate")

    # Calculate max differences
    result = []
    for key, group in effect_groups.items():
        values = group["values"]
        ate_values = [v for v in values.values() if v is not None]

        max_diff = None
        if len(ate_values) >= 2:
            max_diff = max(ate_values) - min(ate_values)

        result.append({
            "treatment": group["treatment"],
            "outcome": group["outcome"],
            "values": values,
            "max_difference": max_diff,
        })

    return result


def generate_comparison_summary(
    analyses: list[dict[str, Any]],
    comparison_metrics: dict[str, Any]
) -> dict[str, Any]:
    """
    Generate a structured summary of the comparison.

    Args:
        analyses: List of analysis data
        comparison_metrics: Computed comparison metrics

    Returns:
        Summary dictionary with key insights
    """
    num_analyses = len(analyses)
    graph_similarity = comparison_metrics.get("graph_similarity", 0)
    ate_diffs = comparison_metrics.get("ate_differences", [])

    # Count significant differences (>10%)
    significant_diffs = [
        d for d in ate_diffs
        if d.get("max_difference") is not None
        and any(
            abs(d["max_difference"]) > 0.1 * abs(v)
            for v in d["values"].values()
            if v is not None
        )
    ]

    return {
        "num_analyses": num_analyses,
        "graph_similarity_percent": round(graph_similarity * 100, 1),
        "total_effects_compared": len(ate_diffs),
        "significant_differences": len(significant_diffs),
        "agreement_level": (
            "high" if graph_similarity > 0.8
            else "moderate" if graph_similarity > 0.5
            else "low"
        ),
    }


def compare_analyses(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Main comparison function that computes all metrics.

    Args:
        analyses: List of analysis data with graphs and effects

    Returns:
        Complete comparison result with metrics and summary
    """
    if len(analyses) < 2:
        raise ValueError("At least 2 analyses required for comparison")

    # Extract graphs
    graphs = []
    for analysis in analyses:
        causal_graphs = analysis.get("causal_graphs", [])
        if causal_graphs:
            graphs.append(causal_graphs[0])
        else:
            graphs.append({"nodes": [], "edges": []})

    # Compute pairwise graph similarity (average for >2 analyses)
    similarities = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            similarities.append(compute_graph_similarity(graphs[i], graphs[j]))

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0

    # Compute ATE differences
    ate_differences = compute_ate_differences(analyses)

    # Compute confidence deltas
    confidence_deltas = {}
    for analysis in analyses:
        analysis_id = analysis.get("id", "unknown")
        avg_confidence = None
        causal_graphs = analysis.get("causal_graphs", [])
        if causal_graphs:
            confidences = [g.get("confidence") for g in causal_graphs if g.get("confidence")]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        confidence_deltas[analysis_id] = avg_confidence

    comparison_metrics = {
        "graph_similarity": avg_similarity,
        "ate_differences": ate_differences,
        "confidence_deltas": confidence_deltas,
    }

    return {
        "analyses": analyses,
        "comparison_metrics": comparison_metrics,
        "summary": generate_comparison_summary(analyses, comparison_metrics),
    }
