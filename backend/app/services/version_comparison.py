"""Service for comparing analysis versions."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.models.analysis_version import AnalysisVersion

logger = get_logger(__name__)


def compute_config_diff(
    config1: dict[str, Any],
    config2: dict[str, Any],
    path: str = "",
) -> dict[str, Any]:
    """
    Compute deep diff between two config dictionaries.

    Args:
        config1: First config (older version)
        config2: Second config (newer version)
        path: Current path in the config tree (for nested keys)

    Returns:
        Dict with keys: added, removed, modified, unchanged
    """
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    modified: dict[str, Any] = {}
    unchanged: dict[str, Any] = {}

    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        full_path = f"{path}.{key}" if path else key
        in_config1 = key in config1
        in_config2 = key in config2

        if in_config1 and not in_config2:
            removed[full_path] = config1[key]
        elif not in_config1 and in_config2:
            added[full_path] = config2[key]
        elif in_config1 and in_config2:
            val1 = config1[key]
            val2 = config2[key]

            # Handle nested dicts recursively
            if isinstance(val1, dict) and isinstance(val2, dict):
                nested_diff = compute_config_diff(val1, val2, full_path)
                added.update(nested_diff["added"])
                removed.update(nested_diff["removed"])
                modified.update(nested_diff["modified"])
                unchanged.update(nested_diff["unchanged"])
            elif val1 != val2:
                modified[full_path] = {
                    "old": val1,
                    "new": val2,
                }
            else:
                unchanged[full_path] = val1

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
    }


def generate_diff_summary(diff: dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the diff.

    Args:
        diff: Diff dict from compute_config_diff

    Returns:
        Summary string like "3 fields added, 2 modified, 1 removed"
    """
    added_count = len(diff.get("added", {}))
    removed_count = len(diff.get("removed", {}))
    modified_count = len(diff.get("modified", {}))

    parts = []
    if added_count:
        parts.append(f"{added_count} field{'s' if added_count != 1 else ''} added")
    if modified_count:
        parts.append(f"{modified_count} field{'s' if modified_count != 1 else ''} modified")
    if removed_count:
        parts.append(f"{removed_count} field{'s' if removed_count != 1 else ''} removed")

    if not parts:
        return "No changes"

    return ", ".join(parts)


def compute_version_similarity(
    config1: dict[str, Any],
    config2: dict[str, Any],
) -> float:
    """
    Calculate similarity score (0-1) between two configs.

    Uses Jaccard-like similarity based on matching keys and values.

    Args:
        config1: First config
        config2: Second config

    Returns:
        Similarity score between 0 and 1
    """
    diff = compute_config_diff(config1, config2)

    added_count = len(diff.get("added", {}))
    removed_count = len(diff.get("removed", {}))
    modified_count = len(diff.get("modified", {}))
    unchanged_count = len(diff.get("unchanged", {}))

    total_fields = added_count + removed_count + modified_count + unchanged_count

    if total_fields == 0:
        return 1.0  # Both empty = identical

    # Count unchanged as full match, modified as half match
    matching = unchanged_count + (modified_count * 0.5)

    return matching / total_fields


def compare_version_results(
    version1_data: dict[str, Any],
    version2_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare analysis results between two versions.

    Reuses comparison logic from the main comparison service.

    Args:
        version1_data: Analysis data for version 1
        version2_data: Analysis data for version 2

    Returns:
        Comparison results including graph and effect differences
    """
    from app.services.comparison import (
        compute_graph_similarity,
        compute_effect_difference,
    )

    result: dict[str, Any] = {
        "graph_comparison": None,
        "effect_comparison": None,
    }

    # Compare graphs if available
    graphs1 = version1_data.get("causal_graphs", [])
    graphs2 = version2_data.get("causal_graphs", [])

    if graphs1 and graphs2:
        graph1 = graphs1[0] if isinstance(graphs1, list) else graphs1
        graph2 = graphs2[0] if isinstance(graphs2, list) else graphs2

        result["graph_comparison"] = {
            "similarity": compute_graph_similarity(graph1, graph2),
        }

    # Compare treatment effects if available
    effects1 = version1_data.get("treatment_effects", [])
    effects2 = version2_data.get("treatment_effects", [])

    if effects1 and effects2:
        effect_comparisons = []
        for e1 in effects1:
            for e2 in effects2:
                # Match by treatment and outcome
                treatment1 = e1.get("treatment") or e1.get("treatment_variable")
                treatment2 = e2.get("treatment") or e2.get("treatment_variable")
                outcome1 = e1.get("outcome") or e1.get("outcome_variable")
                outcome2 = e2.get("outcome") or e2.get("outcome_variable")

                if treatment1 == treatment2 and outcome1 == outcome2:
                    diff = compute_effect_difference(e1, e2)
                    effect_comparisons.append({
                        "treatment": treatment1,
                        "outcome": outcome1,
                        "ate_v1": e1.get("ate"),
                        "ate_v2": e2.get("ate"),
                        "difference": diff,
                    })

        result["effect_comparison"] = effect_comparisons

    return result


def generate_version_timeline(
    versions: list[AnalysisVersion],
) -> list[dict[str, Any]]:
    """
    Create timeline visualization data from version history.

    Args:
        versions: List of AnalysisVersion objects (ordered by version_number desc)

    Returns:
        List of timeline entries with version info and metadata
    """
    timeline = []

    for version in versions:
        entry = {
            "version_number": version.version_number,
            "created_at": version.created_at.isoformat() if version.created_at else None,
            "changed_by": str(version.changed_by) if version.changed_by else None,
            "change_summary": version.change_summary,
            "is_manual_snapshot": version.is_manual_snapshot,
            "config_hash": version.config_hash[:12] + "...",  # Truncated for display
            "parent_version_id": str(version.parent_version_id) if version.parent_version_id else None,
        }

        # Add label for special versions
        if version.is_manual_snapshot:
            entry["label"] = f"v{version.version_number} (Manual snapshot)"
        elif version.parent_version_id:
            entry["label"] = f"v{version.version_number} (Reverted)"
        else:
            entry["label"] = f"v{version.version_number}"

        timeline.append(entry)

    return timeline


def extract_config_changes(
    old_config: dict[str, Any],
    new_config: dict[str, Any],
) -> list[str]:
    """
    Extract list of changed field paths between two configs.

    Args:
        old_config: Previous config
        new_config: New config

    Returns:
        List of changed field paths like ["treatment_variable", "config.analysis_types[0]"]
    """
    diff = compute_config_diff(old_config, new_config)

    changed_paths = []
    changed_paths.extend(diff.get("added", {}).keys())
    changed_paths.extend(diff.get("removed", {}).keys())
    changed_paths.extend(diff.get("modified", {}).keys())

    return sorted(changed_paths)


def format_version_label(version: AnalysisVersion) -> str:
    """
    Generate a display label for a version.

    Args:
        version: AnalysisVersion object

    Returns:
        Display label like "v3 (Manual snapshot by user@example.com)"
    """
    label = f"v{version.version_number}"

    if version.is_manual_snapshot:
        label += " (Manual snapshot)"
    elif version.parent_version_id:
        label += " (Reverted)"

    if version.change_summary:
        # Truncate long summaries
        summary = version.change_summary
        if len(summary) > 50:
            summary = summary[:47] + "..."
        label += f": {summary}"

    return label
