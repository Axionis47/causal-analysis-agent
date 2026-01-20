"""Utility functions for analysis versioning."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.models.analysis_version import AnalysisVersion


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of JSON-serialized config (sorted keys).

    Args:
        config: Configuration dictionary

    Returns:
        64-character hex string hash
    """
    config_json = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()


def is_config_changed(
    old_config: dict[str, Any],
    new_config: dict[str, Any],
) -> bool:
    """
    Check if config has changed using deep equality.

    Args:
        old_config: Previous config
        new_config: New config

    Returns:
        True if configs differ, False if identical
    """
    return compute_config_hash(old_config) != compute_config_hash(new_config)


def extract_config_changes(
    old_config: dict[str, Any],
    new_config: dict[str, Any],
    path: str = "",
) -> list[str]:
    """
    Extract list of changed field paths between two configs.

    Args:
        old_config: Previous config
        new_config: New config
        path: Current path prefix for recursion

    Returns:
        List of changed field paths like ["treatment_variable", "config.analysis_types"]
    """
    changes: list[str] = []
    all_keys = set(old_config.keys()) | set(new_config.keys())

    for key in all_keys:
        full_path = f"{path}.{key}" if path else key
        in_old = key in old_config
        in_new = key in new_config

        if in_old and not in_new:
            changes.append(f"{full_path} (removed)")
        elif not in_old and in_new:
            changes.append(f"{full_path} (added)")
        elif in_old and in_new:
            old_val = old_config[key]
            new_val = new_config[key]

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                changes.extend(extract_config_changes(old_val, new_val, full_path))
            elif isinstance(old_val, list) and isinstance(new_val, list):
                if old_val != new_val:
                    changes.append(f"{full_path} (modified)")
            elif old_val != new_val:
                changes.append(f"{full_path} (modified)")

    return changes


def validate_version_number(
    version_number: int,
    max_version: int,
) -> bool:
    """
    Validate that a version number exists.

    Args:
        version_number: Version number to validate
        max_version: Maximum version number in the analysis

    Returns:
        True if version exists, False otherwise
    """
    return 1 <= version_number <= max_version


def format_version_label(version: AnalysisVersion) -> str:
    """
    Generate display label for a version.

    Args:
        version: AnalysisVersion object

    Returns:
        Display label like "v3 (Manual snapshot by user@example.com)"
    """
    parts = [f"v{version.version_number}"]

    if version.is_manual_snapshot:
        parts.append("(Manual snapshot)")
    elif version.parent_version_id:
        parts.append("(Reverted)")

    if version.change_summary:
        summary = version.change_summary
        if len(summary) > 40:
            summary = summary[:37] + "..."
        parts.append(f"- {summary}")

    return " ".join(parts)


def get_version_diff_stats(
    old_config: dict[str, Any],
    new_config: dict[str, Any],
) -> dict[str, int]:
    """
    Get statistics about config changes.

    Args:
        old_config: Previous config
        new_config: New config

    Returns:
        Dict with counts: added, removed, modified, unchanged
    """
    added = 0
    removed = 0
    modified = 0
    unchanged = 0

    all_keys = set(old_config.keys()) | set(new_config.keys())

    for key in all_keys:
        in_old = key in old_config
        in_new = key in new_config

        if in_old and not in_new:
            removed += 1
        elif not in_old and in_new:
            added += 1
        elif old_config[key] != new_config[key]:
            modified += 1
        else:
            unchanged += 1

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
        "total_changes": added + removed + modified,
    }


def serialize_version(version: AnalysisVersion) -> dict[str, Any]:
    """
    Serialize an AnalysisVersion to a dictionary.

    Args:
        version: AnalysisVersion object

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    return {
        "id": str(version.id),
        "analysis_id": str(version.analysis_id),
        "version_number": version.version_number,
        "config": version.config,
        "config_hash": version.config_hash,
        "changed_by": str(version.changed_by) if version.changed_by else None,
        "change_summary": version.change_summary,
        "is_manual_snapshot": version.is_manual_snapshot,
        "parent_version_id": str(version.parent_version_id) if version.parent_version_id else None,
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "updated_at": version.updated_at.isoformat() if version.updated_at else None,
    }
