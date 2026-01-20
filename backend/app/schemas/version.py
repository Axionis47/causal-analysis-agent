"""Pydantic schemas for version operations."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VersionResponse(BaseModel):
    """Response schema for a single version."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    analysis_id: uuid.UUID
    version_number: int
    config: dict[str, Any]
    config_hash: str
    changed_by: uuid.UUID | None
    change_summary: str | None
    is_manual_snapshot: bool
    parent_version_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime


class VersionListResponse(BaseModel):
    """Paginated response for version list."""

    items: list[VersionResponse]
    total: int
    skip: int
    limit: int


class SnapshotRequest(BaseModel):
    """Request schema for creating a manual snapshot."""

    change_summary: str | None = Field(
        None,
        max_length=500,
        description="Optional description of this snapshot",
    )


class RevertRequest(BaseModel):
    """Request schema for reverting to a specific version."""

    version_number: int = Field(
        ...,
        gt=0,
        description="Version number to revert to",
    )
    confirmation: bool = Field(
        ...,
        description="Must be True to confirm the revert operation",
    )


class ConfigDiff(BaseModel):
    """Schema for config diff results."""

    added: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields added in the newer version",
    )
    removed: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields removed in the newer version",
    )
    modified: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields modified between versions",
    )
    unchanged: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields unchanged between versions",
    )


class VersionComparisonResponse(BaseModel):
    """Response schema for comparing two versions."""

    version1: VersionResponse
    version2: VersionResponse
    config_diff: ConfigDiff
    diff_summary: str = Field(
        ...,
        description="Human-readable summary like '3 fields added, 2 modified'",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between 0 and 1",
    )
    results_comparison: dict[str, Any] | None = Field(
        None,
        description="Comparison of analysis results if available",
    )


class VersionTimelineEntry(BaseModel):
    """Schema for a single timeline entry."""

    version_number: int
    created_at: datetime | None
    changed_by: uuid.UUID | None
    change_summary: str | None
    is_manual_snapshot: bool
    config_hash_short: str = Field(
        ...,
        description="Truncated config hash for display",
    )
    parent_version_id: uuid.UUID | None = None
    label: str = Field(
        ...,
        description="Display label like 'v3 (Manual snapshot)'",
    )


class VersionTimelineResponse(BaseModel):
    """Response schema for version timeline."""

    analysis_id: uuid.UUID
    timeline: list[VersionTimelineEntry]
    total_versions: int
    latest_version: int


class VersionStatsResponse(BaseModel):
    """Response schema for version statistics."""

    analysis_id: uuid.UUID
    total_versions: int
    latest_version_number: int
    auto_versions: int
    manual_snapshots: int
    reverts: int
    first_version_date: datetime | None
    last_version_date: datetime | None
