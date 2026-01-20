"""API routes for analysis version operations."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.crud.analysis import analysis_crud
from app.crud.analysis_version import analysis_version_crud
from app.db.database import get_async_session
from app.models.analysis_version import AnalysisVersion
from app.schemas.version import (
    ConfigDiff,
    RevertRequest,
    SnapshotRequest,
    VersionComparisonResponse,
    VersionListResponse,
    VersionResponse,
    VersionStatsResponse,
    VersionTimelineEntry,
    VersionTimelineResponse,
)
from app.services.version_comparison import (
    compute_config_diff,
    compute_version_similarity,
    generate_diff_summary,
    generate_version_timeline,
)

router = APIRouter(prefix="/api/v1/analyses/{analysis_id}/versions", tags=["versions"])
logger = logging.getLogger(__name__)


async def verify_analysis_access(
    analysis_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> None:
    """Verify user has access to the analysis."""
    analysis = await analysis_crud.get(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )
    if analysis.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this analysis",
        )


@router.get(
    "",
    response_model=VersionListResponse,
    summary="List analysis versions",
    responses={
        200: {
            "description": "Paginated list of versions",
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                                "version_number": 3,
                                "config": {"treatment": "x", "outcome": "y"},
                                "config_hash": "abc123...",
                                "changed_by": "550e8400-e29b-41d4-a716-446655440002",
                                "change_summary": "Updated treatment variable",
                                "is_manual_snapshot": False,
                                "created_at": "2026-01-20T15:00:00Z",
                            }
                        ],
                        "total": 3,
                        "skip": 0,
                        "limit": 100,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def list_versions(
    analysis_id: uuid.UUID,
    skip: int = Query(0, ge=0, description="Number of versions to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum versions to return"),
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionListResponse:
    """
    List all versions for an analysis with pagination.

    Returns the complete version history, ordered by version number
    descending (newest first).

    **Version Types:**
    - **Automatic**: Created when analysis config is updated
    - **Manual Snapshot**: Explicitly created by user
    - **Revert**: Created when reverting to a previous version

    **Version Metadata:**
    - `version_number`: Sequential version identifier
    - `config`: Full configuration at this version
    - `config_hash`: Hash for quick comparison
    - `change_summary`: Description of changes
    - `is_manual_snapshot`: Whether manually created
    """
    await verify_analysis_access(analysis_id, user_id, db)

    # Get total count
    total = await analysis_version_crud.get_version_count(db, analysis_id)

    # Get versions
    versions = await analysis_version_crud.get_by_analysis(
        db, analysis_id, skip=skip, limit=limit
    )

    return VersionListResponse(
        items=versions,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/timeline",
    response_model=VersionTimelineResponse,
    summary="Get version timeline",
    responses={
        200: {
            "description": "Version history formatted for timeline visualization",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                        "timeline": [
                            {
                                "version_number": 3,
                                "created_at": "2026-01-20T15:00:00Z",
                                "changed_by": "550e8400-e29b-41d4-a716-446655440002",
                                "change_summary": "Updated treatment variable",
                                "is_manual_snapshot": False,
                                "config_hash_short": "abc123...",
                                "parent_version_id": None,
                                "label": "v3",
                            }
                        ],
                        "total_versions": 3,
                        "latest_version": 3,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_version_timeline(
    analysis_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionTimelineResponse:
    """
    Get version history as a timeline for visualization.

    Returns all versions with metadata optimized for rendering
    a visual timeline in the UI.

    **Timeline Entry Fields:**
    - `version_number`: Version identifier
    - `created_at`: When version was created
    - `changed_by`: User who made the change
    - `change_summary`: Brief description
    - `is_manual_snapshot`: Whether user explicitly created
    - `config_hash_short`: Truncated hash for display
    - `parent_version_id`: For revert relationships
    - `label`: Display label (e.g., "v3")

    **Visualization Uses:**
    - Show version history graph
    - Highlight manual snapshots vs automatic
    - Show revert lineage
    """
    await verify_analysis_access(analysis_id, user_id, db)

    versions = await analysis_version_crud.get_by_analysis(db, analysis_id, limit=1000)
    timeline_data = generate_version_timeline(versions)

    latest_version = await analysis_version_crud.get_latest_version_number(db, analysis_id)

    return VersionTimelineResponse(
        analysis_id=analysis_id,
        timeline=[
            VersionTimelineEntry(
                version_number=entry["version_number"],
                created_at=entry["created_at"],
                changed_by=uuid.UUID(entry["changed_by"]) if entry["changed_by"] else None,
                change_summary=entry["change_summary"],
                is_manual_snapshot=entry["is_manual_snapshot"],
                config_hash_short=entry["config_hash"],
                parent_version_id=uuid.UUID(entry["parent_version_id"]) if entry["parent_version_id"] else None,
                label=entry["label"],
            )
            for entry in timeline_data
        ],
        total_versions=len(versions),
        latest_version=latest_version,
    )


@router.get(
    "/stats",
    response_model=VersionStatsResponse,
    summary="Get version statistics",
    responses={
        200: {
            "description": "Statistics about version history for the analysis",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                        "total_versions": 5,
                        "latest_version_number": 5,
                        "auto_versions": 3,
                        "manual_snapshots": 1,
                        "reverts": 1,
                        "first_version_date": "2026-01-15T10:00:00Z",
                        "last_version_date": "2026-01-20T15:30:00Z",
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_version_stats(
    analysis_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionStatsResponse:
    """
    Get statistics about version history for an analysis.

    Returns aggregate statistics about the version history, useful for
    understanding how actively an analysis has been modified.

    **Statistics Included:**
    - `total_versions`: Total number of versions created
    - `latest_version_number`: Current version number
    - `auto_versions`: Versions created automatically on config changes
    - `manual_snapshots`: Versions explicitly created by users
    - `reverts`: Versions created by reverting to previous versions
    - `first_version_date`: When the analysis was first created
    - `last_version_date`: When the last modification was made

    **Request Example:**
    ```
    GET /api/v1/analyses/{analysis_id}/versions/stats
    Authorization: Bearer <token>
    ```
    """
    await verify_analysis_access(analysis_id, user_id, db)

    versions = await analysis_version_crud.get_by_analysis(db, analysis_id, limit=1000)

    auto_versions = sum(1 for v in versions if not v.is_manual_snapshot and not v.parent_version_id)
    manual_snapshots = sum(1 for v in versions if v.is_manual_snapshot)
    reverts = sum(1 for v in versions if v.parent_version_id)

    first_version_date = min((v.created_at for v in versions), default=None)
    last_version_date = max((v.created_at for v in versions), default=None)

    latest_version = await analysis_version_crud.get_latest_version_number(db, analysis_id)

    return VersionStatsResponse(
        analysis_id=analysis_id,
        total_versions=len(versions),
        latest_version_number=latest_version,
        auto_versions=auto_versions,
        manual_snapshots=manual_snapshots,
        reverts=reverts,
        first_version_date=first_version_date,
        last_version_date=last_version_date,
    )


@router.get(
    "/{version_number}",
    response_model=VersionResponse,
    summary="Get specific version",
    responses={
        200: {
            "description": "Version details with full configuration",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                        "version_number": 3,
                        "config": {
                            "treatment_variable": "education",
                            "outcome_variable": "income",
                            "analysis_type": "treatment_effects",
                            "discovery_method": "auto",
                        },
                        "config_hash": "abc123def456789...",
                        "changed_by": "550e8400-e29b-41d4-a716-446655440002",
                        "change_summary": "Updated treatment variable from age to education",
                        "is_manual_snapshot": False,
                        "parent_version_id": None,
                        "created_at": "2026-01-20T14:30:00Z",
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {
            "description": "Analysis or version not found",
            "content": {
                "application/json": {
                    "examples": {
                        "analysis_not_found": {"value": {"detail": "Analysis not found"}},
                        "version_not_found": {"value": {"detail": "Version 3 not found"}},
                    }
                }
            },
        },
    },
)
async def get_version(
    analysis_id: uuid.UUID,
    version_number: int,
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionResponse:
    """
    Get a specific version by version number.

    Returns the full details of a specific version, including the complete
    configuration state at that point in time.

    **Response Fields:**
    - `id`: Unique identifier for this version record
    - `analysis_id`: The analysis this version belongs to
    - `version_number`: Sequential version number (1, 2, 3, ...)
    - `config`: Full configuration snapshot at this version
    - `config_hash`: Hash for quick comparison between versions
    - `changed_by`: User ID who created this version
    - `change_summary`: Description of what changed
    - `is_manual_snapshot`: Whether this was explicitly created by user
    - `parent_version_id`: For reverts, links to the source version
    - `created_at`: Timestamp when this version was created

    **Request Example:**
    ```
    GET /api/v1/analyses/{analysis_id}/versions/3
    Authorization: Bearer <token>
    ```
    """
    await verify_analysis_access(analysis_id, user_id, db)

    version = await analysis_version_crud.get_version(db, analysis_id, version_number)
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_number} not found",
        )

    return version


@router.post(
    "/snapshot",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create manual snapshot",
    responses={
        201: {
            "description": "Snapshot created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                        "version_number": 4,
                        "config": {"treatment": "x", "outcome": "y"},
                        "config_hash": "def456...",
                        "changed_by": "550e8400-e29b-41d4-a716-446655440002",
                        "change_summary": "Pre-deployment checkpoint",
                        "is_manual_snapshot": True,
                        "created_at": "2026-01-20T16:00:00Z",
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def create_snapshot(
    analysis_id: uuid.UUID,
    snapshot_request: SnapshotRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionResponse:
    """
    Create a manual snapshot of the current analysis configuration.

    This creates a new version without modifying the analysis itself.
    Manual snapshots are useful for:

    - Creating checkpoints before making changes
    - Marking significant milestones
    - Preserving known-good configurations

    **Request Example:**
    ```json
    {
        "change_summary": "Pre-deployment checkpoint"
    }
    ```

    **Automatic vs Manual Versions:**
    - Automatic versions are created when config changes
    - Manual snapshots are explicitly created by users
    - Both can be used as revert targets

    The response includes the new version number and full configuration.
    """
    await verify_analysis_access(analysis_id, user_id, db)

    # Extract request metadata for audit logging
    client_ip = request.client.host if request.client else None
    user_agent_header = request.headers.get("user-agent")
    request_id = request.headers.get("x-request-id")

    try:
        version = await analysis_crud.create_manual_snapshot(
            db,
            analysis_id=analysis_id,
            changed_by=user_id,
            change_summary=snapshot_request.change_summary,
            ip_address=client_ip,
            user_agent=user_agent_header,
            request_id=request_id,
        )

        logger.info(
            "Manual snapshot created",
            extra={
                "analysis_id": str(analysis_id),
                "version_number": version.version_number,
                "user_id": str(user_id),
            },
        )

        return version

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/{v1}/compare/{v2}",
    response_model=VersionComparisonResponse,
    summary="Compare two versions",
    responses={
        200: {
            "description": "Detailed comparison between two versions",
            "content": {
                "application/json": {
                    "example": {
                        "version1": {"version_number": 1, "config": {}},
                        "version2": {"version_number": 3, "config": {}},
                        "config_diff": {
                            "added": {"new_field": "value"},
                            "removed": {"old_field": "old_value"},
                            "modified": {
                                "treatment_variable": {
                                    "old": "x",
                                    "new": "y",
                                }
                            },
                            "unchanged": {"outcome_variable": "z"},
                        },
                        "diff_summary": "2 fields added, 1 removed, 1 modified",
                        "similarity_score": 0.75,
                        "results_comparison": None,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {
            "description": "Analysis or version not found",
            "content": {
                "application/json": {
                    "examples": {
                        "version1": {"value": {"detail": "Version 1 not found"}},
                        "version2": {"value": {"detail": "Version 3 not found"}},
                    }
                }
            },
        },
    },
)
async def compare_versions(
    analysis_id: uuid.UUID,
    v1: int,
    v2: int,
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionComparisonResponse:
    """
    Compare two versions of an analysis configuration.

    Returns a detailed diff showing what changed between versions,
    along with a similarity score.

    **Diff Categories:**
    - `added`: Fields present in v2 but not v1
    - `removed`: Fields present in v1 but not v2
    - `modified`: Fields with different values (shows old/new)
    - `unchanged`: Fields with identical values

    **Similarity Score:**
    - 1.0 = Identical configurations
    - 0.0 = Completely different
    - Calculated based on field overlap and value matching

    **Use Cases:**
    - Review changes before reverting
    - Understand configuration evolution
    - Debug unexpected behavior changes
    """
    await verify_analysis_access(analysis_id, user_id, db)

    version1 = await analysis_version_crud.get_version(db, analysis_id, v1)
    version2 = await analysis_version_crud.get_version(db, analysis_id, v2)

    if version1 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {v1} not found",
        )
    if version2 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {v2} not found",
        )

    # Compute diff
    diff = compute_config_diff(version1.config, version2.config)
    summary = generate_diff_summary(diff)
    similarity = compute_version_similarity(version1.config, version2.config)

    return VersionComparisonResponse(
        version1=version1,
        version2=version2,
        config_diff=ConfigDiff(
            added=diff["added"],
            removed=diff["removed"],
            modified=diff["modified"],
            unchanged=diff["unchanged"],
        ),
        diff_summary=summary,
        similarity_score=similarity,
        results_comparison=None,  # TODO: Add results comparison if both versions have results
    )


@router.post(
    "/{version_number}/revert",
    response_model=VersionResponse,
    summary="Revert to version",
    responses={
        200: {
            "description": "Analysis reverted successfully - returns new version",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                        "version_number": 5,
                        "config": {"treatment": "x", "outcome": "y"},
                        "config_hash": "abc123...",
                        "changed_by": "550e8400-e29b-41d4-a716-446655440002",
                        "change_summary": "Reverted to version 2",
                        "is_manual_snapshot": False,
                        "parent_version_id": "550e8400-e29b-41d4-a716-446655440003",
                        "created_at": "2026-01-20T17:00:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Missing confirmation or version mismatch",
            "content": {
                "application/json": {
                    "examples": {
                        "no_confirm": {"value": {"detail": "Confirmation required. Set confirmation=true to proceed."}},
                        "mismatch": {"value": {"detail": "Version number in path and body must match"}},
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis or version not found"},
    },
)
async def revert_to_version(
    analysis_id: uuid.UUID,
    version_number: int,
    revert_request: RevertRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_async_session)] = None,
    user_id: Annotated[uuid.UUID, Depends(get_current_user)] = None,
) -> VersionResponse:
    """
    Revert analysis to a specific version.

    Creates a NEW version with the configuration from the target version.
    The original version history is preserved - no data is lost.

    **Request Example:**
    ```json
    {
        "version_number": 2,
        "confirmation": true
    }
    ```

    **Safety Features:**
    - Requires explicit `confirmation: true` to prevent accidents
    - Version number must match in both path and body
    - Original history is preserved (can revert the revert)

    **How it Works:**
    1. Copies configuration from target version
    2. Creates new version with copied config
    3. Links new version to target via `parent_version_id`
    4. Updates analysis to use new configuration

    **Note:** Reverting does not re-run the analysis automatically.
    The new version uses the old configuration, but results remain
    from the previous run. Trigger a new analysis if fresh results needed.
    """
    await verify_analysis_access(analysis_id, user_id, db)

    if not revert_request.confirmation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required. Set confirmation=true to proceed.",
        )

    if revert_request.version_number != version_number:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Version number in path and body must match",
        )

    # Extract request metadata for audit logging
    client_ip = request.client.host if request.client else None
    user_agent_header = request.headers.get("user-agent")
    request_id = request.headers.get("x-request-id")

    try:
        analysis = await analysis_crud.revert_to_version(
            db,
            analysis_id=analysis_id,
            version_number=version_number,
            changed_by=user_id,
            ip_address=client_ip,
            user_agent=user_agent_header,
            request_id=request_id,
        )

        # Get the newly created version
        latest_version = await analysis_version_crud.get_latest_version(db, analysis_id)

        logger.info(
            "Analysis reverted to version",
            extra={
                "analysis_id": str(analysis_id),
                "target_version": version_number,
                "new_version": latest_version.version_number if latest_version else None,
                "user_id": str(user_id),
            },
        )

        return latest_version

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
