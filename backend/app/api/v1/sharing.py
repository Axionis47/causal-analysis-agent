"""API routes for analysis sharing."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import get_current_user, get_current_user_optional
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.db.database import get_async_session
from app.models.analysis import Analysis
from app.models.analysis_share import AnalysisShare, generate_share_token
from app.models.generated_report import ReportType

router = APIRouter(prefix="/api/v1", tags=["sharing"])
logger = get_logger(__name__)


class CreateShareRequest(BaseModel):
    """Request schema for creating a share link."""

    expires_in_days: int | None = Field(
        7,
        ge=1,
        le=365,
        description="Days until the share link expires (null for never)",
    )
    is_public: bool = Field(
        False,
        description="Whether the link can be accessed without authentication",
    )


class ShareResponse(BaseModel):
    """Response schema for share link data."""

    token: str
    share_url: str
    expires_at: datetime | None
    is_public: bool
    view_count: int
    created_at: datetime


class ShareListResponse(BaseModel):
    """Response schema for list of shares."""

    shares: list[ShareResponse]


class SharedAnalysisResponse(BaseModel):
    """Response schema for shared analysis data."""

    analysis_id: str
    kaggle_url: str
    status: str
    shared_by: dict[str, Any]
    share_created_at: datetime
    is_public: bool
    summary: dict[str, Any]
    technical: dict[str, Any]
    visualizations: dict[str, Any]
    validation: dict[str, Any]


@router.post(
    "/analyses/{analysis_id}/share",
    response_model=ShareResponse,
    summary="Create share link",
    responses={
        200: {
            "description": "Share link created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "token": "abc123def456",
                        "share_url": "http://localhost:3000/shared/abc123def456",
                        "expires_at": "2026-01-27T12:00:00Z",
                        "is_public": False,
                        "view_count": 0,
                        "created_at": "2026-01-20T12:00:00Z",
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to share this analysis"},
        404: {"description": "Analysis not found"},
        429: {
            "description": "Maximum share limit reached (10 per analysis)",
            "content": {
                "application/json": {
                    "example": {"detail": "Maximum share limit reached for this analysis"}
                }
            },
        },
    },
)
async def create_share(
    analysis_id: uuid.UUID,
    request: CreateShareRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> ShareResponse:
    """
    Create a shareable link for an analysis.

    Generates a unique token that allows others to view the analysis
    results without needing to be logged in (for public shares) or
    with login required (for private shares).

    **Share Types:**

    - **Public** (`is_public: true`): Anyone with the link can view
    - **Private** (`is_public: false`): Requires authentication to view

    **Expiration:**
    - Set `expires_in_days` to 1-365 for auto-expiring links
    - Set to `null` for links that never expire

    **Limits:**
    - Maximum 10 active share links per analysis
    - Revoke old links if limit is reached

    **Request Example:**
    ```json
    {
        "expires_in_days": 7,
        "is_public": false
    }
    ```

    **View Tracking:**
    Each share link tracks view count, visible in the list shares endpoint.
    """
    analysis = await analysis_crud.get(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )
    if analysis.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to share this analysis",
        )

    # Check share limit (max 10 shares per analysis)
    existing_shares = await db.execute(
        select(AnalysisShare).where(AnalysisShare.analysis_id == analysis_id)
    )
    if len(existing_shares.scalars().all()) >= 10:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum share limit reached for this analysis",
        )

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    # Create share
    share = AnalysisShare(
        analysis_id=analysis_id,
        created_by=user_id,
        share_token=generate_share_token(),
        expires_at=expires_at,
        is_public=request.is_public,
    )
    db.add(share)
    await db.commit()
    await db.refresh(share)

    logger.info(
        "Share link created",
        analysis_id=str(analysis_id),
        share_token=share.share_token[:8],
        is_public=request.is_public,
    )

    base_url = settings.FRONTEND_URL or "http://localhost:3000"
    return ShareResponse(
        token=share.share_token,
        share_url=f"{base_url}/shared/{share.share_token}",
        expires_at=share.expires_at,
        is_public=share.is_public,
        view_count=share.view_count,
        created_at=share.created_at,
    )


@router.get(
    "/analyses/{analysis_id}/shares",
    response_model=ShareListResponse,
    summary="List share links",
    responses={
        200: {
            "description": "List of all share links for the analysis",
            "content": {
                "application/json": {
                    "example": {
                        "shares": [
                            {
                                "token": "abc123def456",
                                "share_url": "http://localhost:3000/shared/abc123def456",
                                "expires_at": "2026-01-27T12:00:00Z",
                                "is_public": False,
                                "view_count": 15,
                                "created_at": "2026-01-20T12:00:00Z",
                            }
                        ]
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to view shares for this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def list_shares(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> ShareListResponse:
    """
    List all active share links for an analysis.

    Returns all share links created for this analysis, including:
    - Token and full share URL
    - Expiration date (if set)
    - Public/private status
    - View count statistics

    **Ordering:** Shares are ordered by creation date (newest first).

    Use this to:
    - Monitor which links are being used (view counts)
    - Identify expired or expiring links
    - Manage share link lifecycle
    """
    analysis = await analysis_crud.get(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )
    if analysis.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view shares for this analysis",
        )

    result = await db.execute(
        select(AnalysisShare)
        .where(AnalysisShare.analysis_id == analysis_id)
        .order_by(AnalysisShare.created_at.desc())
    )
    shares = result.scalars().all()

    base_url = settings.FRONTEND_URL or "http://localhost:3000"
    return ShareListResponse(
        shares=[
            ShareResponse(
                token=share.share_token,
                share_url=f"{base_url}/shared/{share.share_token}",
                expires_at=share.expires_at,
                is_public=share.is_public,
                view_count=share.view_count,
                created_at=share.created_at,
            )
            for share in shares
        ]
    )


@router.delete(
    "/analyses/{analysis_id}/share/{token}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke share link",
)
async def revoke_share(
    analysis_id: uuid.UUID,
    token: str,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> Response:
    """
    Revoke a share link.

    Immediately invalidates the share link. Anyone trying to access
    the link after revocation will receive a 404 error.

    **Warning:** This action is irreversible. The token cannot be
    restored - create a new share link if needed.

    **Use Cases:**
    - Security: Revoke access after sharing period ends
    - Cleanup: Remove unused or old share links
    - Access control: Remove access from specific recipients
    """
    analysis = await analysis_crud.get(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )
    if analysis.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to revoke this share",
        )

    result = await db.execute(
        select(AnalysisShare).where(
            AnalysisShare.analysis_id == analysis_id,
            AnalysisShare.share_token == token,
        )
    )
    share = result.scalar_one_or_none()
    if share is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found",
        )

    await db.delete(share)
    await db.commit()

    logger.info(
        "Share link revoked",
        analysis_id=str(analysis_id),
        share_token=token[:8],
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/shared/{token}",
    response_model=SharedAnalysisResponse,
    summary="Access shared analysis",
    responses={
        200: {
            "description": "Shared analysis data",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                        "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
                        "status": "completed",
                        "shared_by": {
                            "id": "550e8400-e29b-41d4-a716-446655440001",
                            "display_name": "John Doe",
                            "email": None,
                        },
                        "share_created_at": "2026-01-20T12:00:00Z",
                        "is_public": True,
                        "summary": {"key_findings": []},
                        "technical": {"methodology": {}},
                        "visualizations": {"causal_graph": {}},
                        "validation": {"refutation_tests": []},
                    }
                }
            },
        },
        403: {
            "description": "Authentication required for private share link",
            "content": {
                "application/json": {
                    "example": {"detail": "Authentication required to access this private share link"}
                }
            },
        },
        404: {
            "description": "Share link not found or revoked",
            "content": {
                "application/json": {
                    "example": {"detail": "Share link not found or has been revoked"}
                }
            },
        },
        410: {
            "description": "Share link has expired",
            "content": {
                "application/json": {
                    "example": {"detail": "Share link has expired"}
                }
            },
        },
    },
)
async def get_shared_analysis(
    token: str,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID | None = Depends(get_current_user_optional),
) -> SharedAnalysisResponse:
    """
    Access analysis data via a share link.

    Returns the full analysis results for anyone with a valid share token.
    Increments the view count on each access.

    **Access Rules:**

    - **Public links** (`is_public: true`): No authentication required
    - **Private links** (`is_public: false`): Must include Bearer token

    **Response includes:**
    - Analysis metadata (ID, URL, status)
    - Sharer information (limited for privacy)
    - All report sections:
      - Executive summary
      - Technical report
      - Visualizations
      - Validation results

    **Error Cases:**
    - `403 Forbidden`: Private link accessed without authentication
    - `404 Not Found`: Link doesn't exist or was revoked
    - `410 Gone`: Link has expired

    **Privacy:**
    For public shares, sharer's email is not included in response.
    For private shares, email is included since viewer is authenticated.
    """
    result = await db.execute(
        select(AnalysisShare)
        .options(selectinload(AnalysisShare.analysis), selectinload(AnalysisShare.creator))
        .where(AnalysisShare.share_token == token)
    )
    share = result.scalar_one_or_none()

    if share is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or has been revoked",
        )

    # Enforce authentication for private shares
    if not share.is_public and user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required to access this private share link",
        )

    # Check expiration
    if share.expires_at and datetime.now(timezone.utc) > share.expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has expired",
        )

    # Increment view count
    share.view_count += 1
    await db.commit()

    # Load full analysis with relations
    analysis = await analysis_crud.get_with_relations(db, share.analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )

    # Extract report content
    def get_report_content(report_type: ReportType) -> dict[str, Any]:
        for report in analysis.reports:
            if report.report_type == report_type and report.content:
                try:
                    return json.loads(report.content)
                except json.JSONDecodeError:
                    pass
        return {}

    return SharedAnalysisResponse(
        analysis_id=str(analysis.id),
        kaggle_url=analysis.kaggle_url,
        status=analysis.status.value,
        shared_by={
            "id": str(share.creator.id),
            "display_name": getattr(share.creator, "display_name", None),
            "email": share.creator.email if not share.is_public else None,
        },
        share_created_at=share.created_at,
        is_public=share.is_public,
        summary=get_report_content(ReportType.EXECUTIVE),
        technical=get_report_content(ReportType.TECHNICAL),
        visualizations=get_report_content(ReportType.VISUALIZATION),
        validation=get_report_content(ReportType.VALIDATION),
    )
