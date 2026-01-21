"""API routes for analysis comments."""

from __future__ import annotations

import uuid
from typing import Any

import bleach
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import get_current_user
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.db.database import get_async_session
from app.models.analysis_comment import AnalysisComment

router = APIRouter(prefix="/api/v1", tags=["comments"])
logger = get_logger(__name__)

# Allowed HTML tags for sanitization (Markdown rendering)
ALLOWED_TAGS = [
    "p", "br", "strong", "em", "code", "pre", "blockquote",
    "ul", "ol", "li", "a", "h1", "h2", "h3", "h4", "h5", "h6",
]
ALLOWED_ATTRIBUTES = {"a": ["href", "title"]}


class CreateCommentRequest(BaseModel):
    """Request schema for creating a comment."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Comment content (Markdown supported)",
    )
    parent_comment_id: uuid.UUID | None = Field(
        None,
        description="Parent comment ID for replies",
    )

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, value: str) -> str:
        """Sanitize content to prevent XSS."""
        return bleach.clean(
            value,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            strip=True,
        )


class UpdateCommentRequest(BaseModel):
    """Request schema for updating a comment."""

    content: str | None = Field(
        None,
        min_length=1,
        max_length=5000,
    )
    is_resolved: bool | None = None

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, value: str | None) -> str | None:
        """Sanitize content to prevent XSS."""
        if value is None:
            return None
        return bleach.clean(
            value,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            strip=True,
        )


class UserInfo(BaseModel):
    """User information for comment responses."""

    id: str
    email: str | None
    display_name: str | None


class CommentResponse(BaseModel):
    """Response schema for comment data."""

    id: str
    analysis_id: str
    user_id: str
    user: UserInfo | None
    parent_comment_id: str | None
    content: str
    is_resolved: bool
    created_at: str
    updated_at: str


class CommentListResponse(BaseModel):
    """Response schema for list of comments."""

    comments: list[CommentResponse]


@router.post(
    "/analyses/{analysis_id}/comments",
    response_model=CommentResponse,
    summary="Create comment",
    responses={
        200: {
            "description": "Comment created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                        "user_id": "550e8400-e29b-41d4-a716-446655440002",
                        "user": {
                            "id": "550e8400-e29b-41d4-a716-446655440002",
                            "email": "user@example.com",
                            "display_name": "John Doe",
                        },
                        "parent_comment_id": None,
                        "content": "This finding seems significant for our use case.",
                        "is_resolved": False,
                        "created_at": "2026-01-20T12:00:00Z",
                        "updated_at": "2026-01-20T12:00:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Cannot reply to a reply (max 1 level nesting)",
            "content": {
                "application/json": {
                    "example": {"detail": "Cannot reply to a reply (max 1 level nesting)"}
                }
            },
        },
        401: {"description": "Authentication required"},
        404: {
            "description": "Analysis or parent comment not found",
            "content": {
                "application/json": {
                    "examples": {
                        "analysis": {"value": {"detail": "Analysis not found"}},
                        "parent": {"value": {"detail": "Parent comment not found"}},
                    }
                }
            },
        },
        422: {"description": "Validation error - content too long or empty"},
    },
)
async def create_comment(
    analysis_id: uuid.UUID,
    request: CreateCommentRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> CommentResponse:
    """
    Create a new comment on an analysis.

    Comments support Markdown formatting and can be replies to other comments.
    Use comments for team collaboration, noting findings, or asking questions.

    **Request Example (top-level comment):**
    ```json
    {
        "content": "This finding aligns with our hypothesis about **education effects**."
    }
    ```

    **Request Example (reply):**
    ```json
    {
        "content": "Agreed, the effect size is larger than expected.",
        "parent_comment_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```

    **Content Features:**
    - Markdown supported (bold, italic, code, links, lists)
    - Maximum 5000 characters
    - HTML sanitized to prevent XSS

    **Threading:**
    - Maximum 1 level of nesting (comments and replies)
    - Cannot reply to a reply

    **Mentions:**
    Use @email to mention team members (notification feature coming soon).
    """
    # Verify analysis exists and user has access
    analysis = await analysis_crud.get(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )

    # If this is a reply, verify parent comment exists
    if request.parent_comment_id:
        parent_result = await db.execute(
            select(AnalysisComment).where(
                AnalysisComment.id == request.parent_comment_id,
                AnalysisComment.analysis_id == analysis_id,
            )
        )
        parent = parent_result.scalar_one_or_none()
        if parent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parent comment not found",
            )
        # Prevent nested replies (max 1 level deep)
        if parent.parent_comment_id is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot reply to a reply (max 1 level nesting)",
            )

    # Create comment
    comment = AnalysisComment(
        analysis_id=analysis_id,
        user_id=user_id,
        parent_comment_id=request.parent_comment_id,
        content=request.content,
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)

    # Load user info
    result = await db.execute(
        select(AnalysisComment)
        .options(selectinload(AnalysisComment.user))
        .where(AnalysisComment.id == comment.id)
    )
    comment = result.scalar_one()

    logger.info(
        "Comment created",
        analysis_id=str(analysis_id),
        comment_id=str(comment.id),
        is_reply=request.parent_comment_id is not None,
    )

    return _comment_to_response(comment)


@router.get(
    "/analyses/{analysis_id}/comments",
    response_model=CommentListResponse,
    summary="List comments",
    responses={
        200: {
            "description": "List of all comments on the analysis",
            "content": {
                "application/json": {
                    "example": {
                        "comments": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                                "user_id": "550e8400-e29b-41d4-a716-446655440002",
                                "user": {
                                    "id": "550e8400-e29b-41d4-a716-446655440002",
                                    "email": "user@example.com",
                                    "display_name": "John Doe",
                                },
                                "parent_comment_id": None,
                                "content": "Great analysis!",
                                "is_resolved": False,
                                "created_at": "2026-01-20T12:00:00Z",
                                "updated_at": "2026-01-20T12:00:00Z",
                            }
                        ]
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        404: {"description": "Analysis not found"},
    },
)
async def list_comments(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> CommentListResponse:
    """
    List all comments on an analysis.

    Returns all comments including replies, ordered by creation date
    (newest first). Use `parent_comment_id` to build thread structure.

    **Response includes:**
    - Comment content and metadata
    - User information for each comment
    - Resolution status
    - Parent comment ID for threading

    **Building Threads:**
    1. Filter comments where `parent_comment_id` is null (top-level)
    2. For each top-level, find replies by matching `parent_comment_id`
    3. Sort threads by oldest top-level first, replies by oldest first
    """
    # Verify analysis exists
    analysis = await analysis_crud.get(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )

    result = await db.execute(
        select(AnalysisComment)
        .options(selectinload(AnalysisComment.user))
        .where(AnalysisComment.analysis_id == analysis_id)
        .order_by(AnalysisComment.created_at.desc())
    )
    comments = result.scalars().all()

    return CommentListResponse(
        comments=[_comment_to_response(c) for c in comments]
    )


@router.patch(
    "/comments/{comment_id}",
    response_model=CommentResponse,
    summary="Update comment",
    responses={
        200: {
            "description": "Comment updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "content": "Updated comment content",
                        "is_resolved": True,
                        "updated_at": "2026-01-20T13:00:00Z",
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to edit this comment's content"},
        404: {"description": "Comment not found"},
    },
)
async def update_comment(
    comment_id: uuid.UUID,
    request: UpdateCommentRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> CommentResponse:
    """
    Update a comment's content or resolution status.

    **Update Content (owner only):**
    ```json
    {
        "content": "Updated comment text with **markdown**"
    }
    ```

    **Mark as Resolved (anyone):**
    ```json
    {
        "is_resolved": true
    }
    ```

    **Both:**
    ```json
    {
        "content": "Final answer after discussion",
        "is_resolved": true
    }
    ```

    **Permissions:**
    - Only the comment owner can edit content
    - Any authenticated user can mark as resolved/unresolved

    **Resolution Status:**
    Use `is_resolved` to mark discussions as complete. Useful for
    tracking action items and closing discussion threads.
    """
    result = await db.execute(
        select(AnalysisComment)
        .options(selectinload(AnalysisComment.user))
        .where(AnalysisComment.id == comment_id)
    )
    comment = result.scalar_one_or_none()

    if comment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Only owner can edit content, but anyone can resolve
    if request.content is not None and comment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to edit this comment",
        )

    # Update fields
    if request.content is not None:
        comment.content = request.content
    if request.is_resolved is not None:
        comment.is_resolved = request.is_resolved

    await db.commit()
    await db.refresh(comment)

    logger.info(
        "Comment updated",
        comment_id=str(comment_id),
        updated_fields=[k for k, v in request.model_dump().items() if v is not None],
    )

    return _comment_to_response(comment)


@router.delete(
    "/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete comment",
)
async def delete_comment(
    comment_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> Response:
    """
    Delete a comment.

    **Permissions:** Only the comment owner can delete their comments.

    **Warning:** This action is irreversible. The comment and any
    context it provided will be permanently removed.

    **Note:** Deleting a parent comment does NOT delete its replies.
    Replies will remain visible but the `parent_comment_id` will
    reference a non-existent comment.
    """
    result = await db.execute(
        select(AnalysisComment).where(AnalysisComment.id == comment_id)
    )
    comment = result.scalar_one_or_none()

    if comment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    if comment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this comment",
        )

    await db.delete(comment)
    await db.commit()

    logger.info(
        "Comment deleted",
        comment_id=str(comment_id),
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _comment_to_response(comment: AnalysisComment) -> CommentResponse:
    """Convert comment model to response."""
    user_info = None
    if comment.user:
        user_info = UserInfo(
            id=str(comment.user.id),
            email=comment.user.email,
            display_name=getattr(comment.user, "display_name", None),
        )

    return CommentResponse(
        id=str(comment.id),
        analysis_id=str(comment.analysis_id),
        user_id=str(comment.user_id),
        user=user_info,
        parent_comment_id=str(comment.parent_comment_id) if comment.parent_comment_id else None,
        content=comment.content,
        is_resolved=comment.is_resolved,
        created_at=comment.created_at.isoformat(),
        updated_at=comment.updated_at.isoformat(),
    )
