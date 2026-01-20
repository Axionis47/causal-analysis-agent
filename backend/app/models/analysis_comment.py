"""AnalysisComment model for storing discussion threads on analyses."""

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.user import User


class AnalysisComment(BaseModel):
    """
    AnalysisComment model for discussion threads on analyses.

    Supports:
    - Nested replies (single level deep via parent_comment_id)
    - Resolution tracking for Q&A style discussions
    - User mentions (stored as JSON array of user IDs)
    """

    __tablename__ = "analysis_comments"

    # Foreign keys
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_comment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_comments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="For replies - references parent comment",
    )

    # Content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # State
    is_resolved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Marks Q&A style comments as resolved",
    )

    # Mentions (stored as JSONB array of user UUIDs)
    mentioned_users: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        comment="Array of user IDs mentioned in this comment",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="comments",
    )
    user: Mapped["User"] = relationship(
        "User",
        back_populates="analysis_comments",
        foreign_keys=[user_id],
    )
    parent_comment: Mapped[Optional["AnalysisComment"]] = relationship(
        "AnalysisComment",
        remote_side="AnalysisComment.id",
        back_populates="replies",
        foreign_keys=[parent_comment_id],
    )
    replies: Mapped[list["AnalysisComment"]] = relationship(
        "AnalysisComment",
        back_populates="parent_comment",
        foreign_keys=[parent_comment_id],
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_analysis_comments_analysis_created", "analysis_id", "created_at"),
    )

    @property
    def is_reply(self) -> bool:
        """Check if this comment is a reply to another comment."""
        return self.parent_comment_id is not None
