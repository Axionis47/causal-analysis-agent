"""AnalysisVersion model for tracking configuration history."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.user import User


class AnalysisVersion(BaseModel):
    """
    AnalysisVersion model for tracking historical versions of analysis configurations.

    Each version captures a snapshot of the analysis config at a point in time,
    enabling audit trails, comparisons, and rollbacks.
    """

    __tablename__ = "analysis_versions"

    # Foreign key to the parent analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Auto-incrementing version number per analysis
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Version number, auto-incremented per analysis",
    )

    # Snapshot of the config at this version
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Snapshot of analysis config at this version",
    )

    # SHA-256 hash of config for quick comparison
    config_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash of config JSON for quick duplicate detection",
    )

    # User who made this change
    changed_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who created this version",
    )

    # Description of what changed
    change_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Optional description of what changed in this version",
    )

    # Whether this was a manual snapshot or automatic
    is_manual_snapshot: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True if user explicitly created this snapshot, False if auto-created on update",
    )

    # Parent version for tracking revert lineage
    parent_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_versions.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent version ID if this was created via revert",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="versions",
        foreign_keys=[analysis_id],
    )

    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="analysis_versions",
        foreign_keys=[changed_by],
    )

    # Self-referential relationship for parent/children versioning
    parent_version: Mapped[Optional["AnalysisVersion"]] = relationship(
        "AnalysisVersion",
        remote_side="AnalysisVersion.id",
        foreign_keys=[parent_version_id],
        back_populates="child_versions",
    )

    child_versions: Mapped[list["AnalysisVersion"]] = relationship(
        "AnalysisVersion",
        foreign_keys="AnalysisVersion.parent_version_id",
        back_populates="parent_version",
    )

    # Table constraints and indexes
    __table_args__ = (
        # Unique constraint on (analysis_id, version_number)
        UniqueConstraint(
            "analysis_id",
            "version_number",
            name="uq_analysis_versions_analysis_version",
        ),
        # Composite index for efficient version queries
        Index("ix_analysis_versions_analysis_version", "analysis_id", "version_number"),
        # Index for temporal queries
        Index("ix_analysis_versions_created_at", "created_at"),
        # Index for user-based queries
        Index("ix_analysis_versions_changed_by", "changed_by"),
    )

    def __repr__(self) -> str:
        """Generate string representation."""
        return f"<AnalysisVersion(analysis_id={self.analysis_id}, version={self.version_number})>"
