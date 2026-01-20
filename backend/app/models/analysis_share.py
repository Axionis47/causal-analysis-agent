"""AnalysisShare model for storing shareable links to analyses."""

import secrets
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.user import User


def generate_share_token() -> str:
    """Generate a cryptographically secure share token."""
    return secrets.token_urlsafe(24)


class AnalysisShare(BaseModel):
    """
    AnalysisShare model for managing shareable links to analyses.

    Supports:
    - Expiration dates for time-limited sharing
    - Public/private access control
    - View count tracking
    """

    __tablename__ = "analysis_shares"

    # Foreign keys
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Share token (unique, URL-safe)
    share_token: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        default=generate_share_token,
        index=True,
    )

    # Access control
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Null means never expires",
    )
    is_public: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="If true, no authentication required to view",
    )

    # Analytics
    view_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="shares",
    )
    creator: Mapped["User"] = relationship(
        "User",
        back_populates="analysis_shares",
    )

    # Indexes
    __table_args__ = (
        Index("ix_analysis_shares_analysis_created", "analysis_id", "created_by"),
    )

    @property
    def is_expired(self) -> bool:
        """Check if the share link has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(self.expires_at.tzinfo) > self.expires_at

    @property
    def share_url(self) -> str:
        """Generate the full share URL (base URL should be configured)."""
        return f"/shared/{self.share_token}"
