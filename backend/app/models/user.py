"""User model for authentication."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis_comment import AnalysisComment
    from app.models.analysis_share import AnalysisShare
    from app.models.analysis_version import AnalysisVersion


class User(BaseModel):
    """
    User model for authentication and authorization.

    Stores user credentials and profile information.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # Relationships for collaboration features
    analysis_shares: Mapped[list["AnalysisShare"]] = relationship(
        "AnalysisShare",
        back_populates="creator",
        cascade="all, delete-orphan",
    )
    analysis_comments: Mapped[list["AnalysisComment"]] = relationship(
        "AnalysisComment",
        back_populates="user",
        foreign_keys="AnalysisComment.user_id",
        cascade="all, delete-orphan",
    )
    analysis_versions: Mapped[list["AnalysisVersion"]] = relationship(
        "AnalysisVersion",
        back_populates="user",
        foreign_keys="AnalysisVersion.changed_by",
    )

    # Indexes
    __table_args__ = (
        Index("ix_users_email_unique", "email", unique=True),
    )
