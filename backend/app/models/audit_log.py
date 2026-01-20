"""AuditLog model for comprehensive audit trail."""

import enum
import uuid
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.user import User


class AuditAction(str, enum.Enum):
    """Types of auditable actions."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    REVERT = "revert"
    SNAPSHOT = "snapshot"
    VIEW = "view"
    EXPORT = "export"
    SHARE = "share"


class AuditLog(BaseModel):
    """
    AuditLog model for comprehensive audit trail.

    Tracks all significant actions performed on entities for compliance,
    debugging, and security purposes.
    """

    __tablename__ = "audit_logs"

    # Entity being tracked
    entity_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Type of entity (e.g., 'analysis', 'version', 'report')",
    )

    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="ID of the entity being tracked",
    )

    # Action performed
    action: Mapped[AuditAction] = mapped_column(
        Enum(AuditAction),
        nullable=False,
        index=True,
        comment="Type of action performed",
    )

    # User who performed the action
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who performed the action (null for system actions)",
    )

    # Change details
    changes: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Before/after values for the change",
    )

    # Request metadata
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True,
        comment="IP address of the request",
    )

    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="User agent string from the request",
    )

    # Additional context
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description of the action",
    )

    # Request ID for correlation
    request_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Request ID for correlating related audit entries",
    )

    # Relationship to user
    user: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[user_id],
    )

    # Table constraints and indexes
    __table_args__ = (
        # Composite index for entity queries
        Index("ix_audit_logs_entity", "entity_type", "entity_id", "created_at"),
        # Index for user-based queries
        Index("ix_audit_logs_user_action", "user_id", "action", "created_at"),
        # Index for temporal queries
        Index("ix_audit_logs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Generate string representation."""
        return f"<AuditLog({self.entity_type}/{self.entity_id}: {self.action.value})>"
