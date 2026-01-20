"""Add audit_logs table for comprehensive audit trail.

Revision ID: 014_add_audit_logs
Revises: 013_add_analysis_versioning
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "014_add_audit_logs"
down_revision: Union[str, None] = "013_add_analysis_versioning"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "entity_type",
            sa.String(64),
            nullable=False,
            comment="Type of entity (e.g., 'analysis', 'version', 'report')",
        ),
        sa.Column(
            "entity_id",
            UUID(as_uuid=True),
            nullable=False,
            comment="ID of the entity being tracked",
        ),
        sa.Column(
            "action",
            sa.String(32),
            nullable=False,
            comment="Type of action performed",
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            comment="User who performed the action (null for system actions)",
        ),
        sa.Column(
            "changes",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Before/after values for the change",
        ),
        sa.Column(
            "ip_address",
            sa.String(45),
            nullable=True,
            comment="IP address of the request",
        ),
        sa.Column(
            "user_agent",
            sa.String(500),
            nullable=True,
            comment="User agent string from the request",
        ),
        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
            comment="Human-readable description of the action",
        ),
        sa.Column(
            "request_id",
            sa.String(64),
            nullable=True,
            comment="Request ID for correlating related audit entries",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )

    # Create indexes
    op.create_index(
        "ix_audit_logs_entity_type",
        "audit_logs",
        ["entity_type"],
    )
    op.create_index(
        "ix_audit_logs_entity_id",
        "audit_logs",
        ["entity_id"],
    )
    op.create_index(
        "ix_audit_logs_action",
        "audit_logs",
        ["action"],
    )
    op.create_index(
        "ix_audit_logs_user_id",
        "audit_logs",
        ["user_id"],
    )
    op.create_index(
        "ix_audit_logs_request_id",
        "audit_logs",
        ["request_id"],
    )
    op.create_index(
        "ix_audit_logs_created_at",
        "audit_logs",
        ["created_at"],
    )
    # Composite index for entity queries
    op.create_index(
        "ix_audit_logs_entity",
        "audit_logs",
        ["entity_type", "entity_id", "created_at"],
    )
    # Composite index for user-based queries
    op.create_index(
        "ix_audit_logs_user_action",
        "audit_logs",
        ["user_id", "action", "created_at"],
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_audit_logs_user_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_request_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_user_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity_type", table_name="audit_logs")

    # Drop table
    op.drop_table("audit_logs")
