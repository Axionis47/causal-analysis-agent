"""Add analysis_versions table for configuration versioning.

Revision ID: 013_add_analysis_versioning
Revises: 012_add_sensitivity_analysis
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "013_add_analysis_versioning"
down_revision: Union[str, None] = "012_add_sensitivity_analysis"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create analysis_versions table
    op.create_table(
        "analysis_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "version_number",
            sa.Integer(),
            nullable=False,
            comment="Version number, auto-incremented per analysis",
        ),
        sa.Column(
            "config",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Snapshot of analysis config at this version",
        ),
        sa.Column(
            "config_hash",
            sa.String(64),
            nullable=False,
            comment="SHA-256 hash of config JSON for quick duplicate detection",
        ),
        sa.Column(
            "changed_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            comment="User who created this version",
        ),
        sa.Column(
            "change_summary",
            sa.Text(),
            nullable=True,
            comment="Optional description of what changed in this version",
        ),
        sa.Column(
            "is_manual_snapshot",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            comment="True if user explicitly created this snapshot",
        ),
        sa.Column(
            "parent_version_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analysis_versions.id", ondelete="SET NULL"),
            nullable=True,
            comment="Parent version ID if this was created via revert",
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
        "ix_analysis_versions_analysis_id",
        "analysis_versions",
        ["analysis_id"],
    )
    op.create_index(
        "ix_analysis_versions_analysis_version",
        "analysis_versions",
        ["analysis_id", "version_number"],
        unique=True,
    )
    op.create_index(
        "ix_analysis_versions_changed_by",
        "analysis_versions",
        ["changed_by"],
    )
    op.create_index(
        "ix_analysis_versions_created_at",
        "analysis_versions",
        ["created_at"],
    )

    # Create unique constraint
    op.create_unique_constraint(
        "uq_analysis_versions_analysis_version",
        "analysis_versions",
        ["analysis_id", "version_number"],
    )


def downgrade() -> None:
    # Drop unique constraint
    op.drop_constraint(
        "uq_analysis_versions_analysis_version",
        "analysis_versions",
        type_="unique",
    )

    # Drop indexes
    op.drop_index("ix_analysis_versions_created_at", table_name="analysis_versions")
    op.drop_index("ix_analysis_versions_changed_by", table_name="analysis_versions")
    op.drop_index("ix_analysis_versions_analysis_version", table_name="analysis_versions")
    op.drop_index("ix_analysis_versions_analysis_id", table_name="analysis_versions")

    # Drop table
    op.drop_table("analysis_versions")
