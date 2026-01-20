"""Add collaboration tables for sharing and comments.

Revision ID: 011_add_collaboration
Revises: 010_add_pptx_format
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "011_add_collaboration"
down_revision: Union[str, None] = "010_add_pptx_format"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create analysis_shares table
    op.create_table(
        "analysis_shares",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "share_token",
            sa.String(64),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "is_public",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "view_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
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

    # Create indexes for analysis_shares
    op.create_index(
        "ix_analysis_shares_share_token",
        "analysis_shares",
        ["share_token"],
        unique=True,
    )
    op.create_index(
        "ix_analysis_shares_analysis_id",
        "analysis_shares",
        ["analysis_id"],
    )
    op.create_index(
        "ix_analysis_shares_created_by",
        "analysis_shares",
        ["created_by"],
    )

    # Create analysis_comments table
    op.create_table(
        "analysis_comments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_comment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analysis_comments.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "is_resolved",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "mentioned_users",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
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

    # Create indexes for analysis_comments
    op.create_index(
        "ix_analysis_comments_analysis_id",
        "analysis_comments",
        ["analysis_id"],
    )
    op.create_index(
        "ix_analysis_comments_user_id",
        "analysis_comments",
        ["user_id"],
    )
    op.create_index(
        "ix_analysis_comments_parent_comment_id",
        "analysis_comments",
        ["parent_comment_id"],
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_analysis_comments_parent_comment_id", table_name="analysis_comments")
    op.drop_index("ix_analysis_comments_user_id", table_name="analysis_comments")
    op.drop_index("ix_analysis_comments_analysis_id", table_name="analysis_comments")
    op.drop_index("ix_analysis_shares_created_by", table_name="analysis_shares")
    op.drop_index("ix_analysis_shares_analysis_id", table_name="analysis_shares")
    op.drop_index("ix_analysis_shares_share_token", table_name="analysis_shares")

    # Drop tables
    op.drop_table("analysis_comments")
    op.drop_table("analysis_shares")
