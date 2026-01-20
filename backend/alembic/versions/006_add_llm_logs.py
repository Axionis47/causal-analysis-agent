"""Add llm_logs table for prompt/response auditing.

Revision ID: 006_add_llm_logs
Revises: 005_add_quality_warnings
Create Date: 2026-01-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006_add_llm_logs"
down_revision: Union[str, None] = "005_add_quality_warnings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Numeric(10, 4), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_llm_logs_analysis_id", "llm_logs", ["analysis_id"])
    op.create_index("ix_llm_logs_created_at", "llm_logs", ["created_at"])
    op.create_index("ix_llm_logs_provider_model", "llm_logs", ["provider", "model"])


def downgrade() -> None:
    op.drop_index("ix_llm_logs_provider_model", table_name="llm_logs")
    op.drop_index("ix_llm_logs_created_at", table_name="llm_logs")
    op.drop_index("ix_llm_logs_analysis_id", table_name="llm_logs")
    op.drop_table("llm_logs")
