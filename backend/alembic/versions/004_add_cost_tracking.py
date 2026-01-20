"""Add cost tracking columns to analyses table.

Revision ID: 004_add_cost_tracking
Revises: 003_add_users_table
Create Date: 2026-01-19
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004_add_cost_tracking"
down_revision: Union[str, None] = "003_add_users_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add cost tracking columns to analyses table
    op.add_column(
        "analyses",
        sa.Column("llm_tokens_used", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "analyses",
        sa.Column(
            "estimated_cost",
            sa.Numeric(precision=10, scale=4),
            nullable=False,
            server_default="0.0",
        ),
    )

    # Add composite index for efficient quota queries
    op.create_index(
        "ix_analyses_user_id_created_at",
        "analyses",
        ["user_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_user_id_created_at", table_name="analyses")
    op.drop_column("analyses", "estimated_cost")
    op.drop_column("analyses", "llm_tokens_used")
