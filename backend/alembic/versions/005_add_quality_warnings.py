"""Add quality_warnings column to data_understandings.

Revision ID: 005_add_quality_warnings
Revises: 004_add_cost_tracking
Create Date: 2026-01-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005_add_quality_warnings"
down_revision: Union[str, None] = "004_add_cost_tracking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "data_understandings",
        sa.Column(
            "quality_warnings",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("data_understandings", "quality_warnings")
