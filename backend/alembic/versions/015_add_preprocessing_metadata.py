"""Add preprocessing metadata columns to data_understandings.

Revision ID: 015_add_preprocessing_metadata
Revises: 014_add_audit_logs
Create Date: 2026-01-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "015_add_preprocessing_metadata"
down_revision: Union[str, None] = "014_add_audit_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add preprocessing_applied boolean column
    op.add_column(
        "data_understandings",
        sa.Column(
            "preprocessing_applied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # Add preprocessing_steps JSONB column (list of PreprocessingStep objects)
    op.add_column(
        "data_understandings",
        sa.Column(
            "preprocessing_steps",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )

    # Add preprocessed_row_count (nullable, may differ from original)
    op.add_column(
        "data_understandings",
        sa.Column(
            "preprocessed_row_count",
            sa.Integer(),
            nullable=True,
        ),
    )

    # Add preprocessed_column_count (nullable, may increase with one-hot encoding)
    op.add_column(
        "data_understandings",
        sa.Column(
            "preprocessed_column_count",
            sa.Integer(),
            nullable=True,
        ),
    )

    # Add preprocessing_config JSONB column (stores configuration used)
    op.add_column(
        "data_understandings",
        sa.Column(
            "preprocessing_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("data_understandings", "preprocessing_config")
    op.drop_column("data_understandings", "preprocessed_column_count")
    op.drop_column("data_understandings", "preprocessed_row_count")
    op.drop_column("data_understandings", "preprocessing_steps")
    op.drop_column("data_understandings", "preprocessing_applied")
