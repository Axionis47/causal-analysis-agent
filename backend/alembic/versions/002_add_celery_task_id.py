"""Add celery_task_id to analyses.

Revision ID: 002_add_celery_task_id
Revises: 001_initial
Create Date: 2026-01-19
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_add_celery_task_id"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analyses",
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_analyses_celery_task_id",
        "analyses",
        ["celery_task_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_celery_task_id", table_name="analyses")
    op.drop_column("analyses", "celery_task_id")
