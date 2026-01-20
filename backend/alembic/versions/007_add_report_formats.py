"""Ensure reportformat enum includes pdf and markdown.

Revision ID: 007_add_report_formats
Revises: 006_add_llm_logs
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007_add_report_formats"
down_revision: Union[str, None] = "006_add_llm_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'pdf'")
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'markdown'")


def downgrade() -> None:
    pass
