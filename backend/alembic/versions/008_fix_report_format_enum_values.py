"""Align reportformat enum values with application casing.

Revision ID: 008_fix_report_format_enum_values
Revises: 007_add_report_formats
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008_fix_report_format_enum_values"
down_revision: Union[str, None] = "007_add_report_formats"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'pdf'")
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'markdown'")
    op.execute(
        "UPDATE generated_reports SET format = 'pdf' WHERE format::text = 'PDF'"
    )
    op.execute(
        "UPDATE generated_reports SET format = 'markdown' WHERE format::text = 'MARKDOWN'"
    )


def downgrade() -> None:
    pass
