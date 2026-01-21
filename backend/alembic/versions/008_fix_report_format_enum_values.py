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
    # Add new enum values if they don't exist
    # Note: PostgreSQL requires a commit between adding enum values and using them
    # For fresh databases, these values may already exist from the initial schema
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'pdf'")
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'markdown'")


def downgrade() -> None:
    pass
