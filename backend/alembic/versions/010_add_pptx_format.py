"""Add PPTX to report format enum.

Revision ID: 010_add_pptx_format
Revises: 009_add_graph_versioning
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010_add_pptx_format"
down_revision: Union[str, None] = "009_add_graph_versioning"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add PPTX to ReportFormat enum
    op.execute("ALTER TYPE reportformat ADD VALUE IF NOT EXISTS 'pptx'")


def downgrade() -> None:
    # Note: PostgreSQL doesn't support removing enum values directly
    # This is a no-op for safety
    pass
