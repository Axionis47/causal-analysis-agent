"""Add graph versioning columns.

Revision ID: 009_add_graph_versioning
Revises: 008_fix_report_format_enum_values
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "009_add_graph_versioning"
down_revision: Union[str, None] = "008_fix_report_format_enum_values"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add parent_graph_id for graph versioning (self-referencing FK)
    op.add_column(
        "causal_graphs",
        sa.Column(
            "parent_graph_id",
            UUID(as_uuid=True),
            sa.ForeignKey("causal_graphs.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # Add is_user_modified flag
    op.add_column(
        "causal_graphs",
        sa.Column(
            "is_user_modified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # Create index for parent_graph_id
    op.create_index(
        "ix_causal_graphs_parent_graph_id",
        "causal_graphs",
        ["parent_graph_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_causal_graphs_parent_graph_id", table_name="causal_graphs")
    op.drop_column("causal_graphs", "is_user_modified")
    op.drop_column("causal_graphs", "parent_graph_id")
