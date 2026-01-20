"""Add sensitivity_analyses table.

Revision ID: 012_add_sensitivity_analysis
Revises: 011_add_collaboration
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "012_add_sensitivity_analysis"
down_revision: Union[str, None] = "011_add_collaboration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create sensitivity_analyses table
    op.create_table(
        "sensitivity_analyses",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "treatment_effect_id",
            UUID(as_uuid=True),
            sa.ForeignKey("treatment_effects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "method",
            sa.String(128),
            nullable=False,
            comment="Sensitivity analysis method (e.g., linear-partial-R2, evalue)",
        ),
        sa.Column(
            "robustness_value",
            sa.Float(),
            nullable=True,
            comment="Robustness value (RV) - strength of confounding needed to nullify effect",
        ),
        sa.Column(
            "rv_50_percent",
            sa.Float(),
            nullable=True,
            comment="Confounder strength to reduce effect by 50%",
        ),
        sa.Column(
            "rv_100_percent",
            sa.Float(),
            nullable=True,
            comment="Confounder strength to completely nullify effect",
        ),
        sa.Column(
            "evalue",
            sa.Float(),
            nullable=True,
            comment="E-value - minimum confounding strength to explain away effect",
        ),
        sa.Column(
            "evalue_ci",
            sa.Float(),
            nullable=True,
            comment="E-value for confidence interval bound",
        ),
        sa.Column(
            "benchmark_covariate",
            sa.String(256),
            nullable=True,
            comment="Reference covariate used for benchmarking",
        ),
        sa.Column(
            "confounding_strength",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Effect fractions on treatment and outcome",
        ),
        sa.Column(
            "original_effect",
            sa.Float(),
            nullable=True,
            comment="Original treatment effect estimate",
        ),
        sa.Column(
            "passed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            comment="Whether sensitivity analysis passed threshold",
        ),
        sa.Column(
            "confidence_score",
            sa.Float(),
            nullable=True,
            comment="Confidence score (0-1) based on sensitivity results",
        ),
        sa.Column(
            "interpretation",
            sa.Text(),
            nullable=True,
            comment="LLM-generated plain-language interpretation",
        ),
        sa.Column(
            "robustness_assessment",
            sa.String(64),
            nullable=True,
            comment="Assessment category: robust, moderate, or sensitive",
        ),
        sa.Column(
            "details",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="Additional method-specific details",
        ),
        sa.Column(
            "recommendations",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
            comment="Recommendations based on sensitivity results",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )

    # Create indexes
    op.create_index(
        "ix_sensitivity_analyses_analysis_id",
        "sensitivity_analyses",
        ["analysis_id"],
    )
    op.create_index(
        "ix_sensitivity_analyses_treatment_effect_id",
        "sensitivity_analyses",
        ["treatment_effect_id"],
    )
    op.create_index(
        "ix_sensitivity_analyses_analysis_method",
        "sensitivity_analyses",
        ["analysis_id", "method"],
    )
    op.create_index(
        "ix_sensitivity_analyses_passed",
        "sensitivity_analyses",
        ["analysis_id", "passed"],
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_sensitivity_analyses_passed", table_name="sensitivity_analyses")
    op.drop_index("ix_sensitivity_analyses_analysis_method", table_name="sensitivity_analyses")
    op.drop_index("ix_sensitivity_analyses_treatment_effect_id", table_name="sensitivity_analyses")
    op.drop_index("ix_sensitivity_analyses_analysis_id", table_name="sensitivity_analyses")

    # Drop table
    op.drop_table("sensitivity_analyses")
