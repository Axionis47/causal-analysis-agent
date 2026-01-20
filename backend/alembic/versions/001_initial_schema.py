"""Initial schema for Causal Analysis application.

Revision ID: 001_initial
Revises: 
Create Date: 2026-01-19

This migration creates all initial tables for the Causal Analysis multi-agent system:
- analyses: Main analysis job tracking
- datasets: Downloaded Kaggle datasets
- data_understandings: EDA results
- analysis_stages: Pipeline stage tracking
- causal_graphs: Discovered causal relationships
- treatment_effects: Estimated treatment effects
- validation_results: Validation test results
- agent_interactions: Agent questions and responses
- generated_reports: Analysis reports
- user_credentials: Encrypted API credentials
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create analyses table
    op.create_table(
        "analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kaggle_url", sa.String(length=512), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", name="analysisstatus"),
            nullable=False,
        ),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("estimated_duration", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("is_partial", sa.Boolean(), nullable=False, default=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyses_user_id", "analyses", ["user_id"])
    op.create_index("ix_analyses_kaggle_url", "analyses", ["kaggle_url"])
    op.create_index("ix_analyses_status", "analyses", ["status"])
    op.create_index("ix_analyses_user_status", "analyses", ["user_id", "status"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])

    # Create datasets table
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kaggle_url", sa.String(length=512), nullable=False),
        sa.Column("kaggle_dataset_id", sa.String(length=256), nullable=True),
        sa.Column("files", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("selected_file", sa.String(length=256), nullable=True),
        sa.Column("selection_reasoning", sa.Text(), nullable=True),
        sa.Column("gcs_path", sa.String(length=512), nullable=True),
        sa.Column("cache_key", sa.String(length=256), nullable=True),
        sa.Column("cache_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("downloaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("characteristics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_datasets_analysis_id", "datasets", ["analysis_id"])
    op.create_index("ix_datasets_kaggle_url", "datasets", ["kaggle_url"])
    op.create_index("ix_datasets_cache_key", "datasets", ["cache_key"])

    # Create data_understandings table
    op.create_table(
        "data_understandings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("summary_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("columns", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("quality_issues", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("recommended_preprocessing", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("treatment_candidates", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("outcome_candidates", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("confounder_candidates", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("dataset_id"),
    )
    op.create_index("ix_data_understandings_dataset_id", "data_understandings", ["dataset_id"])

    # Create analysis_stages table
    op.create_table(
        "analysis_stages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "stage",
            sa.Enum("DOWNLOAD", "EDA", "DISCOVERY", "TREATMENT", "VALIDATION", "REPORTING", name="stagetype"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", name="stagestatus"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("progress_percent", sa.Integer(), nullable=False, default=0),
        sa.Column("current_step", sa.String(length=256), nullable=True),
        sa.Column("agent_name", sa.String(length=128), nullable=True),
        sa.Column("langsmith_run_id", sa.String(length=256), nullable=True),
        sa.Column("outputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("errors", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_analysis_stages_analysis_id", "analysis_stages", ["analysis_id"])
    op.create_index("ix_analysis_stages_analysis_stage", "analysis_stages", ["analysis_id", "stage"])
    op.create_index("ix_analysis_stages_status", "analysis_stages", ["status"])

    # Create causal_graphs table
    op.create_table(
        "causal_graphs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "method",
            sa.Enum("PC", "GES", "FCI", "OTHER", name="discoverymethod"),
            nullable=False,
        ),
        sa.Column("edges", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("nodes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("graph_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("algorithm_params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("execution_time_seconds", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_causal_graphs_analysis_id", "causal_graphs", ["analysis_id"])
    op.create_index("ix_causal_graphs_analysis_method", "causal_graphs", ["analysis_id", "method"])

    # Create treatment_effects table
    op.create_table(
        "treatment_effects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("treatment_variable", sa.String(length=256), nullable=False),
        sa.Column("outcome_variable", sa.String(length=256), nullable=False),
        sa.Column(
            "method",
            sa.Enum(
                "PROPENSITY_MATCHING", "DOUBLY_ROBUST", "INVERSE_PROBABILITY_WEIGHTING",
                "INSTRUMENTAL_VARIABLE", "CAUSAL_FOREST", "OTHER",
                name="treatmentmethod"
            ),
            nullable=False,
        ),
        sa.Column("ate", sa.Float(), nullable=True),
        sa.Column("ate_ci_lower", sa.Float(), nullable=True),
        sa.Column("ate_ci_upper", sa.Float(), nullable=True),
        sa.Column("confidence_interval", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("att", sa.Float(), nullable=True),
        sa.Column("atc", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("confounders_adjusted", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("sample_size", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("assumptions_checked", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_treatment_effects_analysis_id", "treatment_effects", ["analysis_id"])
    op.create_index("ix_treatment_effects_analysis_method", "treatment_effects", ["analysis_id", "method"])

    # Create validation_results table
    op.create_table(
        "validation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "validation_type",
            sa.Enum("REFUTATION", "SENSITIVITY", "ROBUSTNESS", "CONSENSUS", name="validationtype"),
            nullable=False,
        ),
        sa.Column("method", sa.String(length=256), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("recommendations", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_validation_results_analysis_id", "validation_results", ["analysis_id"])
    op.create_index(
        "ix_validation_results_analysis_type", "validation_results", ["analysis_id", "validation_type"]
    )

    # Create agent_interactions table
    op.create_table(
        "agent_interactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column(
            "question_type",
            sa.Enum(
                "TREATMENT_SELECTION", "CONFOUNDER_IDENTIFICATION", "CAUSAL_DIRECTION",
                "MEDIATOR_CONFIRMATION", "DOMAIN_KNOWLEDGE", "OTHER",
                name="questiontype"
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("options", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("user_response", sa.Text(), nullable=True),
        sa.Column("default_response", sa.Text(), nullable=False),
        sa.Column("response_used", sa.Text(), nullable=True),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False, default=30),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_interactions_analysis_id", "agent_interactions", ["analysis_id"])
    op.create_index("ix_agent_interactions_stage_id", "agent_interactions", ["stage_id"])
    op.create_index(
        "ix_agent_interactions_analysis_responded", "agent_interactions", ["analysis_id", "responded_at"]
    )

    # Create generated_reports table
    op.create_table(
        "generated_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "report_type",
            sa.Enum("EXECUTIVE", "TECHNICAL", "VISUALIZATION", "VALIDATION", name="reporttype"),
            nullable=False,
        ),
        sa.Column(
            "format",
            sa.Enum("JSON", "HTML", "PDF", "MARKDOWN", name="reportformat"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("gcs_path", sa.String(length=512), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_generated_reports_analysis_id", "generated_reports", ["analysis_id"])
    op.create_index("ix_generated_reports_analysis_type", "generated_reports", ["analysis_id", "report_type"])
    op.create_index("ix_generated_reports_format", "generated_reports", ["format"])

    # Create user_credentials table
    op.create_table(
        "user_credentials",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "provider",
            sa.Enum("KAGGLE", name="credentialprovider"),
            nullable=False,
        ),
        sa.Column("username", sa.String(length=256), nullable=False),
        sa.Column("encrypted_api_key", sa.LargeBinary(), nullable=False),
        sa.Column("encryption_key_version", sa.String(length=64), nullable=False, default="v1"),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_credentials_user_id", "user_credentials", ["user_id"])
    op.create_index("ix_user_credentials_user_provider", "user_credentials", ["user_id", "provider"])
    op.create_index("ix_user_credentials_provider_username", "user_credentials", ["provider", "username"])


def downgrade() -> None:
    # Drop tables in reverse order of creation (respecting foreign key constraints)
    op.drop_table("user_credentials")
    op.drop_table("generated_reports")
    op.drop_table("agent_interactions")
    op.drop_table("validation_results")
    op.drop_table("treatment_effects")
    op.drop_table("causal_graphs")
    op.drop_table("analysis_stages")
    op.drop_table("data_understandings")
    op.drop_table("datasets")
    op.drop_table("analyses")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS credentialprovider")
    op.execute("DROP TYPE IF EXISTS reportformat")
    op.execute("DROP TYPE IF EXISTS reporttype")
    op.execute("DROP TYPE IF EXISTS questiontype")
    op.execute("DROP TYPE IF EXISTS validationtype")
    op.execute("DROP TYPE IF EXISTS treatmentmethod")
    op.execute("DROP TYPE IF EXISTS discoverymethod")
    op.execute("DROP TYPE IF EXISTS stagestatus")
    op.execute("DROP TYPE IF EXISTS stagetype")
    op.execute("DROP TYPE IF EXISTS analysisstatus")
