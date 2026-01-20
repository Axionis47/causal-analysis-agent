"""Analysis model for tracking causal analysis jobs."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import DateTime, Enum, Index, Numeric, String, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis_stage import AnalysisStage
    from app.models.agent_interaction import AgentInteraction
    from app.models.analysis_version import AnalysisVersion
    from app.models.causal_graph import CausalGraph
    from app.models.dataset import Dataset
    from app.models.generated_report import GeneratedReport
    from app.models.treatment_effect import TreatmentEffect
    from app.models.validation_result import ValidationResult
    from app.models.analysis_share import AnalysisShare
    from app.models.analysis_comment import AnalysisComment
    from app.models.sensitivity_analysis import SensitivityAnalysis


class AnalysisStatus(str, enum.Enum):
    """Status of an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Analysis(BaseModel):
    """
    Analysis model representing a causal analysis job.

    Stores the configuration, status, and metadata for an analysis run.
    Has relationships to all result entities (datasets, graphs, effects, etc.).
    """

    __tablename__ = "analyses"

    # Core fields
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    kaggle_url: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
    )
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Configuration (JSONB for flexible schema)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # Timestamps for execution tracking
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Execution metadata
    estimated_duration: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Estimated duration in seconds",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    is_partial: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether the analysis completed with partial results",
    )

    # Cost tracking fields
    # llm_tokens_used tracks the total number of LLM tokens consumed during analysis
    # estimated_cost is calculated based on model pricing (e.g., GPT-4: $0.03/1K tokens)
    llm_tokens_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total LLM tokens consumed during analysis",
    )
    estimated_cost: Mapped[float] = mapped_column(
        Numeric(10, 4),
        default=0.0,
        nullable=False,
        comment="Estimated cost in USD based on model pricing",
    )

    # Relationships
    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    stages: Mapped[list["AnalysisStage"]] = relationship(
        "AnalysisStage",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    causal_graphs: Mapped[list["CausalGraph"]] = relationship(
        "CausalGraph",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    treatment_effects: Mapped[list["TreatmentEffect"]] = relationship(
        "TreatmentEffect",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    validation_results: Mapped[list["ValidationResult"]] = relationship(
        "ValidationResult",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    agent_interactions: Mapped[list["AgentInteraction"]] = relationship(
        "AgentInteraction",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    reports: Mapped[list["GeneratedReport"]] = relationship(
        "GeneratedReport",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    shares: Mapped[list["AnalysisShare"]] = relationship(
        "AnalysisShare",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    comments: Mapped[list["AnalysisComment"]] = relationship(
        "AnalysisComment",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    sensitivity_analyses: Mapped[list["SensitivityAnalysis"]] = relationship(
        "SensitivityAnalysis",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    versions: Mapped[list["AnalysisVersion"]] = relationship(
        "AnalysisVersion",
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AnalysisVersion.version_number.desc()",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_analyses_user_status", "user_id", "status"),
        Index("ix_analyses_created_at", "created_at"),
    )
