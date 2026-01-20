"""AnalysisStage model for tracking analysis pipeline stages."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class StageType(str, enum.Enum):
    """Type of analysis stage."""

    DOWNLOAD = "download"
    EDA = "eda"
    DISCOVERY = "discovery"
    TREATMENT = "treatment"
    VALIDATION = "validation"
    REPORTING = "reporting"


class StageStatus(str, enum.Enum):
    """Status of an analysis stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AnalysisStage(BaseModel):
    """
    AnalysisStage model for tracking individual pipeline stages.

    Stores progress and status information for each stage of the analysis.
    """

    __tablename__ = "analysis_stages"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Stage information
    stage: Mapped[StageType] = mapped_column(
        Enum(StageType),
        nullable=False,
    )
    status: Mapped[StageStatus] = mapped_column(
        Enum(StageStatus),
        default=StageStatus.PENDING,
        nullable=False,
    )

    # Timing information
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Progress tracking
    progress_percent: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    current_step: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Current step description for progress display",
    )

    # Agent and tracing information
    agent_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )
    langsmith_run_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="LangSmith trace run ID for debugging",
    )

    # Stage outputs and errors (JSONB for flexibility)
    outputs: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Stage outputs and results",
    )
    errors: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Error details if stage failed",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="stages",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_analysis_stages_analysis_stage", "analysis_id", "stage"),
        Index("ix_analysis_stages_status", "status"),
    )

