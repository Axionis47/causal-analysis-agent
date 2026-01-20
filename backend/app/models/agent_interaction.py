"""AgentInteraction model for storing agent questions and user responses."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class QuestionType(str, enum.Enum):
    """Type of question asked by agents."""

    TREATMENT_SELECTION = "treatment_selection"
    CONFOUNDER_IDENTIFICATION = "confounder_identification"
    CAUSAL_DIRECTION = "causal_direction"
    MEDIATOR_CONFIRMATION = "mediator_confirmation"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    OTHER = "other"


class AgentInteraction(BaseModel):
    """
    AgentInteraction model for storing agent questions and user responses.

    Agents ask questions when confidence is low (<50%) and can
    proceed with defaults if user doesn't respond within timeout.
    """

    __tablename__ = "agent_interactions"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Optional link to specific stage
    stage_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # Question details
    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    question_type: Mapped[QuestionType] = mapped_column(
        Enum(QuestionType),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Agent's confidence in default answer (0-1)",
    )
    options: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
        comment="Available options for user to choose from",
    )

    # Response handling
    user_response: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    default_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Default response used if user doesn't respond",
    )
    response_used: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Actual response used (user or default)",
    )
    responded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    timeout_seconds: Mapped[int] = mapped_column(
        Integer,
        default=30,
        nullable=False,
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="agent_interactions",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_agent_interactions_analysis_responded", "analysis_id", "responded_at"),
    )

