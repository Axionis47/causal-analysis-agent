"""ValidationResult model for storing refutation and validation tests."""

import enum
import uuid
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class ValidationType(str, enum.Enum):
    """Type of validation test."""

    REFUTATION = "refutation"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    CONSENSUS = "consensus"


class ValidationResult(BaseModel):
    """
    ValidationResult model for storing validation and refutation test results.

    Stores the results of various tests used to validate causal claims.
    """

    __tablename__ = "validation_results"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Validation type and method
    validation_type: Mapped[ValidationType] = mapped_column(
        Enum(ValidationType),
        nullable=False,
    )
    method: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Specific validation method (e.g., placebo_treatment, random_cause)",
    )

    # Test results
    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score (0-1)",
    )

    # Detailed results (JSONB for flexibility)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Detailed test results and statistics",
    )

    # Recommendations
    recommendations: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
        comment="Recommendations based on validation results",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="validation_results",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_validation_results_analysis_type", "analysis_id", "validation_type"),
    )

