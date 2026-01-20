"""TreatmentEffect model for storing causal effect estimates."""

import enum
import uuid
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Enum, Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.sensitivity_analysis import SensitivityAnalysis


class TreatmentMethod(str, enum.Enum):
    """Treatment effect estimation method."""

    PROPENSITY_MATCHING = "propensity_matching"
    DOUBLY_ROBUST = "doubly_robust"
    INVERSE_PROBABILITY_WEIGHTING = "ipw"
    INSTRUMENTAL_VARIABLE = "iv"
    CAUSAL_FOREST = "causal_forest"
    OTHER = "other"


class TreatmentEffect(BaseModel):
    """
    TreatmentEffect model representing estimated causal effects.

    Stores the treatment effect estimates, confidence intervals,
    and method-specific details.
    """

    __tablename__ = "treatment_effects"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Treatment and outcome specification
    treatment_variable: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )
    outcome_variable: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )

    # Estimation method
    method: Mapped[TreatmentMethod] = mapped_column(
        Enum(TreatmentMethod),
        nullable=False,
    )

    # Effect estimates
    ate: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average Treatment Effect",
    )
    ate_ci_lower: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Lower bound of 95% confidence interval",
    )
    ate_ci_upper: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Upper bound of 95% confidence interval",
    )
    confidence_interval: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Full confidence interval details",
    )

    # Additional effect estimates
    att: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average Treatment Effect on the Treated",
    )
    atc: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average Treatment Effect on the Control",
    )
    p_value: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # Confounders and sample information
    confounders_adjusted: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
        comment="List of confounders adjusted for",
    )
    sample_size: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Sample sizes: treatment, control, matched",
    )

    # Assumption checks
    assumptions_checked: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Results of assumption checks (positivity, ignorability, SUTVA)",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="treatment_effects",
    )
    sensitivity_analyses: Mapped[list["SensitivityAnalysis"]] = relationship(
        "SensitivityAnalysis",
        back_populates="treatment_effect",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_treatment_effects_analysis_method", "analysis_id", "method"),
    )

