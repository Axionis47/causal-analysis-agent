"""SensitivityAnalysis model for storing sensitivity analysis results."""

import uuid
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.treatment_effect import TreatmentEffect


class SensitivityAnalysis(BaseModel):
    """
    SensitivityAnalysis model for storing detailed sensitivity analysis results.

    Stores robustness values, E-values, and other sensitivity metrics for
    causal effect estimates. Provides richer analysis than ValidationResult
    for sensitivity-specific data.
    """

    __tablename__ = "sensitivity_analyses"

    # Foreign keys
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    treatment_effect_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("treatment_effects.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Sensitivity analysis method
    method: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Sensitivity analysis method (e.g., linear-partial-R2, evalue)",
    )

    # Linear partial R-squared sensitivity results
    robustness_value: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Robustness value (RV) - strength of confounding needed to nullify effect",
    )
    rv_50_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confounder strength to reduce effect by 50%",
    )
    rv_100_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confounder strength to completely nullify effect",
    )

    # E-value sensitivity results
    evalue: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="E-value - minimum confounding strength to explain away effect",
    )
    evalue_ci: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="E-value for confidence interval bound",
    )

    # Benchmark covariate analysis
    benchmark_covariate: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Reference covariate used for benchmarking",
    )
    confounding_strength: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Effect fractions on treatment and outcome",
    )

    # Original effect information
    original_effect: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Original treatment effect estimate",
    )

    # Assessment results
    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether sensitivity analysis passed threshold",
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score (0-1) based on sensitivity results",
    )

    # LLM interpretation
    interpretation: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="LLM-generated plain-language interpretation",
    )
    robustness_assessment: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Assessment category: robust, moderate, or sensitive",
    )

    # Additional details
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Additional method-specific details",
    )
    recommendations: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        comment="Recommendations based on sensitivity results",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="sensitivity_analyses",
    )
    treatment_effect: Mapped[Optional["TreatmentEffect"]] = relationship(
        "TreatmentEffect",
        back_populates="sensitivity_analyses",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_sensitivity_analyses_analysis_method", "analysis_id", "method"),
        Index("ix_sensitivity_analyses_passed", "analysis_id", "passed"),
    )
