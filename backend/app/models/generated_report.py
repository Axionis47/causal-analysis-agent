"""GeneratedReport model for storing analysis reports."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class ReportType(str, enum.Enum):
    """Type of generated report."""

    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"


class ReportFormat(str, enum.Enum):
    """Format of generated report."""

    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    PPTX = "pptx"


class GeneratedReport(BaseModel):
    """
    GeneratedReport model for storing generated analysis reports.

    Supports multiple formats (JSON, HTML, PDF, Markdown) and types
    (executive, technical, visualization, validation).
    """

    __tablename__ = "generated_reports"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Report type and format
    report_type: Mapped[ReportType] = mapped_column(
        Enum(ReportType),
        nullable=False,
    )
    format: Mapped[ReportFormat] = mapped_column(
        Enum(ReportFormat),
        nullable=False,
    )

    # Content (for smaller reports like JSON)
    content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Report content for smaller reports",
    )

    # Storage information
    gcs_path: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="GCS path for larger reports (PDF, HTML)",
    )

    # Metadata
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="reports",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_generated_reports_analysis_type", "analysis_id", "report_type"),
        Index("ix_generated_reports_format", "format"),
    )

