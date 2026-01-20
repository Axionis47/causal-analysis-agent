"""Dataset model for storing downloaded Kaggle datasets."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.data_understanding import DataUnderstanding


class Dataset(BaseModel):
    """
    Dataset model representing a downloaded Kaggle dataset.

    Stores information about the dataset files, cache status,
    and GCS storage location.
    """

    __tablename__ = "datasets"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Kaggle source information
    kaggle_url: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
    )
    kaggle_dataset_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Kaggle dataset identifier (owner/dataset-name)",
    )

    # File information (JSONB for flexible schema)
    files: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        comment="List of files with name, size, format, row_count, column_count",
    )
    selected_file: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Selected file for analysis from multi-file datasets",
    )
    selection_reasoning: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="LLM reasoning for file selection",
    )

    # Storage information
    gcs_path: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="Google Cloud Storage path",
    )

    # Cache information
    cache_key: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        index=True,
    )
    cache_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    downloaded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Dataset metadata and characteristics (JSONB)
    metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="General dataset metadata",
    )
    characteristics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Dataset characteristics from EDA (row_count, column_count, etc.)",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="datasets",
    )
    data_understanding: Mapped[Optional["DataUnderstanding"]] = relationship(
        "DataUnderstanding",
        back_populates="dataset",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_datasets_kaggle_url", "kaggle_url"),
        Index("ix_datasets_cache_key", "cache_key"),
    )

