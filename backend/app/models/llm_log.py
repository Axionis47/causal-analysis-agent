"""LLMLog model for storing prompt/response records."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class LLMLog(BaseModel):
    """Persisted LLM call metadata for debugging and analysis."""

    __tablename__ = "llm_logs"

    analysis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="SET NULL"),
        nullable=True,
    )
    provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    prompt_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    completion_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    total_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    estimated_cost: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    log_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_llm_logs_analysis_id", "analysis_id"),
        Index("ix_llm_logs_created_at", "created_at"),
        Index("ix_llm_logs_provider_model", "provider", "model"),
    )
