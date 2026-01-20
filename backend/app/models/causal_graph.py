"""CausalGraph model for storing discovered causal relationships."""

import enum
import uuid
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class DiscoveryMethod(str, enum.Enum):
    """Causal discovery algorithm used."""

    PC = "pc"
    GES = "ges"
    FCI = "fci"
    OTHER = "other"


class CausalGraph(BaseModel):
    """
    CausalGraph model representing a discovered causal DAG.

    Stores the graph structure, discovery method, and confidence metrics.
    """

    __tablename__ = "causal_graphs"

    # Foreign key to analysis
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Discovery method
    method: Mapped[DiscoveryMethod] = mapped_column(
        Enum(DiscoveryMethod),
        nullable=False,
    )

    # Graph structure (JSONB for NetworkX serialization)
    edges: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        comment="List of edges with source, target, edge_type, confidence, p_value",
    )
    nodes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        comment="List of nodes with name and node_type",
    )
    graph_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Serialized NetworkX graph for visualization",
    )

    # Algorithm parameters and metadata
    algorithm_params: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Parameters used for the discovery algorithm",
    )

    # Confidence and quality metrics
    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Overall confidence score (0-1)",
    )
    execution_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time taken to run the algorithm",
    )

    # Graph versioning
    parent_graph_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("causal_graphs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Parent graph for user-modified versions",
    )
    is_user_modified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this graph was modified by the user",
    )

    # Relationships
    analysis: Mapped["Analysis"] = relationship(
        "Analysis",
        back_populates="causal_graphs",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_causal_graphs_analysis_method", "analysis_id", "method"),
    )

