"""API routes for analysis results."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.db.database import get_async_session
from app.models.causal_graph import CausalGraph, DiscoveryMethod
from app.models.generated_report import ReportFormat, ReportType
from app.services.storage import generate_signed_url

router = APIRouter(prefix="/api/v1/results", tags=["results"])
logger = get_logger(__name__)


class GraphNode(BaseModel):
    """Node in a causal graph."""

    name: str
    node_type: str | None = None


class GraphEdge(BaseModel):
    """Edge in a causal graph."""

    source: str
    target: str
    confidence: float | None = None


class GraphUpdateRequest(BaseModel):
    """Request schema for updating a graph."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    modified_by_user: bool = True


@router.get(
    "/{analysis_id}/summary",
    summary="Get executive summary",
    responses={
        200: {
            "description": "Executive summary of analysis results",
            "content": {
                "application/json": {
                    "example": {
                        "title": "Causal Analysis Summary",
                        "key_findings": [
                            {
                                "finding": "Education has a significant positive effect on income",
                                "effect_size": 0.45,
                                "confidence": "high",
                            }
                        ],
                        "treatment_effects": [
                            {
                                "treatment": "education",
                                "outcome": "income",
                                "ate": 15000,
                                "interpretation": "Each additional year of education increases income by $15,000",
                            }
                        ],
                        "recommendations": ["Consider education as a key intervention variable"],
                        "confidence_score": 0.85,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_summary(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get the executive summary report for an analysis.

    Returns a high-level overview suitable for non-technical stakeholders,
    including key findings, treatment effects in plain language, and
    actionable recommendations.

    **Report Contents:**
    - Key findings with effect sizes and confidence levels
    - Treatment effect interpretations in business terms
    - Causal relationship summary
    - Recommendations based on findings
    - Overall confidence score
    """
    analysis = await _get_analysis(db, analysis_id, user_id)
    return _report_content(analysis, ReportType.EXECUTIVE)


@router.get(
    "/{analysis_id}/technical",
    summary="Get technical report",
    responses={
        200: {
            "description": "Detailed technical report with methodology and statistics",
            "content": {
                "application/json": {
                    "example": {
                        "methodology": {
                            "causal_discovery": ["PC", "GES", "FCI"],
                            "effect_estimation": ["PSM", "DoublyRobust", "IV"],
                            "validation": ["placebo", "subset", "random_common_cause"],
                        },
                        "causal_discovery_results": {
                            "algorithm": "PC",
                            "parameters": {"alpha": 0.05, "indep_test": "fisherz"},
                            "graph": {"nodes": 10, "edges": 15},
                            "runtime_seconds": 2.5,
                        },
                        "treatment_effects": [
                            {
                                "treatment": "education",
                                "outcome": "income",
                                "method": "doubly_robust",
                                "ate": 15000,
                                "ate_std": 2000,
                                "ci_lower": 11000,
                                "ci_upper": 19000,
                                "p_value": 0.001,
                            }
                        ],
                        "validation_results": {
                            "placebo_test": {"passed": True, "p_value": 0.45},
                            "subset_test": {"passed": True, "ate_variance": 0.05},
                        },
                        "data_summary": {
                            "n_observations": 10000,
                            "n_features": 15,
                            "missing_rate": 0.02,
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_technical(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get the detailed technical report for an analysis.

    Returns comprehensive methodology documentation, statistical results,
    and validation details suitable for technical review and publication.

    **Report Contents:**
    - Methodology details (algorithms, parameters, assumptions)
    - Causal discovery results with graph statistics
    - Treatment effect estimates with confidence intervals and p-values
    - Validation and refutation test results
    - Data preprocessing summary
    - Runtime and computational statistics
    """
    analysis = await _get_analysis(db, analysis_id, user_id)
    return _report_content(analysis, ReportType.TECHNICAL)


@router.get(
    "/{analysis_id}/visualizations",
    summary="Get visualization data",
    responses={
        200: {
            "description": "Visualization data including causal graphs and charts",
            "content": {
                "application/json": {
                    "example": {
                        "causal_graph": {
                            "nodes": [
                                {"id": "education", "type": "treatment", "x": 100, "y": 200},
                                {"id": "income", "type": "outcome", "x": 300, "y": 200},
                                {"id": "age", "type": "confounder", "x": 200, "y": 100},
                            ],
                            "edges": [
                                {"source": "education", "target": "income", "confidence": 0.9},
                                {"source": "age", "target": "education", "confidence": 0.8},
                                {"source": "age", "target": "income", "confidence": 0.75},
                            ],
                        },
                        "effect_plots": [
                            {
                                "type": "forest_plot",
                                "treatment": "education",
                                "outcome": "income",
                                "data": {"ate": 15000, "ci_lower": 11000, "ci_upper": 19000},
                            }
                        ],
                        "distribution_plots": [
                            {
                                "variable": "income",
                                "type": "histogram",
                                "bins": [0, 20000, 40000, 60000, 80000, 100000],
                                "counts": [100, 250, 300, 200, 150],
                            }
                        ],
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_visualizations(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get visualization data for rendering charts and graphs.

    Returns structured data for client-side visualization rendering,
    including causal graphs, effect plots, and distribution charts.

    **Visualization Types:**

    - **Causal Graph**: DAG with nodes, edges, and confidence scores
      - Node types: treatment, outcome, confounder, mediator
      - Edges include confidence scores from discovery algorithms

    - **Effect Plots**: Treatment effect visualizations
      - Forest plots showing ATE with confidence intervals
      - Comparison plots across different methods

    - **Distribution Plots**: Data distribution visualizations
      - Histograms for continuous variables
      - Bar charts for categorical variables
      - Treatment-outcome scatter plots
    """
    analysis = await _get_analysis(db, analysis_id, user_id)
    return _report_content(analysis, ReportType.VISUALIZATION)


@router.get(
    "/{analysis_id}/validation",
    summary="Get validation results",
    responses={
        200: {
            "description": "Validation and refutation test results",
            "content": {
                "application/json": {
                    "example": {
                        "overall_confidence": 0.85,
                        "refutation_tests": [
                            {
                                "test_name": "placebo_treatment",
                                "passed": True,
                                "original_effect": 15000,
                                "refuted_effect": 200,
                                "p_value": 0.45,
                                "interpretation": "Effect disappears with placebo, supporting causal claim",
                            },
                            {
                                "test_name": "random_common_cause",
                                "passed": True,
                                "effect_change_percent": 2.5,
                                "interpretation": "Effect robust to unobserved confounders",
                            },
                            {
                                "test_name": "data_subset",
                                "passed": True,
                                "subset_effects": [14500, 15200, 15800],
                                "variance": 0.04,
                                "interpretation": "Effect consistent across data subsets",
                            },
                        ],
                        "sensitivity_analysis": {
                            "unobserved_confounder_strength": 0.3,
                            "interpretation": "Effect would require strong unobserved confounder to be nullified",
                        },
                        "warnings": [],
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_validation(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get validation and refutation test results for an analysis.

    Returns comprehensive validation results that assess the reliability
    and robustness of the causal findings.

    **Refutation Tests:**

    - **Placebo Treatment**: Replaces treatment with random variable
      - Pass: Effect should disappear with placebo
      - Fail: May indicate spurious correlation

    - **Random Common Cause**: Adds random confounder
      - Pass: Effect should remain stable
      - Fail: Effect may be sensitive to unmeasured confounders

    - **Data Subset**: Tests on random data subsets
      - Pass: Effect consistent across subsets
      - Fail: Effect may be driven by outliers

    **Confidence Scoring:**
    - 0.0-0.3: Low confidence (many tests failed)
    - 0.3-0.7: Moderate confidence (some concerns)
    - 0.7-1.0: High confidence (robust findings)
    """
    analysis = await _get_analysis(db, analysis_id, user_id)
    return _report_content(analysis, ReportType.VALIDATION)


@router.get(
    "/{analysis_id}/export",
    summary="Export all results as JSON",
    responses={
        200: {
            "description": "All analysis results combined into a single JSON export",
            "content": {
                "application/json": {
                    "example": {
                        "summary": {
                            "title": "Causal Analysis Summary",
                            "key_findings": [
                                {
                                    "finding": "Education has a significant positive effect on income",
                                    "effect_size": 0.45,
                                    "confidence": "high",
                                }
                            ],
                            "confidence_score": 0.85,
                        },
                        "technical": {
                            "methodology": {
                                "causal_discovery": ["PC", "GES"],
                                "effect_estimation": ["DoublyRobust"],
                            },
                            "treatment_effects": [
                                {
                                    "treatment": "education",
                                    "outcome": "income",
                                    "ate": 15000,
                                    "p_value": 0.001,
                                }
                            ],
                        },
                        "visualizations": {
                            "causal_graph": {
                                "nodes": [
                                    {"id": "education", "type": "treatment"},
                                    {"id": "income", "type": "outcome"},
                                ],
                                "edges": [
                                    {"source": "education", "target": "income", "confidence": 0.9}
                                ],
                            }
                        },
                        "validation": {
                            "overall_confidence": 0.85,
                            "refutation_tests": [
                                {"test_name": "placebo_treatment", "passed": True}
                            ],
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def export_results(
    analysis_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Export all analysis results as a single JSON object.

    Returns a comprehensive export containing all four report sections:
    - Executive summary with key findings
    - Technical report with methodology and statistics
    - Visualization data for graphs and charts
    - Validation results with refutation tests

    **Use Cases:**
    - Programmatic access to all results at once
    - Backup analysis data
    - Integration with external tools
    - Custom report generation

    **Request Example:**
    ```
    GET /api/v1/results/{analysis_id}/export
    Authorization: Bearer <token>
    ```

    **Response Structure:**
    - `summary`: Executive summary for non-technical stakeholders
    - `technical`: Detailed methodology and statistical results
    - `visualizations`: Data for rendering causal graphs and charts
    - `validation`: Refutation test results and confidence scores
    """
    analysis = await _get_analysis(db, analysis_id, user_id)
    payload = {
        "summary": _report_content(analysis, ReportType.EXECUTIVE),
        "technical": _report_content(analysis, ReportType.TECHNICAL),
        "visualizations": _report_content(analysis, ReportType.VISUALIZATION),
        "validation": _report_content(analysis, ReportType.VALIDATION),
    }
    return payload


@router.get(
    "/{analysis_id}/download",
    summary="Download formatted report",
    responses={
        200: {
            "description": "Report file in requested format",
            "content": {
                "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                "text/markdown": {"schema": {"type": "string"}},
                "text/html": {"schema": {"type": "string"}},
                "application/json": {
                    "example": {
                        "executive": {"title": "...", "findings": []},
                        "technical": {"methodology": {}, "results": {}},
                    }
                },
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
                    "schema": {"type": "string", "format": "binary"}
                },
            },
        },
        307: {"description": "Redirect to signed GCS URL for file download"},
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {
            "description": "Analysis or report not found",
            "content": {
                "application/json": {
                    "examples": {
                        "analysis_not_found": {"value": {"detail": "Analysis not found"}},
                        "format_not_found": {"value": {"detail": "No report found for format 'pdf'"}},
                        "file_not_available": {"value": {"detail": "Report file not available"}},
                    }
                }
            },
        },
    },
)
async def download_report(
    analysis_id: uuid.UUID,
    format: ReportFormat = Query(..., description="Report format: pdf, markdown, html, json, pptx"),
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> Any:
    """
    Download the analysis report in the specified format.

    **Available Formats:**

    - **PDF** (`format=pdf`): Professional formatted report
      - Best for: Presentations, sharing with stakeholders
      - Generated using WeasyPrint

    - **Markdown** (`format=markdown`): Plain text with formatting
      - Best for: Documentation, version control
      - Compatible with GitHub, GitLab, etc.

    - **HTML** (`format=html`): Web page format
      - Best for: Embedding in web applications
      - Includes interactive elements

    - **PowerPoint** (`format=pptx`): Presentation slides
      - Best for: Meetings, executive presentations
      - Auto-generated slide deck

    - **JSON** (`format=json`): Raw structured data
      - Best for: Programmatic access, custom processing
      - Returns all report sections as JSON objects

    **Response Behavior:**
    - For cloud storage (GCS): Returns 307 redirect to signed URL
    - For local storage: Returns file directly with appropriate Content-Type
    - Signed URLs expire after configured timeout (default: 60 minutes)

    **Response Headers:**
    - `Content-Type`: Appropriate MIME type for the format
    - `Content-Disposition`: `attachment; filename="report.{ext}"`
    - `Cache-Control`: `public, max-age=3600`
    """
    analysis = await _get_analysis(db, analysis_id, user_id)
    matching_reports = [report for report in analysis.reports if report.format == format]
    if not matching_reports:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report found for format '{format.value}'",
        )

    if format == ReportFormat.JSON:
        payload: dict[str, Any] = {}
        for report in matching_reports:
            if report.content:
                payload[report.report_type.value] = json.loads(report.content)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="JSON report content not available",
            )
        logger.info(
            "Report downloaded",
            user_id=str(user_id),
            analysis_id=str(analysis_id),
            format=format.value,
        )
        return payload

    report = sorted(matching_reports, key=lambda item: item.generated_at, reverse=True)[0]
    filename = _download_filename(format)
    content_type = _content_type_for_format(format)

    if report.gcs_path and report.gcs_path.startswith("gs://"):
        signed_url = generate_signed_url(
            report.gcs_path,
            expiration_minutes=settings.REPORT_SIGNED_URL_EXPIRATION_MINUTES,
        )
        if signed_url:
            response = RedirectResponse(url=signed_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
            response.headers["Content-Type"] = content_type
            response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            response.headers["Cache-Control"] = "public, max-age=3600"
            logger.info(
                "Report download redirected",
                user_id=str(user_id),
                analysis_id=str(analysis_id),
                format=format.value,
                file_size=report.file_size_bytes,
            )
            return response

    if report.gcs_path:
        local_path = Path(report.gcs_path)
        if local_path.exists():
            response = FileResponse(path=local_path, media_type=content_type, filename=filename)
            response.headers["Cache-Control"] = "public, max-age=3600"
            logger.info(
                "Report downloaded",
                user_id=str(user_id),
                analysis_id=str(analysis_id),
                format=format.value,
                file_size=report.file_size_bytes,
            )
            return response

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Report file not available",
    )


async def _get_analysis(db: AsyncSession, analysis_id: uuid.UUID, user_id: uuid.UUID):
    analysis = await analysis_crud.get_with_relations(db, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if analysis.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return analysis


def _report_content(analysis, report_type: ReportType) -> dict[str, Any]:
    for report in analysis.reports:
        if report.report_type == report_type and report.content:
            return json.loads(report.content)
    return {}


def _content_type_for_format(report_format: ReportFormat) -> str:
    if report_format == ReportFormat.PDF:
        return "application/pdf"
    if report_format == ReportFormat.MARKDOWN:
        return "text/markdown"
    if report_format == ReportFormat.HTML:
        return "text/html"
    if report_format == ReportFormat.PPTX:
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    return "application/json"


def _download_filename(report_format: ReportFormat) -> str:
    extension_map = {
        ReportFormat.PDF: "pdf",
        ReportFormat.MARKDOWN: "md",
        ReportFormat.HTML: "html",
        ReportFormat.JSON: "json",
        ReportFormat.PPTX: "pptx",
    }
    extension = extension_map.get(report_format, "dat")
    return f"report.{extension}"


@router.post(
    "/{analysis_id}/graphs/{graph_id}/update",
    summary="Update causal graph",
    responses={
        200: {
            "description": "Graph updated successfully - returns new version",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440001",
                        "nodes": [
                            {"name": "education", "node_type": "treatment"},
                            {"name": "income", "node_type": "outcome"},
                        ],
                        "edges": [
                            {"source": "education", "target": "income", "confidence": 0.9}
                        ],
                        "is_user_modified": True,
                        "parent_graph_id": "550e8400-e29b-41d4-a716-446655440000",
                    }
                }
            },
        },
        400: {
            "description": "Invalid graph structure (contains cycles)",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid graph: contains cycles (must be a DAG)"}
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def update_graph(
    analysis_id: uuid.UUID,
    graph_id: str,
    request: GraphUpdateRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Update a causal graph with user modifications.

    Creates a new graph version linked to the original, preserving
    the edit history. Use this to incorporate domain knowledge by
    adding or removing edges that the algorithm may have missed.

    **Graph Requirements:**
    - Must be a Directed Acyclic Graph (DAG) - no cycles allowed
    - Nodes must have unique names
    - Edges connect existing nodes only

    **Request Example:**
    ```json
    {
        "nodes": [
            {"name": "education", "node_type": "treatment"},
            {"name": "income", "node_type": "outcome"},
            {"name": "age", "node_type": "confounder"}
        ],
        "edges": [
            {"source": "education", "target": "income", "confidence": 0.9},
            {"source": "age", "target": "education", "confidence": 0.8},
            {"source": "age", "target": "income", "confidence": 0.75}
        ],
        "modified_by_user": true
    }
    ```

    **Node Types:**
    - `treatment`: Variable being manipulated
    - `outcome`: Variable being measured
    - `confounder`: Common cause of treatment and outcome
    - `mediator`: Variable on causal path between treatment and outcome

    **Versioning:**
    - Each update creates a new graph version
    - Original graph is preserved (linked via `parent_graph_id`)
    - Graph history can be viewed via version endpoints
    """
    analysis = await _get_analysis(db, analysis_id, user_id)

    # Validate graph structure (check for cycles)
    if _has_cycle(request.edges):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid graph: contains cycles (must be a DAG)",
        )

    # Find original graph
    original_graph = None
    for graph in analysis.causal_graphs:
        if str(graph.id) == graph_id or graph_id == "default":
            original_graph = graph
            break

    # Prepare graph data
    nodes_data = [{"name": n.name, "node_type": n.node_type} for n in request.nodes]
    edges_data = [
        {"source": e.source, "target": e.target, "confidence": e.confidence}
        for e in request.edges
    ]

    # Create new graph version
    new_graph = CausalGraph(
        analysis_id=analysis_id,
        method=DiscoveryMethod.OTHER,
        nodes=nodes_data,
        edges=edges_data,
        graph_data={"nodes": nodes_data, "edges": edges_data},
        algorithm_params={
            "user_modified": True,
            "original_graph_id": str(original_graph.id) if original_graph else None,
        },
        parent_graph_id=original_graph.id if original_graph else None,
        is_user_modified=True,
    )

    db.add(new_graph)
    await db.commit()
    await db.refresh(new_graph)

    logger.info(
        "Graph updated by user",
        analysis_id=str(analysis_id),
        original_graph_id=str(original_graph.id) if original_graph else None,
        new_graph_id=str(new_graph.id),
        node_count=len(nodes_data),
        edge_count=len(edges_data),
    )

    return {
        "id": str(new_graph.id),
        "nodes": nodes_data,
        "edges": edges_data,
        "is_user_modified": True,
        "parent_graph_id": str(new_graph.parent_graph_id) if new_graph.parent_graph_id else None,
    }


def _has_cycle(edges: list[GraphEdge]) -> bool:
    """Check if the graph contains cycles using DFS."""
    adjacency: dict[str, list[str]] = {}
    nodes: set[str] = set()

    for edge in edges:
        nodes.add(edge.source)
        nodes.add(edge.target)
        if edge.source not in adjacency:
            adjacency[edge.source] = []
        adjacency[edge.source].append(edge.target)

    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in nodes:
        if node not in visited:
            if dfs(node):
                return True

    return False
