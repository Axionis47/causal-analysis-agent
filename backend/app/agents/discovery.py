"""Causal discovery agent using PC/GES/FCI algorithms."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from app.agents.base import AgentResult, BaseAgent
from app.agents.llm_utils import call_llm_with_retry
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.crud.dataset import dataset_crud
from app.db.database import get_session_context
from app.llm.router import ROUTER
from app.models.agent_interaction import AgentInteraction, QuestionType
from app.models.analysis_stage import StageType
from app.models.causal_graph import CausalGraph, DiscoveryMethod
from app.prompts.discovery import build_discovery_prompt
from app.services.data_loader import load_dataframe
from app.services.retry import is_transient_io_error, is_transient_tool_error, retry_sync
from app.services.storage import infer_local_path
from app.services.tracing import traced
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)

# Cross-validation configuration defaults
DEFAULT_CV_FOLDS = 5
DEFAULT_STABILITY_THRESHOLD = 0.6
DEFAULT_SUBSAMPLE_FRACTION = 0.8


class CausalDiscoveryAgent(BaseAgent):
    name = "Causal Discovery Agent"
    stage = StageType.DISCOVERY

    async def _run(self, state):
        if state.get("causal_graph_ids"):
            return AgentResult(state=state, outputs={"cached": True}, message="Graphs cached")

        agent_logger = logger.bind(
            analysis_id=state.get("analysis_id"),
            stage=self.stage.value,
            agent_name=self.name,
        )
        analysis_id = uuid.UUID(state["analysis_id"])
        dataset_id = uuid.UUID(state["dataset_id"])
        async with get_session_context() as session:
            dataset = await dataset_crud.get(session, dataset_id)
            analysis = await analysis_crud.get(session, analysis_id)
            if dataset is None or analysis is None:
                raise ValueError("Dataset or analysis not found")

        df = retry_sync(_load_dataframe, dataset, retry_on=is_transient_io_error)
        data_matrix, variable_names = _prepare_matrix(df)
        registry = ToolRegistry()
        tools = registry.available_tools(
            "discovery",
            sample_size=len(df),
            data_types=["numerical"],
            overrides=analysis.config or {},
        )
        if not tools:
            tools = registry.available_tools("discovery", sample_size=len(df), data_types=["numerical"], overrides={})

        graph_ids: list[str] = []
        outputs: dict[str, Any] = {"tools": []}
        graph_payload: dict[str, Any] = {}

        # Get cross-validation configuration
        num_folds = getattr(settings, "DISCOVERY_CV_FOLDS", DEFAULT_CV_FOLDS)
        stability_threshold = getattr(settings, "DISCOVERY_STABILITY_THRESHOLD", DEFAULT_STABILITY_THRESHOLD)
        subsample_fraction = getattr(settings, "DISCOVERY_SUBSAMPLE_FRACTION", DEFAULT_SUBSAMPLE_FRACTION)

        # Allow analysis config override for subsample fraction if supported
        analysis_config = analysis.config or {}
        if "discovery_subsample_fraction" in analysis_config:
            subsample_fraction = float(analysis_config["discovery_subsample_fraction"])

        for tool in tools:
            agent_logger.info(
                "Discovery algorithm selected",
                algorithm=tool.name,
                sample_size=len(df),
                variable_count=len(variable_names),
            )
            start = time.time()
            graph_payload = retry_sync(
                _run_discovery,
                tool.name,
                data_matrix,
                variable_names,
                retry_on=is_transient_tool_error,
            )
            elapsed = time.time() - start
            agent_logger.info(
                "Graph generated",
                algorithm=tool.name,
                node_count=len(graph_payload.get("nodes", [])),
                edge_count=len(graph_payload.get("edges", [])),
                confidence_threshold=0.5,
            )

            # Run cross-validation for stability selection
            cv_start = time.time()
            cv_result = retry_sync(
                _cross_validate_discovery,
                data_matrix,
                variable_names,
                tool.name,
                num_folds=num_folds,
                stability_threshold=stability_threshold,
                subsample_fraction=subsample_fraction,
                retry_on=is_transient_tool_error,
            )
            cv_elapsed = time.time() - cv_start

            if cv_result:
                agent_logger.info(
                    "Cross-validation completed",
                    algorithm=tool.name,
                    stable_edge_count=len(cv_result.get("stable_edges", [])),
                    original_edge_count=len(graph_payload.get("edges", [])),
                    mean_edge_stability=cv_result.get("mean_stability"),
                    cv_folds=num_folds,
                )

                # Build consensus graph from stable edges
                consensus_graph = _build_consensus_graph(
                    cv_result.get("edge_frequencies", {}),
                    variable_names,
                    stability_threshold,
                )

                # Check if consensus graph differs significantly from original
                original_edge_count = len(graph_payload.get("edges", []))
                consensus_edge_count = len(consensus_graph.get("edges", []))
                edge_reduction = (original_edge_count - consensus_edge_count) / max(original_edge_count, 1)

                if edge_reduction > 0.5:
                    # Significant edge reduction - add question for user confirmation
                    question = AgentInteraction(
                        analysis_id=analysis_id,
                        question=f"Cross-validation reduced edges by {edge_reduction:.0%}. Use stable consensus graph?",
                        question_type=QuestionType.CAUSAL_DIRECTION,
                        confidence=cv_result.get("mean_stability", 0.5),
                        options=["Use consensus graph (more conservative)", "Use original graph (all edges)"],
                        default_response="Use consensus graph (more conservative)",
                        response_used="Use consensus graph (more conservative)",
                    )
                    async with get_session_context() as session:
                        session.add(question)
                    state.setdefault("questions", []).append({
                        "question": question.question,
                        "confidence": cv_result.get("mean_stability", 0.5),
                        "options": question.options,
                    })

            llm_insights = await _interpret_graph(
                graph_payload, agent_instance=self, state=state,
            )
            if llm_insights and llm_insights.get("confidence") is not None:
                graph_payload["confidence"] = float(llm_insights["confidence"])
            registry.track_performance(f"discovery:{tool.name}", success=True, elapsed=elapsed)

            # Store original graph
            async with get_session_context() as session:
                algorithm_params = {
                    "tool": tool.name,
                    "alpha": 0.05,
                    "cross_validation_stats": {
                        "num_folds": num_folds,
                        "edge_stability_threshold": stability_threshold,
                        "cv_elapsed_seconds": cv_elapsed,
                    } if cv_result else None,
                }

                graph = CausalGraph(
                    analysis_id=analysis_id,
                    method=_method_from_tool(tool.name),
                    edges=graph_payload["edges"],
                    nodes=graph_payload["nodes"],
                    graph_data=graph_payload["graph_data"],
                    algorithm_params=algorithm_params,
                    confidence=graph_payload.get("confidence"),
                    execution_time_seconds=elapsed,
                )
                session.add(graph)
                await session.flush()
                graph_ids.append(str(graph.id))

                # Store consensus graph if cross-validation was successful
                if cv_result and consensus_graph:
                    consensus_params = {
                        "tool": f"{tool.name}_consensus",
                        "alpha": 0.05,
                        "is_consensus_graph": True,
                        "cross_validation_stats": {
                            "num_folds": num_folds,
                            "edge_stability_threshold": stability_threshold,
                            "stable_edges": cv_result.get("stable_edges", []),
                            "edge_frequencies": cv_result.get("edge_frequencies", {}),
                            "mean_stability": cv_result.get("mean_stability"),
                        },
                    }

                    consensus = CausalGraph(
                        analysis_id=analysis_id,
                        method=_method_from_tool(tool.name),
                        edges=consensus_graph["edges"],
                        nodes=consensus_graph["nodes"],
                        graph_data=consensus_graph["graph_data"],
                        algorithm_params=consensus_params,
                        confidence=cv_result.get("mean_stability"),
                        execution_time_seconds=cv_elapsed,
                    )
                    session.add(consensus)
                    await session.flush()
                    graph_ids.append(str(consensus.id))

            outputs["tools"].append({
                "tool": tool.name,
                "elapsed": elapsed,
                "cv_elapsed": cv_elapsed if cv_result else None,
                "stable_edges": len(cv_result.get("stable_edges", [])) if cv_result else None,
            })

        state["causal_graph_ids"] = graph_ids

        confidence = _estimate_confidence(graph_payload.get("edges", []))
        if confidence < 0.5:
            question = AgentInteraction(
                analysis_id=analysis_id,
                question="Low confidence in causal directions. Confirm key treatment/outcome variables?",
                question_type=QuestionType.CAUSAL_DIRECTION,
                confidence=confidence,
                options=state.get("data_characteristics", {}).get("treatment_candidates", []),
                default_response="Proceed with inferred directions",
                response_used="Proceed with inferred directions",
            )
            async with get_session_context() as session:
                session.add(question)
            state.setdefault("questions", []).append({
                "question": question.question,
                "confidence": confidence,
                "options": question.options,
            })

        return AgentResult(state=state, outputs=outputs, message="Discovery complete")


def _load_dataframe(dataset) -> pd.DataFrame:
    """Load dataframe, preferring preprocessed data if available.

    Attempts to load data in the following order:
    1. Preprocessed data from local path
    2. Preprocessed data from GCS (downloads to local path)
    3. Raw data from local path

    This ensures GCS-only artifacts are not ignored.
    """
    from pathlib import Path
    from app.services.data_loader import load_preprocessed_dataframe

    # Try to load preprocessed data using the centralized loader
    # which handles both local and GCS paths
    preprocessed_df = load_preprocessed_dataframe(dataset)
    if preprocessed_df is not None:
        logger.info(
            "Loaded preprocessed data for discovery",
            row_count=len(preprocessed_df),
            column_count=preprocessed_df.shape[1],
        )
        return preprocessed_df

    # If preprocessed data not available via sync loader, try async GCS download
    # This handles cases where we're not in an async context but need GCS data
    if dataset.dataset_metadata:
        preprocessed_path = dataset.dataset_metadata.get("preprocessed_path")
        preprocessed_gcs_path = dataset.dataset_metadata.get("preprocessed_gcs_path")

        if preprocessed_path and preprocessed_gcs_path:
            pp_path = Path(preprocessed_path)
            if not pp_path.exists():
                try:
                    import asyncio
                    from app.services.storage import download_to_path

                    # Try to download from GCS
                    pp_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                download_to_path(preprocessed_gcs_path, pp_path)
                            )
                            future.result(timeout=60)
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run
                        asyncio.run(download_to_path(preprocessed_gcs_path, pp_path))

                    if pp_path.exists():
                        logger.info(
                            "Downloaded and loading preprocessed data from GCS for discovery",
                            file_path=str(pp_path),
                        )
                        return load_dataframe(pp_path)
                except Exception as exc:
                    logger.warning(
                        "Failed to download preprocessed data from GCS for discovery",
                        error=str(exc),
                    )

    # Fall back to raw data
    local_path = infer_local_path(dataset.dataset_metadata)
    if local_path and local_path.exists():
        logger.info(
            "Falling back to raw data for discovery",
            file_path=str(local_path),
        )
        return load_dataframe(local_path)
    raise ValueError("No dataset path available for discovery (checked local and GCS)")


def _prepare_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Prepare data matrix for discovery algorithms.

    Assumes data is already preprocessed (numeric columns filled, categoricals encoded).
    Applies emergency fallback for any remaining issues.

    IMPORTANT: Filters out zero-variance columns to avoid blowing up algorithm complexity.
    The PC algorithm is O(n^d) where n is number of variables - zero-variance columns
    add complexity without providing any causal information.
    """
    data = []
    names = []
    dropped_zero_variance = []

    # Collect numeric columns (preprocessed data should already be numeric)
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        series = numeric_df[col]

        # Skip zero-variance columns - they provide no causal information
        # and massively increase algorithm complexity
        if series.std() == 0 or series.nunique() <= 1:
            dropped_zero_variance.append(col)
            continue

        # Emergency fallback: if missing values detected, log warning and fill
        if series.isna().any():
            logger.warning(
                "Missing values detected in preprocessed data; applying emergency fillna",
                column=col,
                missing_count=int(series.isna().sum()),
            )
            series = series.fillna(series.mean())
        data.append(series.to_numpy())
        names.append(col)

    # Handle any remaining categorical columns (fallback for non-preprocessed data)
    for col in df.columns:
        if col in names or col in dropped_zero_variance:
            continue
        series = df[col]
        # Skip zero-variance categorical columns too
        if series.nunique() <= 1:
            dropped_zero_variance.append(col)
            continue
        if series.nunique() <= 20:
            logger.warning(
                "Non-numeric column in discovery; applying emergency encoding",
                column=col,
            )
            encoded = pd.factorize(series.fillna("missing"))[0]
            data.append(encoded)
            names.append(col)

    if dropped_zero_variance:
        logger.info(
            "Dropped zero-variance columns for discovery (no causal information)",
            dropped_count=len(dropped_zero_variance),
            dropped_columns=dropped_zero_variance[:10],  # Show first 10
        )

    if not data:
        raise ValueError("No usable columns for discovery")

    logger.info(
        "Prepared discovery matrix",
        variable_count=len(names),
        sample_count=len(df),
        variables=names,
    )

    matrix = np.column_stack(data)
    return matrix, names


def _run_discovery(tool_name: str, data: np.ndarray, names: list[str]) -> dict[str, Any]:
    if tool_name == "pc":
        return _run_pc(data, names)
    if tool_name == "ges":
        return _run_ges(data, names)
    if tool_name == "fci":
        return _run_fci(data, names)
    return _fallback_graph(names)


@traced("tool.discovery.pc")
def _run_pc(data: np.ndarray, names: list[str]) -> dict[str, Any]:
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except Exception:
        return _fallback_graph(names)
    cg = pc(data, alpha=0.05)
    return _graph_from_causallearn(cg.G, names)


@traced("tool.discovery.ges")
def _run_ges(data: np.ndarray, names: list[str]) -> dict[str, Any]:
    try:
        from causallearn.search.ScoreBased.GES import ges
    except Exception:
        return _fallback_graph(names)
    result = ges(data)
    return _graph_from_causallearn(result["G"], names)


@traced("tool.discovery.fci")
def _run_fci(data: np.ndarray, names: list[str]) -> dict[str, Any]:
    try:
        from causallearn.search.ConstraintBased.FCI import fci
    except Exception:
        return _fallback_graph(names)
    graph, _ = fci(data, alpha=0.05)
    return _graph_from_causallearn(graph, names)


def _graph_from_causallearn(graph, names: list[str]) -> dict[str, Any]:
    edges = []
    nodes = [{"name": name, "node_type": "variable"} for name in names]
    try:
        matrix = graph.graph
    except Exception:
        return _fallback_graph(names)

    for i, source in enumerate(names):
        for j, target in enumerate(names):
            if i == j:
                continue
            if matrix[i, j] != 0:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "edge_type": "directed",
                        "confidence": 0.6,
                    }
                )
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(names)
    nx_graph.add_edges_from([(e["source"], e["target"]) for e in edges])
    return {
        "nodes": nodes,
        "edges": edges,
        "graph_data": nx.node_link_data(nx_graph),
        "confidence": _estimate_confidence(edges),
    }


def _fallback_graph(names: list[str]) -> dict[str, Any]:
    nodes = [{"name": name, "node_type": "variable"} for name in names]
    edges = []
    for idx in range(min(len(names) - 1, 3)):
        edges.append({"source": names[idx], "target": names[idx + 1], "edge_type": "directed", "confidence": 0.4})
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(names)
    nx_graph.add_edges_from([(e["source"], e["target"]) for e in edges])
    return {"nodes": nodes, "edges": edges, "graph_data": nx.node_link_data(nx_graph), "confidence": 0.4}


def _estimate_confidence(edges: list[dict[str, Any]]) -> float:
    if not edges:
        return 0.0
    return float(sum(edge.get("confidence", 0.5) for edge in edges) / len(edges))


def _method_from_tool(name: str) -> DiscoveryMethod:
    if name == "ges":
        return DiscoveryMethod.GES
    if name == "fci":
        return DiscoveryMethod.FCI
    return DiscoveryMethod.PC


async def _interpret_graph(
    payload: dict[str, Any],
    agent_instance: CausalDiscoveryAgent | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = build_discovery_prompt(payload.get("nodes", []), payload.get("edges", []))
    analysis_id = state.get("analysis_id") if state else None
    result = await call_llm_with_retry(ROUTER.structured_output, prompt, analysis_id=analysis_id)

    # Track LLM usage if agent instance and state are provided
    if agent_instance and state and result:
        response_text = str(result) if result else ""
        estimated_tokens = agent_instance._estimate_tokens(prompt, response_text)
        await agent_instance._track_llm_usage(state, estimated_tokens, "gpt-4")

    return result


@traced("tool.discovery.cross_validation")
def _cross_validate_discovery(
    data: np.ndarray,
    names: list[str],
    algorithm_name: str,
    num_folds: int = 5,
    stability_threshold: float = 0.6,
    subsample_fraction: float = 0.8,
) -> dict[str, Any] | None:
    """
    Perform cross-validation for causal discovery to assess edge stability.

    Runs the discovery algorithm on multiple subsamples of the data and
    computes the frequency with which each edge appears across folds.

    Args:
        data: Data matrix (n_samples x n_variables)
        names: Variable names
        algorithm_name: Name of discovery algorithm (pc, ges, fci)
        num_folds: Number of cross-validation folds
        stability_threshold: Minimum frequency for an edge to be considered stable
        subsample_fraction: Fraction of data to use in each fold

    Returns:
        Dict with stable_edges, edge_frequencies, consensus_graph, mean_stability
    """
    n_samples = data.shape[0]
    subsample_size = int(n_samples * subsample_fraction)

    if subsample_size < 50:
        # Not enough data for meaningful cross-validation
        return None

    # Track edge frequencies across folds
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    all_edges: set[tuple[str, str]] = set()

    try:
        for fold in range(num_folds):
            # Random subsample (without replacement for stability)
            np.random.seed(fold * 42)  # Deterministic for reproducibility
            indices = np.random.choice(n_samples, size=subsample_size, replace=False)
            data_subsample = data[indices]

            # Run discovery algorithm on subsample
            try:
                graph_result = _run_discovery(algorithm_name, data_subsample, names)
                edges = graph_result.get("edges", [])

                for edge in edges:
                    source = edge.get("source")
                    target = edge.get("target")
                    if source and target:
                        edge_key = (source, target)
                        edge_counts[edge_key] += 1
                        all_edges.add(edge_key)
            except Exception:
                # Skip failed folds
                continue

        if not all_edges:
            return None

        # Compute edge frequencies
        edge_frequencies = {
            f"{source}->{target}": count / num_folds
            for (source, target), count in edge_counts.items()
        }

        # Identify stable edges (appeared in >= threshold fraction of folds)
        stable_edges = [
            {"source": source, "target": target, "frequency": count / num_folds}
            for (source, target), count in edge_counts.items()
            if count / num_folds >= stability_threshold
        ]

        # Compute mean stability across all edges
        if edge_frequencies:
            mean_stability = sum(edge_frequencies.values()) / len(edge_frequencies)
        else:
            mean_stability = 0.0

        return {
            "stable_edges": stable_edges,
            "edge_frequencies": edge_frequencies,
            "num_folds": num_folds,
            "stability_threshold": stability_threshold,
            "mean_stability": mean_stability,
            "total_unique_edges": len(all_edges),
        }
    except Exception as e:
        logger.warning("Cross-validation failed", algorithm=algorithm_name, error=str(e))
        return None


def _build_consensus_graph(
    edge_frequencies: dict[str, float],
    names: list[str],
    stability_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Build a consensus graph from edge frequency data.

    Includes only edges that appeared in at least stability_threshold
    fraction of cross-validation folds.

    Args:
        edge_frequencies: Dict mapping "source->target" to frequency (0-1)
        names: List of variable names
        stability_threshold: Minimum frequency for edge inclusion

    Returns:
        Graph payload with nodes, edges, and graph_data
    """
    nodes = [{"name": name, "node_type": "variable"} for name in names]
    edges = []

    for edge_str, frequency in edge_frequencies.items():
        if frequency >= stability_threshold:
            parts = edge_str.split("->")
            if len(parts) == 2:
                source, target = parts
                edges.append({
                    "source": source,
                    "target": target,
                    "edge_type": "directed",
                    "confidence": frequency,  # Use stability as confidence
                })

    # Build NetworkX graph
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(names)
    nx_graph.add_edges_from([(e["source"], e["target"]) for e in edges])

    return {
        "nodes": nodes,
        "edges": edges,
        "graph_data": nx.node_link_data(nx_graph),
        "confidence": _estimate_confidence(edges) if edges else 0.0,
    }
