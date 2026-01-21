"""Ground truth validation module for E2E evaluation of causal analysis results.

This module provides validation utilities for comparing analysis results against
domain-expected ground truth values defined in datasets.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evals.metrics import ate_bias, precision_recall


def load_datasets_yaml() -> dict[str, Any]:
    """Load the datasets registry from YAML file.

    Returns:
        Dictionary containing all dataset configurations.

    Raises:
        FileNotFoundError: If datasets.yaml is not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    yaml_path = Path(__file__).parent / "datasets.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("datasets", {})


def _normalize_edge(edge: tuple[str, str] | dict[str, str]) -> tuple[str, str]:
    """Normalize edge representation to lowercase tuple.

    Args:
        edge: Edge as tuple or dict with 'source' and 'target' keys.

    Returns:
        Tuple of (source, target) with normalized casing and whitespace.
    """
    if isinstance(edge, dict):
        source = edge.get("source", "")
        target = edge.get("target", "")
    else:
        source, target = edge
    return (source.strip().lower(), target.strip().lower())


def _edges_to_set(edges: list[tuple[str, str] | dict[str, str]]) -> set[tuple[str, str]]:
    """Convert list of edges to normalized set of tuples.

    Args:
        edges: List of edges in various formats.

    Returns:
        Set of normalized (source, target) tuples.
    """
    return {_normalize_edge(edge) for edge in edges}


def compute_graph_metrics(
    true_edges: list[tuple[str, str] | dict[str, str]],
    pred_edges: list[tuple[str, str] | dict[str, str]],
) -> dict[str, float]:
    """Compute precision, recall, and F1 for graph edge prediction.

    Reuses the precision_recall function from evals.metrics but handles
    edge normalization for case-insensitive comparison.

    Args:
        true_edges: Ground truth edges from dataset config.
        pred_edges: Predicted edges from analysis.

    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    true_set = _edges_to_set(true_edges)
    pred_set = _edges_to_set(pred_edges)
    # Convert back to list of tuples for precision_recall
    return precision_recall(list(true_set), list(pred_set))


def format_validation_report(dataset_id: str, results: dict[str, Any]) -> str:
    """Generate a human-readable validation report.

    Args:
        dataset_id: Identifier of the dataset being validated.
        results: Dictionary containing validation results.

    Returns:
        Formatted string report.
    """
    lines = [
        f"Validation Report: {dataset_id}",
        "=" * 50,
    ]

    if "ate" in results:
        ate = results["ate"]
        status = "PASS" if ate["passed"] else "FAIL"
        lines.extend([
            "",
            f"ATE Validation: {status}",
            f"  Estimated ATE: {ate.get('estimated', 'N/A')}",
            f"  Expected Range: {ate.get('expected_range', 'N/A')}",
            f"  Bias: {ate.get('bias', 'N/A')}",
            f"  Tolerance: {ate.get('tolerance', 'N/A')}",
        ])

    if "graph" in results:
        graph = results["graph"]
        status = "PASS" if graph["passed"] else "FAIL"
        lines.extend([
            "",
            f"Graph Validation: {status}",
            f"  Precision: {graph.get('precision', 0):.3f}",
            f"  Recall: {graph.get('recall', 0):.3f}",
            f"  F1 Score: {graph.get('f1', 0):.3f}",
            f"  Threshold: {graph.get('threshold', 'N/A')}",
        ])

    if "refutation" in results:
        ref = results["refutation"]
        status = "PASS" if ref["passed"] else "FAIL"
        lines.extend([
            "",
            f"Refutation Validation: {status}",
            f"  Pass Rate: {ref.get('pass_rate', 0):.3f}",
            f"  Threshold: {ref.get('threshold', 'N/A')}",
        ])

    overall = all(
        r.get("passed", False)
        for r in [results.get("ate", {}), results.get("graph", {}), results.get("refutation", {})]
        if r
    )
    lines.extend([
        "",
        "=" * 50,
        f"Overall: {'PASS' if overall else 'FAIL'}",
    ])

    return "\n".join(lines)


class GroundTruth:
    """Ground truth validation for causal analysis results.

    This class loads dataset configurations from datasets.yaml and provides
    methods to validate analysis results against domain-expected ground truth.
    """

    def __init__(self) -> None:
        """Initialize GroundTruth with datasets from YAML."""
        self._datasets = load_datasets_yaml()

    @property
    def dataset_ids(self) -> list[str]:
        """Get list of all available dataset IDs."""
        return list(self._datasets.keys())

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Get full configuration for a dataset.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dictionary containing all dataset configuration.

        Raises:
            KeyError: If dataset_id is not found.
        """
        if dataset_id not in self._datasets:
            available = ", ".join(self.dataset_ids)
            raise KeyError(f"Dataset '{dataset_id}' not found. Available: {available}")
        return self._datasets[dataset_id]

    def validate_ate(
        self, dataset_id: str, estimated_ate: float
    ) -> dict[str, Any]:
        """Validate estimated ATE against ground truth range.

        Args:
            dataset_id: Unique identifier for the dataset.
            estimated_ate: The estimated average treatment effect.

        Returns:
            Dictionary with validation result:
                - passed: bool - Whether ATE is within tolerance
                - estimated: float - The estimated ATE value
                - expected_range: tuple - (min, max) expected range
                - expected_midpoint: float - Midpoint of expected range
                - bias: float - Difference from midpoint
                - tolerance: float - Allowed deviation

        Raises:
            KeyError: If dataset_id is not found.
            ValueError: If ground truth ATE range is not defined for dataset.
        """
        dataset = self.get_dataset(dataset_id)
        ground_truth = dataset.get("ground_truth", {})

        ate_range = ground_truth.get("ate_range")
        if ate_range is None:
            raise ValueError(
                f"Dataset '{dataset_id}' does not have ATE ground truth defined. "
                "This dataset may focus on graph discovery only."
            )

        min_ate = ate_range["min"]
        max_ate = ate_range["max"]
        tolerance = ground_truth.get("ate_tolerance", 0.5)

        expected_midpoint = (min_ate + max_ate) / 2
        bias = ate_bias(expected_midpoint, estimated_ate)

        # Check if within tolerance of the expected range
        passed = (min_ate - tolerance) <= estimated_ate <= (max_ate + tolerance)

        return {
            "passed": passed,
            "estimated": estimated_ate,
            "expected_range": (min_ate, max_ate),
            "expected_midpoint": expected_midpoint,
            "bias": bias,
            "tolerance": tolerance,
        }

    def validate_graph(
        self, dataset_id: str, predicted_edges: list[tuple[str, str] | dict[str, str]]
    ) -> dict[str, Any]:
        """Validate predicted graph edges against expected structure.

        Args:
            dataset_id: Unique identifier for the dataset.
            predicted_edges: List of predicted edges as tuples or dicts.

        Returns:
            Dictionary with validation result:
                - passed: bool - Whether F1 meets threshold
                - precision: float - Edge prediction precision
                - recall: float - Edge prediction recall
                - f1: float - F1 score
                - threshold: float - Minimum required F1
                - expected_edge_count: int - Number of expected edges
                - predicted_edge_count: int - Number of predicted edges

        Raises:
            KeyError: If dataset_id is not found.
        """
        dataset = self.get_dataset(dataset_id)
        ground_truth = dataset.get("ground_truth", {})
        causal_structure = dataset.get("causal_structure", {})

        expected_edges = causal_structure.get("expected_edges", [])
        threshold = ground_truth.get("graph_f1_threshold", 0.7)

        metrics = compute_graph_metrics(expected_edges, predicted_edges)

        return {
            "passed": metrics["f1"] >= threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": threshold,
            "expected_edge_count": len(expected_edges),
            "predicted_edge_count": len(predicted_edges),
        }

    def validate_refutation(
        self, dataset_id: str, pass_rate: float
    ) -> dict[str, Any]:
        """Validate refutation test pass rate.

        Args:
            dataset_id: Unique identifier for the dataset.
            pass_rate: Proportion of refutation tests that passed (0.0 to 1.0).

        Returns:
            Dictionary with validation result:
                - passed: bool - Whether pass rate meets threshold
                - pass_rate: float - Actual pass rate
                - threshold: float - Minimum required pass rate

        Raises:
            KeyError: If dataset_id is not found.
        """
        dataset = self.get_dataset(dataset_id)
        ground_truth = dataset.get("ground_truth", {})

        threshold = ground_truth.get("validation_pass_rate_threshold", 0.8)

        return {
            "passed": pass_rate >= threshold,
            "pass_rate": pass_rate,
            "threshold": threshold,
        }

    def get_analysis_config(self, dataset_id: str) -> dict[str, Any]:
        """Get analysis configuration for API submission.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dictionary containing analysis configuration suitable
            for submitting to the causal analysis API. Uses the API's
            expected field names: treatment_variable and outcome_variable.

        Raises:
            KeyError: If dataset_id is not found.
        """
        dataset = self.get_dataset(dataset_id)
        analysis_config = dataset.get("analysis_config", {})
        causal_structure = dataset.get("causal_structure", {})

        return {
            "treatment_variable": causal_structure.get("treatment"),
            "outcome_variable": causal_structure.get("outcome"),
            "confounders": causal_structure.get("confounders", []),
            "analysis_types": analysis_config.get("analysis_types", ["discovery"]),
            "enable_preprocessing": analysis_config.get("enable_preprocessing", True),
            "preprocessing_config": analysis_config.get("preprocessing_config", {}),
            "override_quality_warnings": analysis_config.get(
                "override_quality_warnings", False
            ),
        }

    def get_kaggle_info(self, dataset_id: str) -> dict[str, Any]:
        """Get Kaggle download information for a dataset.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dictionary with Kaggle metadata:
                - url: str - Kaggle URL
                - kaggle_type: str - 'competition' or 'dataset'
                - expected_file: str - Primary CSV filename

        Raises:
            KeyError: If dataset_id is not found.
        """
        dataset = self.get_dataset(dataset_id)
        metadata = dataset.get("metadata", {})

        return {
            "url": dataset.get("kaggle_url"),
            "kaggle_type": metadata.get("kaggle_type", "dataset"),
            "expected_file": metadata.get("expected_file"),
        }

    def get_stress_test_focus(self, dataset_id: str) -> list[str]:
        """Get stress test features for a dataset.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            List of stress test focus areas for this dataset.

        Raises:
            KeyError: If dataset_id is not found.
        """
        dataset = self.get_dataset(dataset_id)
        return dataset.get("stress_test_focus", [])

    def validate_all(
        self,
        dataset_id: str,
        estimated_ate: float | None = None,
        predicted_edges: list[tuple[str, str] | dict[str, str]] | None = None,
        refutation_pass_rate: float | None = None,
    ) -> dict[str, Any]:
        """Validate all provided metrics against ground truth.

        Args:
            dataset_id: Unique identifier for the dataset.
            estimated_ate: Optional estimated ATE to validate.
            predicted_edges: Optional predicted edges to validate.
            refutation_pass_rate: Optional refutation pass rate to validate.

        Returns:
            Dictionary with all validation results.

        Raises:
            KeyError: If dataset_id is not found.
        """
        results: dict[str, Any] = {"dataset_id": dataset_id}

        if estimated_ate is not None:
            try:
                results["ate"] = self.validate_ate(dataset_id, estimated_ate)
            except ValueError as e:
                results["ate"] = {"passed": None, "error": str(e)}

        if predicted_edges is not None:
            results["graph"] = self.validate_graph(dataset_id, predicted_edges)

        if refutation_pass_rate is not None:
            results["refutation"] = self.validate_refutation(
                dataset_id, refutation_pass_rate
            )

        # Overall pass status
        validations = [results.get(k) for k in ["ate", "graph", "refutation"]]
        passed_results = [v for v in validations if v and v.get("passed") is not None]
        results["overall_passed"] = all(v["passed"] for v in passed_results) if passed_results else None

        return results

    def generate_report(
        self,
        dataset_id: str,
        estimated_ate: float | None = None,
        predicted_edges: list[tuple[str, str] | dict[str, str]] | None = None,
        refutation_pass_rate: float | None = None,
    ) -> str:
        """Generate a human-readable validation report.

        Args:
            dataset_id: Unique identifier for the dataset.
            estimated_ate: Optional estimated ATE to validate.
            predicted_edges: Optional predicted edges to validate.
            refutation_pass_rate: Optional refutation pass rate to validate.

        Returns:
            Formatted string report.
        """
        results = self.validate_all(
            dataset_id,
            estimated_ate=estimated_ate,
            predicted_edges=predicted_edges,
            refutation_pass_rate=refutation_pass_rate,
        )
        return format_validation_report(dataset_id, results)
