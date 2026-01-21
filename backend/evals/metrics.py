"""Evaluation metrics for synthetic datasets and E2E validation.

This module provides metrics functions for:
- Precision/recall for graph edges
- ATE bias calculation
- Refutation test pass rates
- Completeness checks
- Confidence interval metrics
- P-value validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.validation_result import ValidationResult


def precision_recall(
    true_edges: Iterable[tuple[str, str]], pred_edges: Iterable[tuple[str, str]]
) -> dict[str, float]:
    """Calculate precision, recall, and F1 for edge prediction.

    Args:
        true_edges: Ground truth edges.
        pred_edges: Predicted edges.

    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def ate_bias(true_ate: float, estimated_ate: float) -> float:
    """Calculate bias between true and estimated ATE.

    Args:
        true_ate: True average treatment effect.
        estimated_ate: Estimated average treatment effect.

    Returns:
        Bias (estimated - true).
    """
    return estimated_ate - true_ate


# Refutation Metrics


def refutation_pass_rate(validation_results: list["ValidationResult"]) -> float:
    """Calculate pass rate for refutation tests.

    Filters validation results by refutation type and computes the
    proportion of tests that passed.

    Args:
        validation_results: List of ValidationResult objects.

    Returns:
        Pass rate as float between 0.0 and 1.0.
    """
    refutation_tests = [
        r for r in validation_results if r.validation_type.value == "refutation"
    ]
    if not refutation_tests:
        return 0.0
    passed_count = sum(1 for r in refutation_tests if r.passed)
    return passed_count / len(refutation_tests)


def refutation_summary(
    validation_results: list["ValidationResult"],
) -> dict[str, dict[str, Any]]:
    """Generate summary of refutation tests grouped by method.

    Args:
        validation_results: List of ValidationResult objects.

    Returns:
        Dictionary with method-level statistics:
            - method_name: {total, passed, failed, pass_rate}
    """
    refutation_tests = [
        r for r in validation_results if r.validation_type.value == "refutation"
    ]

    method_stats: dict[str, dict[str, Any]] = {}

    for test in refutation_tests:
        method = test.method
        if method not in method_stats:
            method_stats[method] = {"total": 0, "passed": 0, "failed": 0}

        method_stats[method]["total"] += 1
        if test.passed:
            method_stats[method]["passed"] += 1
        else:
            method_stats[method]["failed"] += 1

    # Calculate pass rate for each method
    for method, stats in method_stats.items():
        if stats["total"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total"]
        else:
            stats["pass_rate"] = 0.0

    return method_stats


# Completeness Metrics


def check_completeness(analysis: "Analysis") -> dict[str, Any]:
    """Verify completeness of analysis results.

    Checks analysis status and presence of all expected components.

    Args:
        analysis: Analysis model instance with relationships loaded.

    Returns:
        Dictionary with completeness information:
            - complete: bool - Whether analysis is fully complete
            - missing_components: list - Names of missing components
            - completion_percentage: float - Percentage of completion
    """
    missing_components = []

    # Check analysis status
    is_completed = analysis.status.value == "completed"
    if not is_completed:
        missing_components.append("analysis_status")

    # Check required relationships
    if not analysis.datasets:
        missing_components.append("datasets")
    if not analysis.causal_graphs:
        missing_components.append("causal_graphs")
    if not analysis.treatment_effects:
        missing_components.append("treatment_effects")
    if not analysis.validation_results:
        missing_components.append("validation_results")
    if not analysis.reports:
        missing_components.append("reports")

    # Check report types
    expected_report_types = {"executive", "technical", "visualization", "validation"}
    generated_types = {r.report_type.value for r in analysis.reports}
    missing_types = expected_report_types - generated_types
    for rt in missing_types:
        missing_components.append(f"report_type:{rt}")

    # Check report formats
    expected_formats = {"json", "html", "pdf", "markdown", "pptx"}
    generated_formats = {r.format.value for r in analysis.reports}
    missing_formats = expected_formats - generated_formats
    for fmt in missing_formats:
        missing_components.append(f"report_format:{fmt}")

    # Calculate completion percentage
    total_checks = 5 + len(expected_report_types) + len(expected_formats)  # 5 core + types + formats
    passed_checks = total_checks - len(missing_components)
    completion_percentage = (passed_checks / total_checks) * 100

    return {
        "complete": len(missing_components) == 0,
        "missing_components": missing_components,
        "completion_percentage": round(completion_percentage, 1),
    }


def check_stage_completion(analysis: "Analysis") -> dict[str, dict[str, Any]]:
    """Verify completion status of all analysis stages.

    Args:
        analysis: Analysis model instance with stages relationship loaded.

    Returns:
        Dictionary with stage-level completion status:
            - stage_name: {status, completed, started_at, completed_at}
    """
    expected_stages = {
        "download",
        "eda",
        "discovery",
        "treatment",
        "validation",
        "reporting",
    }

    stage_status: dict[str, dict[str, Any]] = {}

    for stage in analysis.stages:
        stage_name = stage.stage.value
        stage_status[stage_name] = {
            "status": stage.status.value,
            "completed": stage.status.value == "completed",
            "started_at": stage.started_at.isoformat() if stage.started_at else None,
            "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
            "progress_percent": stage.progress_percent,
        }

    # Mark missing stages
    found_stages = {s.stage.value for s in analysis.stages}
    for stage_name in expected_stages - found_stages:
        stage_status[stage_name] = {
            "status": "missing",
            "completed": False,
            "started_at": None,
            "completed_at": None,
            "progress_percent": 0,
        }

    return stage_status


# Confidence Interval Metrics


def ci_coverage(
    ci_lower: float, ci_upper: float, true_value: float
) -> dict[str, Any]:
    """Check if confidence interval covers the true value.

    Args:
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        true_value: The true value to check coverage for.

    Returns:
        Dictionary with coverage information:
            - covers: bool - Whether CI contains true value
            - width: float - Width of the confidence interval
            - lower: float - Lower bound
            - upper: float - Upper bound
    """
    covers = ci_lower <= true_value <= ci_upper
    width = ci_upper - ci_lower

    return {
        "covers": covers,
        "width": width,
        "lower": ci_lower,
        "upper": ci_upper,
    }


def ci_width_ratio(
    ci_lower: float, ci_upper: float, expected_range: tuple[float, float]
) -> float:
    """Compute ratio of CI width to expected range width.

    A ratio close to 1.0 indicates the CI width is similar to the
    expected range width.

    Args:
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        expected_range: Tuple of (min, max) expected values.

    Returns:
        Ratio of CI width to expected range width.
    """
    ci_width = ci_upper - ci_lower
    expected_width = expected_range[1] - expected_range[0]

    if expected_width == 0:
        return float("inf") if ci_width > 0 else 1.0

    return ci_width / expected_width


# P-value Metrics


def check_significance(p_value: float, alpha: float = 0.05) -> bool:
    """Check if p-value indicates statistical significance.

    Args:
        p_value: The p-value to check.
        alpha: Significance level (default 0.05).

    Returns:
        True if p-value < alpha (statistically significant).
    """
    return p_value < alpha


def multiple_testing_correction(
    p_values: list[float], method: str = "bonferroni"
) -> list[float]:
    """Apply multiple testing correction to p-values.

    Args:
        p_values: List of p-values to correct.
        method: Correction method ('bonferroni' or 'holm').

    Returns:
        List of corrected p-values.
    """
    n = len(p_values)
    if n == 0:
        return []

    if method == "bonferroni":
        # Bonferroni correction: multiply each p-value by n
        return [min(p * n, 1.0) for p in p_values]

    elif method == "holm":
        # Holm-Bonferroni step-down procedure
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [0.0] * n

        for rank, (original_idx, p) in enumerate(indexed):
            multiplier = n - rank
            corrected_p = min(p * multiplier, 1.0)
            corrected[original_idx] = corrected_p

        return corrected

    else:
        raise ValueError(f"Unknown correction method: {method}")


# Aggregate Metrics


def aggregate_validation_metrics(
    validation_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate validation metrics across multiple analyses.

    Args:
        validation_results: List of validation result dictionaries.

    Returns:
        Aggregated metrics including pass rates and averages.
    """
    if not validation_results:
        return {
            "total_analyses": 0,
            "overall_pass_rate": 0.0,
            "ate_pass_rate": 0.0,
            "graph_pass_rate": 0.0,
            "refutation_pass_rate": 0.0,
        }

    total = len(validation_results)

    # Count passes for each validation type
    overall_passed = sum(1 for r in validation_results if r.get("overall_passed", False))

    ate_results = [r.get("ate", {}) for r in validation_results]
    ate_passed = sum(1 for a in ate_results if a.get("passed", False))

    graph_results = [r.get("graph", {}) for r in validation_results]
    graph_passed = sum(1 for g in graph_results if g.get("passed", False))

    refutation_results = [r.get("refutation", {}) for r in validation_results]
    refutation_passed = sum(1 for ref in refutation_results if ref.get("passed", False))

    # Calculate average metrics
    f1_scores = [g.get("f1", 0.0) for g in graph_results if "f1" in g]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    refutation_rates = [ref.get("pass_rate", 0.0) for ref in refutation_results if "pass_rate" in ref]
    avg_refutation_rate = sum(refutation_rates) / len(refutation_rates) if refutation_rates else 0.0

    return {
        "total_analyses": total,
        "overall_pass_rate": overall_passed / total,
        "ate_pass_rate": ate_passed / total,
        "graph_pass_rate": graph_passed / total,
        "refutation_pass_rate": refutation_passed / total,
        "average_f1_score": avg_f1,
        "average_refutation_rate": avg_refutation_rate,
    }
