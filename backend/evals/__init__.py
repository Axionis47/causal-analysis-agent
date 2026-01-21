"""Evaluation utilities for causal analysis E2E validation.

This package provides:
- Ground truth validation against datasets.yaml configurations
- Database query validators for real analysis results
- Metrics for ATE, graph edges, refutation tests, and completeness
- Report generation (JSON and HTML)
"""

from evals.ground_truth import GroundTruth, load_datasets_yaml
from evals.metrics import (
    aggregate_validation_metrics,
    ate_bias,
    check_completeness,
    check_significance,
    check_stage_completion,
    ci_coverage,
    ci_width_ratio,
    multiple_testing_correction,
    precision_recall,
    refutation_pass_rate,
    refutation_summary,
)
from evals.validators import (
    ResultValidator,
    aggregate_multi_dataset_report,
    assert_validation_passed,
    create_test_analysis,
    generate_html_report,
    generate_json_report,
    populate_test_results,
    print_validation_summary,
    validate_multiple_analyses,
    validate_with_db,
)

__all__ = [
    # Ground truth
    "GroundTruth",
    "load_datasets_yaml",
    # Metrics - core
    "ate_bias",
    "precision_recall",
    # Metrics - refutation
    "refutation_pass_rate",
    "refutation_summary",
    # Metrics - completeness
    "check_completeness",
    "check_stage_completion",
    # Metrics - confidence interval
    "ci_coverage",
    "ci_width_ratio",
    # Metrics - p-value
    "check_significance",
    "multiple_testing_correction",
    # Metrics - aggregate
    "aggregate_validation_metrics",
    # Validators
    "ResultValidator",
    "validate_with_db",
    "validate_multiple_analyses",
    # Report generation
    "generate_json_report",
    "generate_html_report",
    "aggregate_multi_dataset_report",
    # Testing utilities
    "create_test_analysis",
    "populate_test_results",
    "assert_validation_passed",
    "print_validation_summary",
]
