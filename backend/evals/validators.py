"""Database query validators for E2E evaluation of causal analysis results.

This module provides async validator classes that query real database results
using SQLAlchemy patterns and validate against ground truth expectations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.database import get_session_context
from app.models.analysis import Analysis, AnalysisStatus
from app.models.analysis_stage import AnalysisStage, StageStatus, StageType
from app.models.causal_graph import CausalGraph
from app.models.generated_report import GeneratedReport, ReportFormat, ReportType
from app.models.treatment_effect import TreatmentEffect
from app.models.validation_result import ValidationResult, ValidationType
from evals.ground_truth import GroundTruth
from evals.metrics import (
    check_significance,
    ci_coverage,
    ci_width_ratio,
    refutation_pass_rate,
    refutation_summary,
)

logger = logging.getLogger(__name__)


class ResultValidator:
    """Validator class for database query-based validation of analysis results.

    This class queries real database results using async SQLAlchemy patterns
    and validates against ground truth expectations from datasets.yaml.
    """

    def __init__(self, db: AsyncSession, ground_truth: GroundTruth) -> None:
        """Initialize validator with database session and ground truth.

        Args:
            db: Async SQLAlchemy session for database queries.
            ground_truth: GroundTruth instance for expected values.
        """
        self._db = db
        self._ground_truth = ground_truth

    async def _get_analysis_with_relationships(
        self, analysis_id: UUID
    ) -> Analysis | None:
        """Fetch analysis with all related entities loaded.

        Args:
            analysis_id: UUID of the analysis to fetch.

        Returns:
            Analysis instance with relationships loaded, or None if not found.
        """
        result = await self._db.execute(
            select(Analysis)
            .where(Analysis.id == analysis_id)
            .options(
                selectinload(Analysis.treatment_effects),
                selectinload(Analysis.causal_graphs),
                selectinload(Analysis.validation_results),
                selectinload(Analysis.stages),
                selectinload(Analysis.reports),
                selectinload(Analysis.datasets),
            )
        )
        return result.scalar_one_or_none()

    async def validate_ate_from_db(
        self, analysis_id: UUID, dataset_id: str
    ) -> dict[str, Any]:
        """Validate ATE estimate from database against ground truth.

        Queries TreatmentEffect table, extracts consensus or primary method ATE,
        and validates against the expected range from ground truth.

        Args:
            analysis_id: UUID of the analysis.
            dataset_id: Dataset identifier for ground truth lookup.

        Returns:
            Dictionary with validation results:
                - passed: Whether ATE is within tolerance
                - estimated: The estimated ATE value
                - expected_range: (min, max) expected range
                - bias: Difference from expected midpoint
                - method_used: Estimation method
                - error: Error message if validation failed
        """
        try:
            result = await self._db.execute(
                select(TreatmentEffect).where(
                    TreatmentEffect.analysis_id == analysis_id
                )
            )
            treatment_effects = result.scalars().all()

            if not treatment_effects:
                return {
                    "passed": False,
                    "error": "No treatment effects found for analysis",
                    "estimated": None,
                    "expected_range": None,
                    "bias": None,
                    "method_used": None,
                }

            # Get primary treatment effect (first one or consensus if available)
            primary_effect = treatment_effects[0]
            estimated_ate = primary_effect.ate
            method_used = primary_effect.method.value if primary_effect.method else None

            if estimated_ate is None:
                return {
                    "passed": False,
                    "error": "ATE value is None",
                    "estimated": None,
                    "expected_range": None,
                    "bias": None,
                    "method_used": method_used,
                }

            # Validate against ground truth
            validation_result = self._ground_truth.validate_ate(
                dataset_id, estimated_ate
            )

            return {
                "passed": validation_result["passed"],
                "estimated": estimated_ate,
                "expected_range": validation_result["expected_range"],
                "bias": validation_result["bias"],
                "tolerance": validation_result["tolerance"],
                "method_used": method_used,
            }

        except KeyError as e:
            return {
                "passed": False,
                "error": f"Dataset not found: {e}",
                "estimated": None,
                "expected_range": None,
                "bias": None,
                "method_used": None,
            }
        except ValueError as e:
            return {
                "passed": False,
                "error": str(e),
                "estimated": None,
                "expected_range": None,
                "bias": None,
                "method_used": None,
            }
        except Exception as e:
            logger.error(
                f"Error validating ATE for analysis {analysis_id}: {e}",
                exc_info=True,
            )
            return {
                "passed": False,
                "error": f"Validation error: {e}",
                "estimated": None,
                "expected_range": None,
                "bias": None,
                "method_used": None,
            }

    async def validate_graph_from_db(
        self, analysis_id: UUID, dataset_id: str
    ) -> dict[str, Any]:
        """Validate causal graph edges from database against ground truth.

        Queries CausalGraph table, extracts edges from JSONB, normalizes edge
        format, and computes precision/recall/F1 against expected edges.

        Args:
            analysis_id: UUID of the analysis.
            dataset_id: Dataset identifier for ground truth lookup.

        Returns:
            Dictionary with validation results:
                - passed: Whether F1 meets threshold
                - precision: Edge prediction precision
                - recall: Edge prediction recall
                - f1: F1 score
                - threshold: Minimum required F1
                - method_used: Discovery algorithm used
                - error: Error message if validation failed
        """
        try:
            result = await self._db.execute(
                select(CausalGraph).where(CausalGraph.analysis_id == analysis_id)
            )
            causal_graphs = result.scalars().all()

            if not causal_graphs:
                return {
                    "passed": False,
                    "error": "No causal graphs found for analysis",
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "threshold": None,
                    "method_used": None,
                }

            # Get primary graph (first one)
            primary_graph = causal_graphs[0]
            edges = primary_graph.edges or []
            method_used = primary_graph.method.value if primary_graph.method else None

            # Normalize edge format (handle dict with source/target)
            normalized_edges = []
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get("source", "")
                    target = edge.get("target", "")
                    normalized_edges.append({"source": source, "target": target})
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    normalized_edges.append({"source": edge[0], "target": edge[1]})

            # Validate against ground truth
            validation_result = self._ground_truth.validate_graph(
                dataset_id, normalized_edges
            )

            return {
                "passed": validation_result["passed"],
                "precision": validation_result["precision"],
                "recall": validation_result["recall"],
                "f1": validation_result["f1"],
                "threshold": validation_result["threshold"],
                "expected_edge_count": validation_result["expected_edge_count"],
                "predicted_edge_count": validation_result["predicted_edge_count"],
                "method_used": method_used,
            }

        except KeyError as e:
            return {
                "passed": False,
                "error": f"Dataset not found: {e}",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "threshold": None,
                "method_used": None,
            }
        except Exception as e:
            logger.error(
                f"Error validating graph for analysis {analysis_id}: {e}",
                exc_info=True,
            )
            return {
                "passed": False,
                "error": f"Validation error: {e}",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "threshold": None,
                "method_used": None,
            }

    async def validate_refutation_from_db(
        self, analysis_id: UUID, dataset_id: str
    ) -> dict[str, Any]:
        """Validate refutation test pass rate from database.

        Queries ValidationResult table, filters by refutation type, computes
        pass rate, and validates against the expected threshold.

        Args:
            analysis_id: UUID of the analysis.
            dataset_id: Dataset identifier for ground truth lookup.

        Returns:
            Dictionary with validation results:
                - passed: Whether pass rate meets threshold
                - pass_rate: Actual pass rate
                - threshold: Minimum required pass rate
                - tests_run: Total refutation tests
                - tests_passed: Number of passed tests
                - error: Error message if validation failed
        """
        try:
            result = await self._db.execute(
                select(ValidationResult).where(
                    ValidationResult.analysis_id == analysis_id,
                    ValidationResult.validation_type == ValidationType.REFUTATION,
                )
            )
            refutation_results = result.scalars().all()

            if not refutation_results:
                return {
                    "passed": False,
                    "error": "No refutation tests found for analysis",
                    "pass_rate": 0.0,
                    "threshold": None,
                    "tests_run": 0,
                    "tests_passed": 0,
                }

            # Compute pass rate
            tests_run = len(refutation_results)
            tests_passed = sum(1 for r in refutation_results if r.passed)
            pass_rate = tests_passed / tests_run if tests_run > 0 else 0.0

            # Validate against ground truth
            validation_result = self._ground_truth.validate_refutation(
                dataset_id, pass_rate
            )

            return {
                "passed": validation_result["passed"],
                "pass_rate": pass_rate,
                "threshold": validation_result["threshold"],
                "tests_run": tests_run,
                "tests_passed": tests_passed,
            }

        except KeyError as e:
            return {
                "passed": False,
                "error": f"Dataset not found: {e}",
                "pass_rate": 0.0,
                "threshold": None,
                "tests_run": 0,
                "tests_passed": 0,
            }
        except Exception as e:
            logger.error(
                f"Error validating refutation for analysis {analysis_id}: {e}",
                exc_info=True,
            )
            return {
                "passed": False,
                "error": f"Validation error: {e}",
                "pass_rate": 0.0,
                "threshold": None,
                "tests_run": 0,
                "tests_passed": 0,
            }

    async def validate_pvalues_from_db(
        self, analysis_id: UUID, alpha: float = 0.05
    ) -> dict[str, Any]:
        """Validate p-values from treatment effect estimates.

        Queries TreatmentEffect table, extracts p_value field, and checks
        statistical significance against the specified alpha level.

        Args:
            analysis_id: UUID of the analysis.
            alpha: Significance level (default 0.05).

        Returns:
            Dictionary with validation results:
                - passed: Whether p-value indicates significance
                - p_value: The p-value
                - threshold: Alpha level used
                - significant: Whether result is statistically significant
                - error: Error message if validation failed
        """
        try:
            result = await self._db.execute(
                select(TreatmentEffect).where(
                    TreatmentEffect.analysis_id == analysis_id
                )
            )
            treatment_effects = result.scalars().all()

            if not treatment_effects:
                return {
                    "passed": False,
                    "error": "No treatment effects found for analysis",
                    "p_value": None,
                    "threshold": alpha,
                    "significant": False,
                }

            # Get primary treatment effect
            primary_effect = treatment_effects[0]
            p_value = primary_effect.p_value

            if p_value is None:
                return {
                    "passed": False,
                    "error": "P-value is None",
                    "p_value": None,
                    "threshold": alpha,
                    "significant": False,
                }

            significant = check_significance(p_value, alpha)

            return {
                "passed": significant,
                "p_value": p_value,
                "threshold": alpha,
                "significant": significant,
            }

        except Exception as e:
            logger.error(
                f"Error validating p-values for analysis {analysis_id}: {e}",
                exc_info=True,
            )
            return {
                "passed": False,
                "error": f"Validation error: {e}",
                "p_value": None,
                "threshold": alpha,
                "significant": False,
            }

    async def validate_ci_coverage_from_db(
        self, analysis_id: UUID, dataset_id: str, coverage_threshold: float = 0.8
    ) -> dict[str, Any]:
        """Validate confidence interval coverage from database.

        Queries TreatmentEffect table, extracts CI bounds, and checks if the
        confidence interval overlaps with the expected ATE range.

        Args:
            analysis_id: UUID of the analysis.
            dataset_id: Dataset identifier for ground truth lookup.
            coverage_threshold: Minimum coverage ratio (default 0.8).

        Returns:
            Dictionary with validation results:
                - passed: Whether CI coverage is adequate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
                - expected_range: Expected ATE range
                - coverage: Coverage ratio
                - error: Error message if validation failed
        """
        try:
            result = await self._db.execute(
                select(TreatmentEffect).where(
                    TreatmentEffect.analysis_id == analysis_id
                )
            )
            treatment_effects = result.scalars().all()

            if not treatment_effects:
                return {
                    "passed": False,
                    "error": "No treatment effects found for analysis",
                    "ci_lower": None,
                    "ci_upper": None,
                    "expected_range": None,
                    "coverage": 0.0,
                }

            # Get primary treatment effect
            primary_effect = treatment_effects[0]
            ci_lower = primary_effect.ate_ci_lower
            ci_upper = primary_effect.ate_ci_upper

            if ci_lower is None or ci_upper is None:
                return {
                    "passed": False,
                    "error": "Confidence interval bounds are None",
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "expected_range": None,
                    "coverage": 0.0,
                }

            # Get expected ATE range from ground truth
            dataset = self._ground_truth.get_dataset(dataset_id)
            ground_truth_config = dataset.get("ground_truth", {})
            ate_range = ground_truth_config.get("ate_range")

            if ate_range is None:
                return {
                    "passed": False,
                    "error": "No ATE range defined in ground truth",
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "expected_range": None,
                    "coverage": 0.0,
                }

            expected_range = (ate_range["min"], ate_range["max"])
            expected_midpoint = (expected_range[0] + expected_range[1]) / 2

            # Check if CI contains the expected midpoint
            ci_result = ci_coverage(ci_lower, ci_upper, expected_midpoint)

            # Compute coverage ratio
            coverage = ci_width_ratio(ci_lower, ci_upper, expected_range)

            # Determine if passed: CI should cover expected value and have adequate width
            passed = ci_result["covers"] and coverage >= coverage_threshold

            return {
                "passed": passed,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "expected_range": expected_range,
                "expected_midpoint": expected_midpoint,
                "covers_expected": ci_result["covers"],
                "ci_width": ci_result["width"],
                "coverage": coverage,
                "coverage_threshold": coverage_threshold,
            }

        except KeyError as e:
            return {
                "passed": False,
                "error": f"Dataset not found: {e}",
                "ci_lower": None,
                "ci_upper": None,
                "expected_range": None,
                "coverage": 0.0,
            }
        except Exception as e:
            logger.error(
                f"Error validating CI coverage for analysis {analysis_id}: {e}",
                exc_info=True,
            )
            return {
                "passed": False,
                "error": f"Validation error: {e}",
                "ci_lower": None,
                "ci_upper": None,
                "expected_range": None,
                "coverage": 0.0,
            }

    async def check_completeness(self, analysis_id: UUID) -> dict[str, Any]:
        """Check completeness of analysis results.

        Verifies all expected stages completed, checks analysis status,
        and validates presence of required components.

        Args:
            analysis_id: UUID of the analysis.

        Returns:
            Dictionary with completeness results:
                - complete: Whether analysis is fully complete
                - missing_components: List of missing components
                - completion_percentage: Percentage of completion
                - status: Analysis status
                - stages: Stage-level completion status
                - reports: Report generation status
        """
        try:
            analysis = await self._get_analysis_with_relationships(analysis_id)

            if analysis is None:
                return {
                    "complete": False,
                    "error": "Analysis not found",
                    "missing_components": ["analysis"],
                    "completion_percentage": 0.0,
                }

            missing_components = []
            stage_status = {}
            report_status = {}

            # Check analysis status
            is_completed = analysis.status == AnalysisStatus.COMPLETED

            # Check stages
            expected_stages = {
                StageType.DOWNLOAD,
                StageType.EDA,
                StageType.DISCOVERY,
                StageType.TREATMENT,
                StageType.VALIDATION,
                StageType.REPORTING,
            }
            completed_stages = set()

            for stage in analysis.stages:
                stage_status[stage.stage.value] = {
                    "status": stage.status.value,
                    "completed": stage.status == StageStatus.COMPLETED,
                }
                if stage.status == StageStatus.COMPLETED:
                    completed_stages.add(stage.stage)

            missing_stages = expected_stages - completed_stages
            for stage in missing_stages:
                missing_components.append(f"stage:{stage.value}")

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

            # Check report types and formats
            expected_report_types = {
                ReportType.EXECUTIVE,
                ReportType.TECHNICAL,
                ReportType.VISUALIZATION,
                ReportType.VALIDATION,
            }
            expected_formats = {
                ReportFormat.JSON,
                ReportFormat.HTML,
                ReportFormat.PDF,
                ReportFormat.MARKDOWN,
                ReportFormat.PPTX,
            }

            generated_report_types = {r.report_type for r in analysis.reports}
            generated_formats = {r.format for r in analysis.reports}

            missing_report_types = expected_report_types - generated_report_types
            missing_formats = expected_formats - generated_formats

            for rt in missing_report_types:
                missing_components.append(f"report_type:{rt.value}")

            for fmt in missing_formats:
                missing_components.append(f"report_format:{fmt.value}")

            report_status = {
                "types_generated": [rt.value for rt in generated_report_types],
                "formats_generated": [f.value for f in generated_formats],
                "missing_types": [rt.value for rt in missing_report_types],
                "missing_formats": [f.value for f in missing_formats],
            }

            # Calculate completion percentage
            total_checks = (
                len(expected_stages)
                + len(expected_report_types)
                + len(expected_formats)
                + 5  # datasets, graphs, effects, validations, reports
            )
            passed_checks = total_checks - len(missing_components)
            completion_percentage = (passed_checks / total_checks) * 100

            complete = is_completed and len(missing_components) == 0

            return {
                "complete": complete,
                "missing_components": missing_components,
                "completion_percentage": round(completion_percentage, 1),
                "status": analysis.status.value,
                "stages": stage_status,
                "reports": report_status,
            }

        except Exception as e:
            logger.error(
                f"Error checking completeness for analysis {analysis_id}: {e}",
                exc_info=True,
            )
            return {
                "complete": False,
                "error": f"Completeness check error: {e}",
                "missing_components": [],
                "completion_percentage": 0.0,
            }

    async def validate_analysis_complete(
        self, analysis_id: UUID, dataset_id: str
    ) -> dict[str, Any]:
        """Run all validators and aggregate results.

        Comprehensive validation method that runs all individual validators
        and returns a complete validation report.

        Args:
            analysis_id: UUID of the analysis.
            dataset_id: Dataset identifier for ground truth lookup.

        Returns:
            Dictionary with comprehensive validation results including
            all individual validator results and overall pass status.
        """
        results: dict[str, Any] = {
            "analysis_id": str(analysis_id),
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Run all validators
        results["ate"] = await self.validate_ate_from_db(analysis_id, dataset_id)
        results["graph"] = await self.validate_graph_from_db(analysis_id, dataset_id)
        results["refutation"] = await self.validate_refutation_from_db(
            analysis_id, dataset_id
        )
        results["pvalue"] = await self.validate_pvalues_from_db(analysis_id)
        results["ci_coverage"] = await self.validate_ci_coverage_from_db(
            analysis_id, dataset_id
        )
        results["completeness"] = await self.check_completeness(analysis_id)

        # Compute overall pass status
        validations = [
            results["ate"],
            results["graph"],
            results["refutation"],
            results["pvalue"],
            results["ci_coverage"],
        ]

        # Only count validations that don't have errors
        valid_results = [v for v in validations if "error" not in v]
        if valid_results:
            results["overall_passed"] = all(v.get("passed", False) for v in valid_results)
        else:
            results["overall_passed"] = False

        # Count passes and failures
        results["summary"] = {
            "total_validators": len(validations),
            "passed": sum(1 for v in validations if v.get("passed", False)),
            "failed": sum(1 for v in validations if not v.get("passed", False)),
            "errors": sum(1 for v in validations if "error" in v),
        }

        return results


# Session context helper functions


async def validate_with_db(analysis_id: UUID, dataset_id: str) -> dict[str, Any]:
    """Validate analysis results using database session context.

    Convenience function that manages session lifecycle automatically.

    Args:
        analysis_id: UUID of the analysis.
        dataset_id: Dataset identifier for ground truth lookup.

    Returns:
        Comprehensive validation results dictionary.
    """
    async with get_session_context() as db:
        ground_truth = GroundTruth()
        validator = ResultValidator(db, ground_truth)
        return await validator.validate_analysis_complete(analysis_id, dataset_id)


async def validate_multiple_analyses(
    analysis_dataset_pairs: list[tuple[UUID, str]]
) -> list[dict[str, Any]]:
    """Validate multiple analyses in batch.

    Args:
        analysis_dataset_pairs: List of (analysis_id, dataset_id) tuples.

    Returns:
        List of validation results for each analysis.
    """
    results = []
    async with get_session_context() as db:
        ground_truth = GroundTruth()
        validator = ResultValidator(db, ground_truth)
        for analysis_id, dataset_id in analysis_dataset_pairs:
            result = await validator.validate_analysis_complete(analysis_id, dataset_id)
            results.append(result)
    return results


# Report generation functions


def generate_json_report(
    validation_results: dict[str, Any], output_path: Path
) -> Path:
    """Generate JSON report from validation results.

    Args:
        validation_results: Dictionary containing validation results.
        output_path: Path to write the JSON report.

    Returns:
        Path to the generated report file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "e2e_validation",
            "version": "1.0.0",
        },
        "results": validation_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return output_path


def generate_html_report(
    validation_results: dict[str, Any], output_path: Path
) -> Path:
    """Generate HTML report from validation results.

    Creates an HTML report with Bootstrap styling, color-coded pass/fail
    indicators, and summary statistics.

    Args:
        validation_results: Dictionary containing validation results.
        output_path: Path to write the HTML report.

    Returns:
        Path to the generated report file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine overall status
    overall_passed = validation_results.get("overall_passed", False)
    status_class = "success" if overall_passed else "danger"
    status_text = "PASS" if overall_passed else "FAIL"

    # Extract individual results
    ate = validation_results.get("ate", {})
    graph = validation_results.get("graph", {})
    refutation = validation_results.get("refutation", {})
    pvalue = validation_results.get("pvalue", {})
    ci_coverage = validation_results.get("ci_coverage", {})
    completeness = validation_results.get("completeness", {})
    summary = validation_results.get("summary", {})

    def status_badge(passed: bool | None, error: str | None = None) -> str:
        if error:
            return '<span class="badge bg-warning">ERROR</span>'
        if passed is None:
            return '<span class="badge bg-secondary">N/A</span>'
        if passed:
            return '<span class="badge bg-success">PASS</span>'
        return '<span class="badge bg-danger">FAIL</span>'

    def format_value(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (list, tuple)):
            return str(value)
        return str(value)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2E Validation Report - {validation_results.get('dataset_id', 'Unknown')}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; background-color: #f8f9fa; }}
        .card {{ margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-card {{ border-left: 5px solid; }}
        .summary-card.pass {{ border-left-color: #198754; }}
        .summary-card.fail {{ border-left-color: #dc3545; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; }}
        .metric-label {{ color: #6c757d; font-size: 0.875rem; }}
        .table th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">E2E Validation Report</h1>

        <!-- Summary Card -->
        <div class="card summary-card {'pass' if overall_passed else 'fail'}">
            <div class="card-body">
                <div class="row align-items-center">
                    <div class="col-md-4">
                        <h2 class="text-{status_class}">Overall: {status_text}</h2>
                        <p class="mb-0">Dataset: <strong>{validation_results.get('dataset_id', 'Unknown')}</strong></p>
                        <p class="mb-0">Analysis ID: <code>{validation_results.get('analysis_id', 'Unknown')}</code></p>
                    </div>
                    <div class="col-md-8">
                        <div class="row text-center">
                            <div class="col">
                                <div class="metric-value text-success">{summary.get('passed', 0)}</div>
                                <div class="metric-label">Passed</div>
                            </div>
                            <div class="col">
                                <div class="metric-value text-danger">{summary.get('failed', 0)}</div>
                                <div class="metric-label">Failed</div>
                            </div>
                            <div class="col">
                                <div class="metric-value text-warning">{summary.get('errors', 0)}</div>
                                <div class="metric-label">Errors</div>
                            </div>
                            <div class="col">
                                <div class="metric-value">{summary.get('total_validators', 0)}</div>
                                <div class="metric-label">Total</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ATE Validation -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">ATE Validation {status_badge(ate.get('passed'), ate.get('error'))}</h5>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <tr><th>Estimated ATE</th><td>{format_value(ate.get('estimated'))}</td></tr>
                    <tr><th>Expected Range</th><td>{format_value(ate.get('expected_range'))}</td></tr>
                    <tr><th>Bias</th><td>{format_value(ate.get('bias'))}</td></tr>
                    <tr><th>Tolerance</th><td>{format_value(ate.get('tolerance'))}</td></tr>
                    <tr><th>Method Used</th><td>{format_value(ate.get('method_used'))}</td></tr>
                    {f'<tr><th>Error</th><td class="text-danger">{ate.get("error")}</td></tr>' if ate.get('error') else ''}
                </table>
            </div>
        </div>

        <!-- Graph Validation -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Graph Validation {status_badge(graph.get('passed'), graph.get('error'))}</h5>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <tr><th>Precision</th><td>{format_value(graph.get('precision'))}</td></tr>
                    <tr><th>Recall</th><td>{format_value(graph.get('recall'))}</td></tr>
                    <tr><th>F1 Score</th><td>{format_value(graph.get('f1'))}</td></tr>
                    <tr><th>Threshold</th><td>{format_value(graph.get('threshold'))}</td></tr>
                    <tr><th>Expected Edges</th><td>{format_value(graph.get('expected_edge_count'))}</td></tr>
                    <tr><th>Predicted Edges</th><td>{format_value(graph.get('predicted_edge_count'))}</td></tr>
                    <tr><th>Method Used</th><td>{format_value(graph.get('method_used'))}</td></tr>
                    {f'<tr><th>Error</th><td class="text-danger">{graph.get("error")}</td></tr>' if graph.get('error') else ''}
                </table>
            </div>
        </div>

        <!-- Refutation Validation -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Refutation Validation {status_badge(refutation.get('passed'), refutation.get('error'))}</h5>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <tr><th>Pass Rate</th><td>{format_value(refutation.get('pass_rate'))}</td></tr>
                    <tr><th>Threshold</th><td>{format_value(refutation.get('threshold'))}</td></tr>
                    <tr><th>Tests Run</th><td>{format_value(refutation.get('tests_run'))}</td></tr>
                    <tr><th>Tests Passed</th><td>{format_value(refutation.get('tests_passed'))}</td></tr>
                    {f'<tr><th>Error</th><td class="text-danger">{refutation.get("error")}</td></tr>' if refutation.get('error') else ''}
                </table>
            </div>
        </div>

        <!-- P-value Validation -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">P-value Validation {status_badge(pvalue.get('passed'), pvalue.get('error'))}</h5>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <tr><th>P-value</th><td>{format_value(pvalue.get('p_value'))}</td></tr>
                    <tr><th>Alpha Level</th><td>{format_value(pvalue.get('threshold'))}</td></tr>
                    <tr><th>Significant</th><td>{'Yes' if pvalue.get('significant') else 'No'}</td></tr>
                    {f'<tr><th>Error</th><td class="text-danger">{pvalue.get("error")}</td></tr>' if pvalue.get('error') else ''}
                </table>
            </div>
        </div>

        <!-- CI Coverage Validation -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">CI Coverage Validation {status_badge(ci_coverage.get('passed'), ci_coverage.get('error'))}</h5>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <tr><th>CI Lower</th><td>{format_value(ci_coverage.get('ci_lower'))}</td></tr>
                    <tr><th>CI Upper</th><td>{format_value(ci_coverage.get('ci_upper'))}</td></tr>
                    <tr><th>Expected Range</th><td>{format_value(ci_coverage.get('expected_range'))}</td></tr>
                    <tr><th>Coverage</th><td>{format_value(ci_coverage.get('coverage'))}</td></tr>
                    <tr><th>Covers Expected</th><td>{'Yes' if ci_coverage.get('covers_expected') else 'No'}</td></tr>
                    {f'<tr><th>Error</th><td class="text-danger">{ci_coverage.get("error")}</td></tr>' if ci_coverage.get('error') else ''}
                </table>
            </div>
        </div>

        <!-- Completeness Check -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Completeness Check {status_badge(completeness.get('complete'), completeness.get('error'))}</h5>
            </div>
            <div class="card-body">
                <table class="table table-sm">
                    <tr><th>Analysis Status</th><td>{format_value(completeness.get('status'))}</td></tr>
                    <tr><th>Completion %</th><td>{format_value(completeness.get('completion_percentage'))}%</td></tr>
                    <tr><th>Missing Components</th><td>{', '.join(completeness.get('missing_components', [])) or 'None'}</td></tr>
                    {f'<tr><th>Error</th><td class="text-danger">{completeness.get("error")}</td></tr>' if completeness.get('error') else ''}
                </table>
            </div>
        </div>

        <footer class="text-muted text-center mt-4">
            <p>Generated at {validation_results.get('timestamp', datetime.now(timezone.utc).isoformat())}</p>
        </footer>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)

    return output_path


def aggregate_multi_dataset_report(
    results: list[dict[str, Any]], output_dir: Path
) -> Path:
    """Generate aggregated report across multiple datasets.

    Creates a dashboard HTML with overview table and individual dataset reports.

    Args:
        results: List of validation results for each dataset.
        output_dir: Directory to write reports.

    Returns:
        Path to the dashboard HTML file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual reports
    for result in results:
        dataset_id = result.get("dataset_id", "unknown")
        json_path = output_dir / "json" / f"{dataset_id}_validation.json"
        html_path = output_dir / "html" / f"{dataset_id}_validation.html"
        generate_json_report(result, json_path)
        generate_html_report(result, html_path)

    # Calculate aggregate statistics
    total = len(results)
    passed = sum(1 for r in results if r.get("overall_passed", False))
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Build dashboard HTML
    rows = ""
    for result in results:
        dataset_id = result.get("dataset_id", "unknown")
        overall = result.get("overall_passed", False)
        summary = result.get("summary", {})
        status_class = "success" if overall else "danger"
        status_text = "PASS" if overall else "FAIL"

        rows += f"""
        <tr>
            <td><a href="html/{dataset_id}_validation.html">{dataset_id}</a></td>
            <td><span class="badge bg-{status_class}">{status_text}</span></td>
            <td>{summary.get('passed', 0)}</td>
            <td>{summary.get('failed', 0)}</td>
            <td>{summary.get('errors', 0)}</td>
            <td><a href="json/{dataset_id}_validation.json" class="btn btn-sm btn-outline-primary">JSON</a></td>
        </tr>"""

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2E Validation Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; background-color: #f8f9fa; }}
        .summary-box {{ padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-box.success {{ background-color: #d1e7dd; }}
        .summary-box.danger {{ background-color: #f8d7da; }}
        .metric-value {{ font-size: 2rem; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">E2E Validation Dashboard</h1>

        <!-- Summary -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="summary-box {'success' if pass_rate >= 80 else 'danger'}">
                    <div class="metric-value">{pass_rate:.1f}%</div>
                    <div>Pass Rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="summary-box success">
                    <div class="metric-value text-success">{passed}</div>
                    <div>Passed</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="summary-box danger">
                    <div class="metric-value text-danger">{failed}</div>
                    <div>Failed</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="summary-box" style="background-color: #e2e3e5;">
                    <div class="metric-value">{total}</div>
                    <div>Total Datasets</div>
                </div>
            </div>
        </div>

        <!-- Results Table -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Dataset Validation Results</h5>
            </div>
            <div class="card-body">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Status</th>
                            <th>Passed</th>
                            <th>Failed</th>
                            <th>Errors</th>
                            <th>Report</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        </div>

        <footer class="text-muted text-center mt-4">
            <p>Generated at {datetime.now(timezone.utc).isoformat()}</p>
        </footer>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    dashboard_path = output_dir / "dashboard.html"
    with open(dashboard_path, "w") as f:
        f.write(dashboard_html)

    return dashboard_path


# Testing utilities


async def create_test_analysis(
    db: AsyncSession, dataset_id: str, kaggle_url: str = "https://example.com"
) -> Analysis:
    """Create minimal Analysis for testing.

    Args:
        db: Async database session.
        dataset_id: Dataset identifier.
        kaggle_url: Kaggle URL for the analysis.

    Returns:
        Created Analysis instance.
    """
    analysis = Analysis(
        kaggle_url=kaggle_url,
        status=AnalysisStatus.COMPLETED,
        config={"dataset_id": dataset_id},
    )
    db.add(analysis)
    await db.flush()
    return analysis


async def populate_test_results(
    db: AsyncSession,
    analysis_id: UUID,
    dataset_id: str,
    ground_truth: GroundTruth,
) -> None:
    """Populate analysis with test results matching ground truth.

    Args:
        db: Async database session.
        analysis_id: UUID of the analysis.
        dataset_id: Dataset identifier.
        ground_truth: GroundTruth instance for expected values.
    """
    dataset = ground_truth.get_dataset(dataset_id)
    gt_config = dataset.get("ground_truth", {})
    causal_structure = dataset.get("causal_structure", {})

    # Add treatment effect with ATE in expected range
    ate_range = gt_config.get("ate_range", {})
    expected_ate = (ate_range.get("min", 0) + ate_range.get("max", 0)) / 2

    treatment_effect = TreatmentEffect(
        analysis_id=analysis_id,
        treatment_variable=causal_structure.get("treatment", "treatment"),
        outcome_variable=causal_structure.get("outcome", "outcome"),
        method="propensity_matching",
        ate=expected_ate,
        ate_ci_lower=expected_ate - 0.1,
        ate_ci_upper=expected_ate + 0.1,
        p_value=0.01,
    )
    db.add(treatment_effect)

    # Add causal graph with expected edges
    expected_edges = causal_structure.get("expected_edges", [])
    causal_graph = CausalGraph(
        analysis_id=analysis_id,
        method="pc",
        edges=expected_edges,
        nodes=[],
    )
    db.add(causal_graph)

    # Add refutation validation results
    for method in ["placebo_treatment", "random_cause", "data_subset"]:
        validation = ValidationResult(
            analysis_id=analysis_id,
            validation_type=ValidationType.REFUTATION,
            method=method,
            passed=True,
            confidence_score=0.95,
        )
        db.add(validation)

    await db.flush()


def assert_validation_passed(validation_result: dict[str, Any]) -> None:
    """Assert that a validation result passed.

    Args:
        validation_result: Validation result dictionary.

    Raises:
        AssertionError: If validation did not pass.
    """
    assert validation_result.get("passed", False), (
        f"Validation failed: {validation_result.get('error', 'Unknown error')}"
    )


def print_validation_summary(validation_result: dict[str, Any]) -> None:
    """Pretty-print validation results for debugging.

    Args:
        validation_result: Validation result dictionary.
    """
    print("\n" + "=" * 60)
    print(f"Dataset: {validation_result.get('dataset_id', 'Unknown')}")
    print(f"Analysis ID: {validation_result.get('analysis_id', 'Unknown')}")
    print(f"Overall: {'PASS' if validation_result.get('overall_passed') else 'FAIL'}")
    print("=" * 60)

    for key in ["ate", "graph", "refutation", "pvalue", "ci_coverage", "completeness"]:
        result = validation_result.get(key, {})
        status = "PASS" if result.get("passed") else "FAIL"
        if "error" in result:
            status = f"ERROR: {result['error']}"
        print(f"  {key}: {status}")

    summary = validation_result.get("summary", {})
    print("-" * 60)
    print(
        f"Summary: {summary.get('passed', 0)}/{summary.get('total_validators', 0)} passed"
    )
    print("=" * 60 + "\n")
