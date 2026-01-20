"""Data quality validation utilities for pre-flight checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Validation thresholds (configurable via environment variables)
DATA_QUALITY_MIN_ROWS = settings.DATA_QUALITY_MIN_ROWS
DATA_QUALITY_WARN_ROWS = settings.DATA_QUALITY_WARN_ROWS
DATA_QUALITY_MAX_MISSING_PERCENT = settings.DATA_QUALITY_MAX_MISSING_PERCENT
DATA_QUALITY_WARN_MISSING_PERCENT = settings.DATA_QUALITY_WARN_MISSING_PERCENT
DATA_QUALITY_MIN_NUMERIC_COLUMNS = settings.DATA_QUALITY_MIN_NUMERIC_COLUMNS
DATA_QUALITY_WARN_NUMERIC_COLUMNS = settings.DATA_QUALITY_WARN_NUMERIC_COLUMNS
DATA_QUALITY_MAX_DUPLICATE_PERCENT = settings.DATA_QUALITY_MAX_DUPLICATE_PERCENT
DATA_QUALITY_MAX_IMBALANCE_RATIO = settings.DATA_QUALITY_MAX_IMBALANCE_RATIO


@dataclass
class ValidationResult:
    """
    Aggregated validation outcome for a dataset.

    Example warning structure:
        {
            "level": "warning|error",
            "message": "Human-readable description",
            "metric": "row_count|missing_values|numeric_columns|variance|duplicates|class_imbalance",
            "value": 50,
            "threshold": 100,
            "affected_columns": ["column_name"],
        }
    """

    passed: bool
    warnings: list[dict[str, Any]]
    can_proceed_with_override: bool


class DataQualityValidator:
    """Runs lightweight, user-facing data quality checks before full EDA."""

    async def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a dataframe against quality thresholds.

        Args:
            df: The dataframe to validate.

        Returns:
            ValidationResult with aggregated warnings and errors.
        """
        row_count, column_count = df.shape
        logger.info(
            "Data quality validation started",
            row_count=row_count,
            column_count=column_count,
        )

        checks = [
            ("row_count", self._check_row_count),
            ("missing_values", self._check_missing_values),
            ("numeric_columns", self._check_numeric_columns),
            ("variance", self._check_variance),
            ("duplicates", self._check_duplicates),
            ("class_imbalance", self._check_class_imbalance),
        ]

        warnings: list[dict[str, Any]] = []
        for metric, check in checks:
            issues = check(df)
            warnings.extend(issues)
            self._log_rule_result(metric, issues)

        error_count = sum(1 for issue in warnings if issue.get("level") == "error")
        warning_count = sum(1 for issue in warnings if issue.get("level") == "warning")
        passed = error_count == 0
        result = ValidationResult(
            passed=passed,
            warnings=warnings,
            can_proceed_with_override=passed,
        )

        logger.info(
            "Data quality validation completed",
            row_count=row_count,
            column_count=column_count,
            passed=passed,
            warning_count=warning_count,
            error_count=error_count,
        )
        return result

    def _check_row_count(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate minimum row count.

        Errors when rows < DATA_QUALITY_MIN_ROWS.
        Warns when rows < DATA_QUALITY_WARN_ROWS.
        """
        row_count = int(len(df))
        if row_count < DATA_QUALITY_MIN_ROWS:
            return [
                {
                    "level": "error",
                    "message": (
                        f"Dataset has {row_count} rows; minimum is {DATA_QUALITY_MIN_ROWS}."
                    ),
                    "metric": "row_count",
                    "value": row_count,
                    "threshold": DATA_QUALITY_MIN_ROWS,
                    "affected_columns": [],
                }
            ]
        if row_count < DATA_QUALITY_WARN_ROWS:
            return [
                {
                    "level": "warning",
                    "message": (
                        f"Dataset has {row_count} rows; recommended minimum is {DATA_QUALITY_WARN_ROWS}."
                    ),
                    "metric": "row_count",
                    "value": row_count,
                    "threshold": DATA_QUALITY_WARN_ROWS,
                    "affected_columns": [],
                }
            ]
        return []

    def _check_missing_values(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate missing value ratios per column.

        Errors when missing ratio > DATA_QUALITY_MAX_MISSING_PERCENT.
        Warns when missing ratio > DATA_QUALITY_WARN_MISSING_PERCENT (when below max).
        """
        issues: list[dict[str, Any]] = []
        total_rows = len(df)
        warn_threshold = DATA_QUALITY_WARN_MISSING_PERCENT
        use_warning = warn_threshold < DATA_QUALITY_MAX_MISSING_PERCENT
        for column in df.columns:
            series = df[column]
            missing_ratio = float(series.isna().mean()) if total_rows else 0.0
            if missing_ratio > DATA_QUALITY_MAX_MISSING_PERCENT:
                issues.append(
                    {
                        "level": "error",
                        "message": (
                            f"{column} has {missing_ratio:.0%} missing values; max is "
                            f"{DATA_QUALITY_MAX_MISSING_PERCENT:.0%}."
                        ),
                        "metric": "missing_values",
                        "value": missing_ratio,
                        "threshold": DATA_QUALITY_MAX_MISSING_PERCENT,
                        "affected_columns": [column],
                    }
                )
            elif use_warning and missing_ratio > warn_threshold:
                issues.append(
                    {
                        "level": "warning",
                        "message": (
                            f"{column} has {missing_ratio:.0%} missing values; warning at "
                            f"{warn_threshold:.0%}."
                        ),
                        "metric": "missing_values",
                        "value": missing_ratio,
                        "threshold": warn_threshold,
                        "affected_columns": [column],
                    }
                )
        return issues

    def _check_numeric_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate minimum numeric column counts.

        Errors when numeric columns < DATA_QUALITY_MIN_NUMERIC_COLUMNS.
        Warns when numeric columns < DATA_QUALITY_WARN_NUMERIC_COLUMNS.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_count = int(len(numeric_columns))
        if numeric_count < DATA_QUALITY_MIN_NUMERIC_COLUMNS:
            return [
                {
                    "level": "error",
                    "message": (
                        f"Dataset has {numeric_count} numeric columns; minimum is "
                        f"{DATA_QUALITY_MIN_NUMERIC_COLUMNS}."
                    ),
                    "metric": "numeric_columns",
                    "value": numeric_count,
                    "threshold": DATA_QUALITY_MIN_NUMERIC_COLUMNS,
                    "affected_columns": list(numeric_columns),
                }
            ]
        if numeric_count < DATA_QUALITY_WARN_NUMERIC_COLUMNS:
            return [
                {
                    "level": "warning",
                    "message": (
                        f"Dataset has {numeric_count} numeric columns; recommended minimum is "
                        f"{DATA_QUALITY_WARN_NUMERIC_COLUMNS}."
                    ),
                    "metric": "numeric_columns",
                    "value": numeric_count,
                    "threshold": DATA_QUALITY_WARN_NUMERIC_COLUMNS,
                    "affected_columns": list(numeric_columns),
                }
            ]
        return []

    def _check_variance(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Warn when numeric columns have zero variance.

        Uses variance == 0 as the threshold.
        """
        issues: list[dict[str, Any]] = []
        for column in df.select_dtypes(include=[np.number]).columns:
            series = df[column].dropna()
            if series.empty:
                continue
            if series.nunique() <= 1:
                issues.append(
                    {
                        "level": "warning",
                        "message": f"{column} has zero variance.",
                        "metric": "variance",
                        "value": 0.0,
                        "threshold": 0.0,
                        "affected_columns": [column],
                    }
                )
        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Warn when duplicate rows exceed DATA_QUALITY_MAX_DUPLICATE_PERCENT.
        """
        if df.empty:
            return []
        duplicate_ratio = float(df.duplicated().mean())
        if duplicate_ratio > DATA_QUALITY_MAX_DUPLICATE_PERCENT:
            return [
                {
                    "level": "warning",
                    "message": (
                        f"Duplicate rows are {duplicate_ratio:.0%}; warning threshold is "
                        f"{DATA_QUALITY_MAX_DUPLICATE_PERCENT:.0%}."
                    ),
                    "metric": "duplicates",
                    "value": duplicate_ratio,
                    "threshold": DATA_QUALITY_MAX_DUPLICATE_PERCENT,
                    "affected_columns": [],
                }
            ]
        return []

    def _check_class_imbalance(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Warn when binary columns exceed DATA_QUALITY_MAX_IMBALANCE_RATIO imbalance.
        """
        issues: list[dict[str, Any]] = []
        for column in df.columns:
            series = df[column].dropna()
            if series.nunique() != 2:
                continue
            counts = series.value_counts()
            total = float(counts.sum())
            if total == 0:
                continue
            imbalance_ratio = float(counts.max() / total)
            if imbalance_ratio > DATA_QUALITY_MAX_IMBALANCE_RATIO:
                issues.append(
                    {
                        "level": "warning",
                        "message": (
                            f"{column} is imbalanced at {imbalance_ratio:.0%}; warning threshold is "
                            f"{DATA_QUALITY_MAX_IMBALANCE_RATIO:.0%}."
                        ),
                        "metric": "class_imbalance",
                        "value": imbalance_ratio,
                        "threshold": DATA_QUALITY_MAX_IMBALANCE_RATIO,
                        "affected_columns": [column],
                    }
                )
        return issues

    @staticmethod
    def _log_rule_result(metric: str, issues: list[dict[str, Any]]) -> None:
        error_count = sum(1 for issue in issues if issue.get("level") == "error")
        warning_count = sum(1 for issue in issues if issue.get("level") == "warning")
        if error_count:
            status = "failed"
        elif warning_count:
            status = "warned"
        else:
            status = "passed"
        logger.info(
            "Data quality rule evaluated",
            metric=metric,
            status=status,
            error_count=error_count,
            warning_count=warning_count,
        )
