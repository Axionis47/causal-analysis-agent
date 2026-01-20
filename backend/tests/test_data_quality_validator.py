"""Unit tests for DataQualityValidator."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.data_quality import DataQualityValidator


def _make_numeric_df(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": range(rows),
            "b": range(rows, rows * 2),
            "c": range(rows * 2, rows * 3),
        }
    )


@pytest.mark.asyncio
async def test_validate_sufficient_rows():
    validator = DataQualityValidator()

    good_df = _make_numeric_df(1000)
    good_result = await validator.validate_dataframe(good_df)

    assert good_result.passed is True
    assert good_result.warnings == []

    bad_df = _make_numeric_df(50)
    bad_result = await validator.validate_dataframe(bad_df)

    assert bad_result.passed is False
    assert any(
        issue["metric"] == "row_count" and issue["level"] == "error"
        for issue in bad_result.warnings
    )


@pytest.mark.asyncio
async def test_validate_missing_values():
    validator = DataQualityValidator()
    df = _make_numeric_df(1000)
    df["missing_high"] = [None] * 600 + list(range(400))
    df["missing_higher"] = [None] * 850 + list(range(150))

    result = await validator.validate_dataframe(df)

    warning_metrics = {
        (issue["metric"], issue["level"], issue["affected_columns"][0])
        for issue in result.warnings
    }
    assert ("missing_values", "error", "missing_high") in warning_metrics
    assert ("missing_values", "error", "missing_higher") in warning_metrics


@pytest.mark.asyncio
async def test_validate_numeric_columns():
    validator = DataQualityValidator()

    error_df = pd.DataFrame(
        {
            "numeric": range(600),
            "category": ["a", "b", "c"] * 200,
        }
    )
    error_result = await validator.validate_dataframe(error_df)
    assert error_result.passed is False
    assert any(
        issue["metric"] == "numeric_columns" and issue["level"] == "error"
        for issue in error_result.warnings
    )

    warn_df = pd.DataFrame(
        {
            "numeric_one": range(600),
            "numeric_two": range(600, 1200),
            "category": ["a", "b", "c"] * 200,
        }
    )
    warn_result = await validator.validate_dataframe(warn_df)
    assert warn_result.passed is True
    assert any(
        issue["metric"] == "numeric_columns" and issue["level"] == "warning"
        for issue in warn_result.warnings
    )


@pytest.mark.asyncio
async def test_validate_zero_variance():
    validator = DataQualityValidator()
    df = _make_numeric_df(600)
    df["constant"] = 1

    result = await validator.validate_dataframe(df)

    assert any(
        issue["metric"] == "variance" and issue["level"] == "warning"
        for issue in result.warnings
    )


@pytest.mark.asyncio
async def test_validate_duplicates():
    validator = DataQualityValidator()
    base = _make_numeric_df(600)
    df = pd.concat([base, base.iloc[:400]], ignore_index=True)

    result = await validator.validate_dataframe(df)

    assert any(
        issue["metric"] == "duplicates" and issue["level"] == "warning"
        for issue in result.warnings
    )


@pytest.mark.asyncio
async def test_validate_class_imbalance():
    validator = DataQualityValidator()
    df = _make_numeric_df(1000)
    df["flag"] = [0] * 980 + [1] * 20

    result = await validator.validate_dataframe(df)

    assert any(
        issue["metric"] == "class_imbalance" and issue["level"] == "warning"
        for issue in result.warnings
    )


@pytest.mark.asyncio
async def test_validation_result_aggregation():
    validator = DataQualityValidator()
    df = _make_numeric_df(600)
    df["missing_warn"] = [None] * 360 + list(range(240))
    df = pd.concat([df, df.iloc[:300]], ignore_index=True)

    result = await validator.validate_dataframe(df)

    metrics = {issue["metric"] for issue in result.warnings}
    assert "missing_values" in metrics
    assert "duplicates" in metrics


@pytest.mark.asyncio
async def test_can_proceed_with_override():
    validator = DataQualityValidator()

    warn_df = _make_numeric_df(200)
    warn_result = await validator.validate_dataframe(warn_df)
    assert warn_result.can_proceed_with_override is True

    error_df = _make_numeric_df(50)
    error_result = await validator.validate_dataframe(error_df)
    assert error_result.can_proceed_with_override is False
