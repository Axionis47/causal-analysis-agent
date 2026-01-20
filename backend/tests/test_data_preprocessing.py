"""Unit tests for DataPreprocessor service."""

import numpy as np
import pandas as pd
import pytest

from app.services.data_preprocessing import DataPreprocessor, PreprocessingStep


class TestPreprocessingStep:
    """Tests for PreprocessingStep dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        step = PreprocessingStep(
            step_type="imputation",
            affected_columns=["col1", "col2"],
            parameters={"strategy": "mean"},
            statistics={"imputed_count": 10},
        )
        result = step.to_dict()

        assert result["step_type"] == "imputation"
        assert result["affected_columns"] == ["col1", "col2"]
        assert result["parameters"] == {"strategy": "mean"}
        assert result["statistics"] == {"imputed_count": 10}

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "step_type": "encoding",
            "affected_columns": ["category"],
            "parameters": {"method": "onehot"},
            "statistics": {"n_unique": 5},
        }
        step = PreprocessingStep.from_dict(data)

        assert step.step_type == "encoding"
        assert step.affected_columns == ["category"]
        assert step.parameters == {"method": "onehot"}
        assert step.statistics == {"n_unique": 5}


class TestDataPreprocessorImputation:
    """Tests for missing value imputation."""

    def test_impute_numeric_mean(self):
        """Test mean imputation for numeric columns."""
        df = pd.DataFrame({
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0],
            "other": [1, 2, 3, 4, 5],
        })
        columns_info = [
            {"name": "numeric", "inferred_type": "numeric", "missing_rate": 0.2},
            {"name": "other", "inferred_type": "numeric", "missing_rate": 0.0},
        ]

        preprocessor = DataPreprocessor(config={"enable_imputation": True})
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Check no missing values remain
        assert df_result["numeric"].isna().sum() == 0
        # Mean of [1, 2, 4, 5] is 3.0
        assert df_result["numeric"].iloc[2] == 3.0

        # Check step metadata
        imputation_steps = [s for s in steps if s.step_type == "imputation"]
        assert len(imputation_steps) >= 1

    def test_impute_categorical_mode(self):
        """Test mode imputation for categorical columns."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", np.nan, "A"],
        })
        columns_info = [
            {"name": "category", "inferred_type": "categorical", "missing_rate": 0.2},
        ]

        preprocessor = DataPreprocessor(config={"enable_imputation": True})
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Check no missing values remain
        assert df_result["category"].isna().sum() == 0
        # Mode is "A"
        assert df_result["category"].iloc[3] == "A"

    def test_skip_high_missing_columns(self):
        """Test that columns with >80% missing are skipped."""
        df = pd.DataFrame({
            "mostly_missing": [np.nan, np.nan, np.nan, np.nan, 1.0],
        })
        columns_info = [
            {"name": "mostly_missing", "inferred_type": "numeric", "missing_rate": 0.8},
        ]

        preprocessor = DataPreprocessor(config={"enable_imputation": True})
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # High missing columns should still have missing values
        assert df_result["mostly_missing"].isna().sum() == 4


class TestDataPreprocessorOutliers:
    """Tests for outlier handling."""

    def test_outlier_winsorization_iqr(self):
        """Test IQR-based outlier winsorization."""
        # Create data with outliers
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
        })
        columns_info = [
            {"name": "values", "inferred_type": "numeric"},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_outlier_handling": True,
            "outlier_method": "iqr",
            "outlier_threshold": 1.5,
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Outlier should be capped
        assert df_result["values"].max() < 100

        # Check step metadata
        outlier_steps = [s for s in steps if s.step_type == "outlier_handling"]
        assert len(outlier_steps) == 1

    def test_outlier_winsorization_zscore(self):
        """Test z-score based outlier winsorization."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 1000],  # 1000 is an extreme outlier
        })
        columns_info = [
            {"name": "values", "inferred_type": "numeric"},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_outlier_handling": True,
            "outlier_method": "zscore",
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Outlier should be capped
        assert df_result["values"].max() < 1000


class TestDataPreprocessorEncoding:
    """Tests for categorical encoding."""

    def test_onehot_encoding_low_cardinality(self):
        """Test one-hot encoding for low cardinality columns."""
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
        })
        columns_info = [
            {"name": "category", "inferred_type": "categorical"},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_encoding": True,
            "categorical_encoding_strategy": "onehot",
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Original column should be replaced with dummies
        assert "category" not in df_result.columns
        assert "category_B" in df_result.columns or "category_C" in df_result.columns

    def test_frequency_encoding_high_cardinality(self):
        """Test frequency encoding for high cardinality columns."""
        # Create high cardinality column
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(60)],
        })
        columns_info = [
            {"name": "category", "inferred_type": "categorical"},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_encoding": True,
            "categorical_encoding_strategy": "auto",
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Should use frequency encoding (values between 0 and 1)
        assert df_result["category"].min() >= 0
        assert df_result["category"].max() <= 1

    def test_ordinal_encoding_medium_cardinality(self):
        """Test ordinal encoding for medium cardinality columns."""
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(30)],
        })
        columns_info = [
            {"name": "category", "inferred_type": "categorical"},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_encoding": True,
            "categorical_encoding_strategy": "auto",
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Should be integers (ordinal encoding)
        assert df_result["category"].dtype in [np.int64, np.int32, int]


class TestDataPreprocessorScaling:
    """Tests for numeric scaling."""

    def test_standard_scaling(self):
        """Test StandardScaler is applied correctly."""
        df = pd.DataFrame({
            "values": [10, 20, 30, 40, 50],
        })
        columns_info = [
            {"name": "values", "inferred_type": "numeric"},
        ]

        preprocessor = DataPreprocessor(config={"enable_scaling": True})
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # After standard scaling, mean should be ~0 and std should be ~1
        assert abs(df_result["values"].mean()) < 0.01
        assert abs(df_result["values"].std() - 1.0) < 0.01

    def test_skip_binary_columns(self):
        """Test that binary columns are not scaled."""
        df = pd.DataFrame({
            "binary": [0, 1, 0, 1, 1],
            "numeric": [10, 20, 30, 40, 50],
        })
        columns_info = [
            {"name": "binary", "inferred_type": "numeric"},
            {"name": "numeric", "inferred_type": "numeric"},
        ]

        preprocessor = DataPreprocessor(config={"enable_scaling": True})
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Binary column should remain 0/1
        assert set(df_result["binary"].unique()).issubset({0, 1, 0.0, 1.0})
        # Numeric column should be scaled
        assert abs(df_result["numeric"].mean()) < 0.01


class TestDataPreprocessorFeatureEngineering:
    """Tests for feature engineering."""

    def test_interaction_terms(self):
        """Test interaction term creation."""
        df = pd.DataFrame({
            "treatment": [0, 1, 0, 1, 0],
            "confounder1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "outcome": [10, 20, 30, 40, 50],
        })
        columns_info = [
            {"name": "treatment", "inferred_type": "numeric"},
            {"name": "confounder1", "inferred_type": "numeric"},
            {"name": "outcome", "inferred_type": "numeric"},
        ]

        preprocessor = DataPreprocessor(config={"enable_feature_engineering": True})
        df_result, steps = preprocessor.preprocess(
            df,
            columns_info,
            [],
            treatment_candidates=["treatment"],
            outcome_candidates=["outcome"],
        )

        # Should have interaction term
        assert "fe_treatment_x_confounder1" in df_result.columns

    def test_polynomial_features(self):
        """Test polynomial feature creation."""
        df = pd.DataFrame({
            "treatment": [1, 2, 3, 4, 5],
            "outcome": [10, 20, 30, 40, 50],
        })
        columns_info = [
            {"name": "treatment", "inferred_type": "numeric"},
            {"name": "outcome", "inferred_type": "numeric"},
        ]

        preprocessor = DataPreprocessor(config={"enable_feature_engineering": True})
        df_result, steps = preprocessor.preprocess(
            df,
            columns_info,
            [],
            treatment_candidates=["treatment"],
            outcome_candidates=["outcome"],
        )

        # Should have squared terms
        assert "fe_treatment_squared" in df_result.columns
        assert "fe_outcome_squared" in df_result.columns

        # Verify squared values
        assert df_result["fe_treatment_squared"].iloc[0] == 1
        assert df_result["fe_treatment_squared"].iloc[4] == 25


class TestDataPreprocessorFullPipeline:
    """Tests for full preprocessing pipeline."""

    def test_full_pipeline(self):
        """Test running all preprocessing steps together."""
        df = pd.DataFrame({
            "numeric_with_missing": [1.0, np.nan, 3.0, 4.0, 5.0],
            "numeric_with_outlier": [1, 2, 3, 4, 100],
            "category": ["A", "B", "A", "B", "A"],
        })
        columns_info = [
            {"name": "numeric_with_missing", "inferred_type": "numeric", "missing_rate": 0.2},
            {"name": "numeric_with_outlier", "inferred_type": "numeric", "missing_rate": 0.0},
            {"name": "category", "inferred_type": "categorical", "missing_rate": 0.0},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_imputation": True,
            "enable_outlier_handling": True,
            "enable_encoding": True,
            "enable_scaling": True,
            "enable_feature_engineering": False,
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Verify preprocessing was applied
        assert df_result["numeric_with_missing"].isna().sum() == 0
        assert df_result["numeric_with_outlier"].max() < 100
        assert "category" not in df_result.columns  # Encoded

        # Check steps were tracked
        assert len(steps) >= 3

    def test_disabled_preprocessing(self):
        """Test that preprocessing can be completely disabled."""
        df = pd.DataFrame({
            "values": [1, np.nan, 3],
        })
        columns_info = [
            {"name": "values", "inferred_type": "numeric", "missing_rate": 0.33},
        ]

        preprocessor = DataPreprocessor(config={
            "enable_imputation": False,
            "enable_outlier_handling": False,
            "enable_encoding": False,
            "enable_scaling": False,
            "enable_feature_engineering": False,
        })
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Data should be unchanged
        assert df_result["values"].isna().sum() == 1
        assert len(steps) == 0

    def test_step_metadata_tracking(self):
        """Test that step metadata is properly tracked."""
        df = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0, np.nan, 5.0],
        })
        columns_info = [
            {"name": "numeric", "inferred_type": "numeric", "missing_rate": 0.4},
        ]

        preprocessor = DataPreprocessor(config={"enable_imputation": True})
        df_result, steps = preprocessor.preprocess(df, columns_info, [])

        # Find imputation step
        imputation_step = next(s for s in steps if s.step_type == "imputation")

        # Verify metadata
        assert "numeric" in imputation_step.affected_columns
        assert imputation_step.parameters["strategy"] == "mean"
        assert "imputed_values" in imputation_step.statistics


class TestEDAAgentIntegration:
    """Tests for EDA agent integration with preprocessing."""

    @pytest.mark.asyncio
    async def test_preprocessing_disabled_config(self):
        """Test that preprocessing is skipped when disabled in config."""
        # This is a placeholder for integration test
        # In a full test suite, this would mock the EDA agent and verify behavior
        pass

    @pytest.mark.asyncio
    async def test_preprocessed_dataset_saved(self):
        """Test that preprocessed dataset is saved correctly."""
        # This is a placeholder for integration test
        pass


class TestDiscoveryTreatmentIntegration:
    """Tests for discovery/treatment agent integration."""

    def test_load_preprocessed_data_fallback(self):
        """Test fallback to raw data when preprocessed not available."""
        # This is a placeholder for integration test
        pass
