"""Data preprocessing service for systematic transformations after EDA analysis."""

from dataclasses import dataclass, field, asdict
from typing import Any, Literal
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStep:
    """Track each preprocessing operation."""

    step_type: Literal[
        "imputation", "encoding", "scaling", "outlier_handling", "feature_engineering"
    ]
    affected_columns: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingStep":
        """Deserialize from dictionary."""
        return cls(**data)


class DataPreprocessor:
    """Centralized data preprocessing service."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with configuration.

        Args:
            config: Configuration dict with keys:
                - enable_imputation: bool (default True)
                - enable_outlier_handling: bool (default True)
                - enable_encoding: bool (default True)
                - enable_scaling: bool (default True)
                - enable_feature_engineering: bool (default False)
                - outlier_method: "iqr" | "zscore" (default "iqr")
                - outlier_threshold: float (default 1.5)
                - categorical_encoding_strategy: "auto" | "onehot" | "target" | "frequency" (default "auto")
        """
        config = config or {}
        self.config = {
            "enable_imputation": config.get("enable_imputation", True),
            "enable_outlier_handling": config.get("enable_outlier_handling", True),
            "enable_encoding": config.get("enable_encoding", True),
            "enable_scaling": config.get("enable_scaling", True),
            "enable_feature_engineering": config.get("enable_feature_engineering", False),
            "outlier_method": config.get("outlier_method", "iqr"),
            "outlier_threshold": config.get("outlier_threshold", 1.5),
            "categorical_encoding_strategy": config.get(
                "categorical_encoding_strategy", "auto"
            ),
        }

    def preprocess(
        self,
        df: pd.DataFrame,
        columns_info: list[dict],
        quality_issues: list[dict],
        treatment_candidates: list[str] | None = None,
        outcome_candidates: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[PreprocessingStep]]:
        """Apply preprocessing transformations systematically.

        Args:
            df: Input DataFrame
            columns_info: Column metadata from EDA analysis
            quality_issues: Quality issues detected during EDA
            treatment_candidates: Treatment variable candidates for feature engineering
            outcome_candidates: Outcome variable candidates for feature engineering

        Returns:
            Tuple of (preprocessed DataFrame, list of PreprocessingStep objects)
        """
        df_processed = df.copy()
        steps: list[PreprocessingStep] = []

        # Build column type lookup
        columns_by_type = self._categorize_columns(columns_info)

        # Apply preprocessing steps in order
        if self.config["enable_imputation"]:
            df_processed, imputation_steps = self._impute_missing(
                df_processed, columns_info, columns_by_type
            )
            steps.extend(imputation_steps)

        if self.config["enable_outlier_handling"]:
            df_processed, outlier_steps = self._handle_outliers(
                df_processed,
                columns_by_type,
                method=self.config["outlier_method"],
                threshold=self.config["outlier_threshold"],
            )
            steps.extend(outlier_steps)

        if self.config["enable_encoding"]:
            df_processed, encoding_steps = self._encode_categorical(
                df_processed, columns_info, columns_by_type, outcome_candidates
            )
            steps.extend(encoding_steps)

        if self.config["enable_feature_engineering"]:
            df_processed, fe_steps = self._engineer_features(
                df_processed,
                columns_info,
                columns_by_type,
                treatment_candidates or [],
                outcome_candidates or [],
            )
            steps.extend(fe_steps)

        if self.config["enable_scaling"]:
            df_processed, scaling_steps = self._scale_numeric(
                df_processed, columns_by_type
            )
            steps.extend(scaling_steps)

        logger.info(
            f"Preprocessing complete: {len(steps)} steps applied, "
            f"shape changed from {df.shape} to {df_processed.shape}"
        )

        return df_processed, steps

    def _categorize_columns(
        self, columns_info: list[dict]
    ) -> dict[str, list[str]]:
        """Categorize columns by type.

        Maps EDA dtype labels to preprocessing categories:
        - numeric: "numerical", "binary", "numeric", "int64", "float64", "int", "float"
        - categorical: "categorical", "object", "category", "bool"
        - datetime: "datetime", "datetime64[ns]"
        - text: "text"

        Note: EDAAgent._columns_info() uses _dtype_label() which returns:
        - "binary" for boolean columns
        - "numerical" for numeric columns
        - "datetime" for datetime columns
        - "categorical" for low-cardinality object columns (<=20 unique)
        - "text" for high-cardinality object columns (>20 unique)
        """
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        text_cols = []

        for col_info in columns_info:
            col_name = col_info.get("name", col_info.get("column_name", ""))
            col_type = col_info.get("inferred_type", col_info.get("dtype", ""))

            # Include EDA labels "numerical" and "binary" as numeric types
            if col_type in ("numeric", "numerical", "binary", "int64", "float64", "int", "float"):
                numeric_cols.append(col_name)
            elif col_type in ("categorical", "object", "category", "bool"):
                categorical_cols.append(col_name)
            elif col_type in ("datetime", "datetime64[ns]"):
                datetime_cols.append(col_name)
            elif col_type in ("text",):
                text_cols.append(col_name)

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "text": text_cols,
        }

    def _impute_missing(
        self,
        df: pd.DataFrame,
        columns_info: list[dict],
        columns_by_type: dict[str, list[str]],
    ) -> tuple[pd.DataFrame, list[PreprocessingStep]]:
        """Impute missing values.

        For numeric columns: use mean imputation
        For categorical columns: use mode imputation
        Skip columns with >80% missing values

        Note: EDAAgent._columns_info() returns missing_percent as a fraction (0-1).
        This method reads missing_percent from columns_info and uses it for the
        >80% skip condition.
        """
        steps: list[PreprocessingStep] = []

        # Build missing rate lookup from EDA's columns_info
        # EDA provides "missing_percent" as a fraction (0-1), e.g., 0.85 for 85%
        missing_rates = {}
        for col_info in columns_info:
            col_name = col_info.get("name", col_info.get("column_name", ""))
            # Read missing_percent from EDA (primary), fall back to other keys
            missing_rate = col_info.get(
                "missing_percent",
                col_info.get("missing_rate", col_info.get("null_percentage", 0))
            )
            # Normalize: if it's a string percentage like "85%", convert to fraction
            if isinstance(missing_rate, str):
                missing_rate = float(missing_rate.rstrip("%")) / 100
            # If it's > 1, assume it's a percentage (e.g., 85) and convert to fraction
            elif isinstance(missing_rate, (int, float)) and missing_rate > 1:
                missing_rate = missing_rate / 100
            missing_rates[col_name] = float(missing_rate)

        # Impute numeric columns
        numeric_to_impute = [
            col
            for col in columns_by_type["numeric"]
            if col in df.columns
            and df[col].isna().any()
            and missing_rates.get(col, 0) <= 0.8
        ]

        if numeric_to_impute:
            imputer = SimpleImputer(strategy="mean")
            imputed_values = {}

            for col in numeric_to_impute:
                original_missing = df[col].isna().sum()
                col_mean = df[col].mean()
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                imputed_values[col] = {
                    "imputed_count": int(original_missing),
                    "imputed_value": float(col_mean) if pd.notna(col_mean) else 0.0,
                }

            steps.append(
                PreprocessingStep(
                    step_type="imputation",
                    affected_columns=numeric_to_impute,
                    parameters={"strategy": "mean", "column_type": "numeric"},
                    statistics={"imputed_values": imputed_values},
                )
            )

        # Impute categorical columns
        categorical_to_impute = [
            col
            for col in columns_by_type["categorical"]
            if col in df.columns
            and df[col].isna().any()
            and missing_rates.get(col, 0) <= 0.8
        ]

        if categorical_to_impute:
            imputer = SimpleImputer(strategy="most_frequent")
            imputed_values = {}

            for col in categorical_to_impute:
                original_missing = df[col].isna().sum()
                col_mode = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                imputed_values[col] = {
                    "imputed_count": int(original_missing),
                    "imputed_value": str(col_mode),
                }

            steps.append(
                PreprocessingStep(
                    step_type="imputation",
                    affected_columns=categorical_to_impute,
                    parameters={"strategy": "most_frequent", "column_type": "categorical"},
                    statistics={"imputed_values": imputed_values},
                )
            )

        # Log warnings for skipped columns
        skipped_cols = [
            col
            for col in df.columns
            if missing_rates.get(col, 0) > 0.8 and df[col].isna().any()
        ]
        if skipped_cols:
            logger.warning(
                f"Skipped imputation for columns with >80% missing: {skipped_cols}"
            )

        return df, steps

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        columns_by_type: dict[str, list[str]],
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> tuple[pd.DataFrame, list[PreprocessingStep]]:
        """Handle outliers in numeric columns.

        IQR method: cap at Q1 - threshold*IQR and Q3 + threshold*IQR
        Z-score method: cap at values where |z| > 3
        """
        steps: list[PreprocessingStep] = []
        numeric_cols = [
            col for col in columns_by_type["numeric"] if col in df.columns
        ]

        if not numeric_cols:
            return df, steps

        outlier_stats = {}

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            if method == "iqr":
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            else:  # zscore
                mean = col_data.mean()
                std = col_data.std()
                if std == 0:
                    continue
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std

            # Count outliers before capping
            outliers_below = (df[col] < lower_bound).sum()
            outliers_above = (df[col] > upper_bound).sum()
            total_outliers = outliers_below + outliers_above

            if total_outliers > 0:
                # Apply winsorization
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                outlier_stats[col] = {
                    "outliers_detected": int(total_outliers),
                    "outliers_below": int(outliers_below),
                    "outliers_above": int(outliers_above),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }

        if outlier_stats:
            steps.append(
                PreprocessingStep(
                    step_type="outlier_handling",
                    affected_columns=list(outlier_stats.keys()),
                    parameters={
                        "method": method,
                        "threshold": threshold,
                        "action": "winsorization",
                    },
                    statistics={"outlier_stats": outlier_stats},
                )
            )

        return df, steps

    def _encode_categorical(
        self,
        df: pd.DataFrame,
        columns_info: list[dict],
        columns_by_type: dict[str, list[str]],
        outcome_candidates: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[PreprocessingStep]]:
        """Encode categorical and text columns.

        Encoding strategies by cardinality:
        - Low cardinality (<=10): one-hot encoding
        - Medium cardinality (11-50): ordinal encoding or target encoding (if outcome available)
        - High cardinality (>50): frequency encoding or target encoding (if outcome available)

        For "auto" strategy:
        - Uses target encoding when an outcome variable is available and the column
          has medium-to-high cardinality (>10 unique values)
        - Falls back to frequency/ordinal encoding when no outcome is available

        Text columns (high-cardinality object columns identified by EDA) are also
        encoded using frequency or target encoding.
        """
        steps: list[PreprocessingStep] = []
        outcome_candidates = outcome_candidates or []

        # Include both categorical and text columns for encoding
        categorical_cols = [
            col for col in columns_by_type["categorical"] if col in df.columns
        ]
        text_cols = [
            col for col in columns_by_type.get("text", []) if col in df.columns
        ]
        all_cols_to_encode = categorical_cols + text_cols

        if not all_cols_to_encode:
            return df, steps

        strategy = self.config["categorical_encoding_strategy"]
        encoding_stats = {}

        # Determine outcome column for target encoding
        outcome_col = None
        if outcome_candidates:
            for candidate in outcome_candidates:
                if candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
                    outcome_col = candidate
                    break

        for col in all_cols_to_encode:
            # Skip encoding outcome columns themselves
            if col in outcome_candidates:
                continue

            n_unique = df[col].nunique()
            is_text_col = col in text_cols

            if strategy == "onehot" and not is_text_col:
                # One-hot encoding (not for text columns due to high cardinality)
                if n_unique <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    encoding_stats[col] = {
                        "method": "onehot",
                        "n_unique": int(n_unique),
                        "new_columns": list(dummies.columns),
                    }
                else:
                    # Fall back to frequency for high cardinality even when onehot requested
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0)
                    encoding_stats[col] = {
                        "method": "frequency",
                        "n_unique": int(n_unique),
                        "mapping": {str(k): float(v) for k, v in freq_map.items()},
                    }

            elif strategy == "target" or (strategy == "auto" and outcome_col and n_unique > 10):
                # Target encoding: replace categories with mean of outcome
                if outcome_col and outcome_col in df.columns:
                    target_means = df.groupby(col)[outcome_col].mean()
                    global_mean = df[outcome_col].mean()
                    # Apply smoothing to avoid overfitting on rare categories
                    # smoothing factor based on category frequency
                    category_counts = df[col].value_counts()
                    smoothing_weight = 10  # minimum samples for full weight
                    smoothed_means = {}
                    for cat, mean in target_means.items():
                        count = category_counts.get(cat, 0)
                        # Weighted average: (count * category_mean + smoothing_weight * global_mean) / (count + smoothing_weight)
                        smoothed_mean = (count * mean + smoothing_weight * global_mean) / (count + smoothing_weight)
                        smoothed_means[cat] = smoothed_mean
                    df[col] = df[col].map(smoothed_means).fillna(global_mean)
                    encoding_stats[col] = {
                        "method": "target",
                        "n_unique": int(n_unique),
                        "outcome_column": outcome_col,
                        "global_mean": float(global_mean),
                        "mapping": {str(k): float(v) for k, v in smoothed_means.items()},
                    }
                else:
                    # No outcome available, fall back to frequency encoding
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[col] = df[col].map(freq_map).fillna(0)
                    encoding_stats[col] = {
                        "method": "frequency",
                        "n_unique": int(n_unique),
                        "mapping": {str(k): float(v) for k, v in freq_map.items()},
                    }

            elif strategy == "frequency" or (strategy == "auto" and n_unique > 50) or is_text_col:
                # Frequency encoding for high cardinality or text columns
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map).fillna(0)
                encoding_stats[col] = {
                    "method": "frequency",
                    "n_unique": int(n_unique),
                    "mapping": {str(k): float(v) for k, v in freq_map.items()},
                }

            elif strategy == "auto" and n_unique <= 10 and not is_text_col:
                # One-hot encoding for low cardinality categorical columns
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoding_stats[col] = {
                    "method": "onehot",
                    "n_unique": int(n_unique),
                    "new_columns": list(dummies.columns),
                }

            else:  # ordinal encoding for medium cardinality (11-50) when no outcome
                # Ordinal encoding based on frequency
                categories = df[col].value_counts().index.tolist()
                mapping = {cat: i for i, cat in enumerate(categories)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
                encoding_stats[col] = {
                    "method": "ordinal",
                    "n_unique": int(n_unique),
                    "mapping": {str(k): int(v) for k, v in mapping.items()},
                }

        if encoding_stats:
            steps.append(
                PreprocessingStep(
                    step_type="encoding",
                    affected_columns=list(encoding_stats.keys()),
                    parameters={
                        "strategy": strategy,
                        "outcome_column_used": outcome_col,
                    },
                    statistics={"encoding_stats": encoding_stats},
                )
            )

        return df, steps

    def _engineer_features(
        self,
        df: pd.DataFrame,
        columns_info: list[dict],
        columns_by_type: dict[str, list[str]],
        treatment_candidates: list[str],
        outcome_candidates: list[str],
    ) -> tuple[pd.DataFrame, list[PreprocessingStep]]:
        """Engineer new features.

        - Interaction terms: treatment * top confounders
        - Polynomial features: squared terms for treatment and outcome
        - Binned features: quartile bins for highly skewed continuous variables
        """
        steps: list[PreprocessingStep] = []
        fe_stats = {"interaction_terms": [], "polynomial_features": [], "binned_features": []}

        numeric_cols = [
            col for col in columns_by_type["numeric"] if col in df.columns
        ]

        # Identify potential confounders (numeric columns that aren't treatment/outcome)
        confounders = [
            col
            for col in numeric_cols
            if col not in treatment_candidates and col not in outcome_candidates
        ][:3]  # Top 3 confounders

        # Create interaction terms
        for treatment in treatment_candidates:
            if treatment not in df.columns:
                continue
            for confounder in confounders:
                if confounder not in df.columns:
                    continue
                interaction_col = f"fe_{treatment}_x_{confounder}"
                df[interaction_col] = df[treatment] * df[confounder]
                fe_stats["interaction_terms"].append(interaction_col)

        # Create polynomial features (squared terms)
        for col in treatment_candidates + outcome_candidates:
            if col not in df.columns or col not in numeric_cols:
                continue
            squared_col = f"fe_{col}_squared"
            df[squared_col] = df[col] ** 2
            fe_stats["polynomial_features"].append(squared_col)

        # Create binned features for highly skewed columns
        for col in numeric_cols:
            if col in treatment_candidates or col in outcome_candidates:
                continue
            col_data = df[col].dropna()
            if len(col_data) < 4:
                continue
            skewness = col_data.skew()
            if abs(skewness) > 1.0:  # Highly skewed
                binned_col = f"fe_{col}_binned"
                df[binned_col] = pd.qcut(
                    df[col], q=4, labels=False, duplicates="drop"
                )
                fe_stats["binned_features"].append(binned_col)

        all_new_features = (
            fe_stats["interaction_terms"]
            + fe_stats["polynomial_features"]
            + fe_stats["binned_features"]
        )

        if all_new_features:
            steps.append(
                PreprocessingStep(
                    step_type="feature_engineering",
                    affected_columns=all_new_features,
                    parameters={
                        "treatment_candidates": treatment_candidates,
                        "outcome_candidates": outcome_candidates,
                    },
                    statistics=fe_stats,
                )
            )

        return df, steps

    def _scale_numeric(
        self,
        df: pd.DataFrame,
        columns_by_type: dict[str, list[str]],
    ) -> tuple[pd.DataFrame, list[PreprocessingStep]]:
        """Scale numeric columns using StandardScaler.

        Skip binary columns (only 0/1 values).
        """
        steps: list[PreprocessingStep] = []
        numeric_cols = [
            col for col in columns_by_type["numeric"] if col in df.columns
        ]

        # Also include any new numeric columns created during encoding/feature engineering
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = list(set(numeric_cols) | set(all_numeric))

        # Filter out binary columns
        cols_to_scale = []
        for col in numeric_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 2 or not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                cols_to_scale.append(col)

        if not cols_to_scale:
            return df, steps

        scaler = StandardScaler()
        scaling_stats = {}

        for col in cols_to_scale:
            if df[col].isna().all():
                continue
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std == 0:
                continue
            df[col] = scaler.fit_transform(df[[col]]).ravel()
            scaling_stats[col] = {
                "mean": float(col_mean),
                "std": float(col_std),
            }

        if scaling_stats:
            steps.append(
                PreprocessingStep(
                    step_type="scaling",
                    affected_columns=list(scaling_stats.keys()),
                    parameters={"method": "standard", "mean": 0, "std": 1},
                    statistics={"scaling_params": scaling_stats},
                )
            )

        return df, steps
