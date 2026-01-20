"""Preprocessing configuration schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""

    enable_preprocessing: bool = Field(
        default=True,
        description="Whether to apply preprocessing transformations",
    )
    enable_imputation: bool = Field(
        default=True,
        description="Whether to impute missing values",
    )
    enable_outlier_handling: bool = Field(
        default=True,
        description="Whether to handle outliers via winsorization",
    )
    enable_encoding: bool = Field(
        default=True,
        description="Whether to encode categorical variables",
    )
    enable_scaling: bool = Field(
        default=True,
        description="Whether to scale numeric columns (StandardScaler)",
    )
    enable_feature_engineering: bool = Field(
        default=False,
        description="Whether to create engineered features (interactions, polynomials, bins)",
    )
    outlier_method: Literal["iqr", "zscore"] = Field(
        default="iqr",
        description="Method for outlier detection: 'iqr' (Interquartile Range) or 'zscore'",
    )
    outlier_threshold: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Threshold for outlier detection (IQR multiplier or z-score limit)",
    )
    categorical_encoding_strategy: Literal["auto", "onehot", "target", "frequency"] = Field(
        default="auto",
        description=(
            "Encoding strategy: 'auto' (based on cardinality), 'onehot' (one-hot encoding), "
            "'target' (target encoding), 'frequency' (frequency encoding)"
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enable_preprocessing": True,
                "enable_imputation": True,
                "enable_outlier_handling": True,
                "enable_encoding": True,
                "enable_scaling": True,
                "enable_feature_engineering": False,
                "outlier_method": "iqr",
                "outlier_threshold": 1.5,
                "categorical_encoding_strategy": "auto",
            }
        }


class PreprocessingStepSchema(BaseModel):
    """Schema for a single preprocessing step."""

    step_type: Literal[
        "imputation", "encoding", "scaling", "outlier_handling", "feature_engineering"
    ]
    affected_columns: list[str]
    parameters: dict
    statistics: dict


class PreprocessingPreviewRequest(BaseModel):
    """Request body for preprocessing preview endpoint."""

    config: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing configuration to preview",
    )


class PreprocessingPreviewResponse(BaseModel):
    """Response from preprocessing preview endpoint."""

    original_shape: list[int] = Field(
        description="Original dataset shape [rows, columns]",
    )
    preprocessed_shape: list[int] = Field(
        description="Preprocessed dataset shape [rows, columns]",
    )
    steps_applied: list[PreprocessingStepSchema] = Field(
        description="List of preprocessing steps that would be applied",
    )
    sample_data: dict = Field(
        description="Sample of preprocessed data (first 10 rows)",
    )
    column_changes: dict = Field(
        default_factory=dict,
        description="Summary of column changes (added, removed, modified)",
    )
