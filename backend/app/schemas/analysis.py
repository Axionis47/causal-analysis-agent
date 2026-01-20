"""Pydantic schemas for analysis requests and responses."""

import re
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.analysis import AnalysisStatus
from app.schemas.preprocessing import PreprocessingConfig
from app.schemas.validation import validate_config_values

KAGGLE_URL_PATTERN = re.compile(
    r"^https://www\.kaggle\.com/(datasets|competitions)/.+$"
)

# Maximum URL length to prevent abuse
MAX_URL_LENGTH = 500


class AnalysisConfig(BaseModel):
    """Config options for analysis runs."""

    model_config = ConfigDict(extra="allow")

    override_quality_warnings: bool = Field(
        False,
        description="Allow proceeding when only non-critical quality warnings exist.",
    )
    enable_preprocessing: bool = Field(
        True,
        description="Whether to apply data preprocessing after EDA.",
    )
    preprocessing_config: PreprocessingConfig | None = Field(
        None,
        description="Detailed preprocessing configuration. If not provided, defaults are used.",
    )
    include_engineered_confounders: bool = Field(
        False,
        description="Whether to include engineered features (fe_*) as confounders in treatment estimation.",
    )

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_config_values(cls, value: Any) -> Any:
        """Sanitize all config values to prevent XSS."""
        if isinstance(value, dict):
            return validate_config_values(value)
        return value

    @model_validator(mode="after")
    def merge_preprocessing_defaults(self) -> "AnalysisConfig":
        """Ensure preprocessing_config has defaults merged if partially provided."""
        if self.preprocessing_config is None and self.enable_preprocessing:
            self.preprocessing_config = PreprocessingConfig()
        return self


class AnalysisCreate(BaseModel):
    """Request schema for creating an analysis."""

    kaggle_url: str = Field(
        ...,
        description="Kaggle dataset or competition URL",
        max_length=MAX_URL_LENGTH,
    )
    config: AnalysisConfig | None = None

    @field_validator("kaggle_url")
    @classmethod
    def validate_kaggle_url(cls, value: str) -> str:
        """Validate Kaggle URL format with enhanced security checks."""
        # Length check
        if len(value) > MAX_URL_LENGTH:
            raise ValueError(f"URL must not exceed {MAX_URL_LENGTH} characters")

        # Parse URL for validation
        parsed = urlparse(value)

        # Scheme must be https only
        if parsed.scheme != "https":
            raise ValueError("URL scheme must be https")

        # Domain must be exactly www.kaggle.com (no subdomains)
        if parsed.netloc != "www.kaggle.com":
            raise ValueError("URL domain must be www.kaggle.com")

        # Reject URLs with embedded credentials
        if parsed.username or parsed.password:
            raise ValueError("URL must not contain embedded credentials")

        # Reject URLs with fragments (could be used for XSS)
        if parsed.fragment:
            raise ValueError("URL must not contain a fragment")

        # Path format validation
        if not KAGGLE_URL_PATTERN.match(value):
            raise ValueError(
                "kaggle_url must match https://www.kaggle.com/datasets/* "
                "or https://www.kaggle.com/competitions/*"
            )

        return value


class AnalysisUpdate(BaseModel):
    """Request schema for updating an analysis."""

    status: AnalysisStatus | None = None
    error_message: str | None = None


class AnalysisResponse(BaseModel):
    """Response schema for analysis data."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID | None
    kaggle_url: str
    status: AnalysisStatus
    config: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    estimated_duration: int | None
    error_message: str | None
    is_partial: bool
    created_at: datetime
    updated_at: datetime
    # Cost tracking fields (optional for backward compatibility)
    llm_tokens_used: int | None = 0
    estimated_cost: float | None = 0.0


class AnalysisListResponse(BaseModel):
    """Paginated response for analyses."""

    items: list[AnalysisResponse]
    total: int
    skip: int
    limit: int


class DataQualityPreviewRequest(BaseModel):
    """Request schema for previewing data quality warnings."""

    kaggle_url: str = Field(
        ...,
        description="Kaggle dataset or competition URL",
        max_length=MAX_URL_LENGTH,
    )
    sample_rows: int | None = Field(
        10000,
        ge=1,
        le=50000,
        description="Number of rows to sample for validation (max 50000).",
    )

    @field_validator("kaggle_url")
    @classmethod
    def validate_kaggle_url(cls, value: str) -> str:
        """Validate Kaggle URL format with enhanced security checks."""
        # Length check
        if len(value) > MAX_URL_LENGTH:
            raise ValueError(f"URL must not exceed {MAX_URL_LENGTH} characters")

        # Parse URL for validation
        parsed = urlparse(value)

        # Scheme must be https only
        if parsed.scheme != "https":
            raise ValueError("URL scheme must be https")

        # Domain must be exactly www.kaggle.com (no subdomains)
        if parsed.netloc != "www.kaggle.com":
            raise ValueError("URL domain must be www.kaggle.com")

        # Reject URLs with embedded credentials
        if parsed.username or parsed.password:
            raise ValueError("URL must not contain embedded credentials")

        # Reject URLs with fragments
        if parsed.fragment:
            raise ValueError("URL must not contain a fragment")

        # Path format validation
        if not KAGGLE_URL_PATTERN.match(value):
            raise ValueError(
                "kaggle_url must match https://www.kaggle.com/datasets/* "
                "or https://www.kaggle.com/competitions/*"
            )

        return value


class DataQualityPreviewResponse(BaseModel):
    """Response schema for data quality preview results."""

    passed: bool
    warnings: list[dict[str, Any]]
    can_proceed_with_override: bool
    dataset_info: dict[str, Any]
    estimated_duration_seconds: int | None


class HypothesisTestRequest(BaseModel):
    """Request schema for hypothesis testing (what-if analysis)."""

    treatment: str = Field(..., description="Treatment variable name")
    outcome: str = Field(..., description="Outcome variable name")
    confounders: list[str] = Field(
        default_factory=list,
        description="List of confounder variable names",
    )
    analysis_types: list[str] = Field(
        default_factory=lambda: ["ate"],
        description="Types of analysis to perform: ate, cate, sensitivity",
    )

    @field_validator("analysis_types")
    @classmethod
    def validate_analysis_types(cls, value: list[str]) -> list[str]:
        """Validate analysis types."""
        valid_types = {"ate", "cate", "sensitivity"}
        for t in value:
            if t not in valid_types:
                raise ValueError(f"Invalid analysis type: {t}. Must be one of {valid_types}")
        return value


class HypothesisTestResponse(BaseModel):
    """Response schema for hypothesis test results."""

    analysis_id: uuid.UUID
    treatment: str
    outcome: str
    confounders: list[str]
    results: dict[str, Any]
    created_at: datetime


class ComparisonRequest(BaseModel):
    """Request schema for comparing analyses."""

    analysis_ids: list[uuid.UUID] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of analysis IDs to compare (2-5)",
    )


class ComparisonResponse(BaseModel):
    """Response schema for analysis comparison."""

    analyses: list[dict[str, Any]]
    effect_differences: list[dict[str, Any]]
    graph_similarities: list[dict[str, Any]]
    summary: dict[str, Any]
