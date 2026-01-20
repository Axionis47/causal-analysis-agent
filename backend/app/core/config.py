"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import List
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Application
    APP_NAME: str = "Causal Analysis API"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console
    ENABLE_STRUCTURED_LOGGING: bool = True

    # Sentry
    SENTRY_DSN: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    SENTRY_ENVIRONMENT: str = ""  # defaults to ENVIRONMENT if not set

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/causal_analysis"
    DATABASE_POOL_SIZE: int = 5  # Minimum connections in pool
    DATABASE_MAX_OVERFLOW: int = 15  # Max = pool_size + max_overflow = 20

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Progress streaming
    PROGRESS_POLL_INTERVAL: float = 2.0
    PROGRESS_MAX_POLL_INTERVAL: float = 10.0
    PROGRESS_STREAM_TIMEOUT: int = 300
    PROGRESS_FALLBACK_TO_DB: bool = True

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Security
    SECRET_KEY: str = "change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"
    CREDENTIALS_ENCRYPTION_KEY: str = "change-this-32-byte-key"

    # Credential encryption key versioning (for key rotation)
    CREDENTIALS_ENCRYPTION_KEY_VERSION: str = "v1"
    CREDENTIALS_ENCRYPTION_KEY_V1: str = ""  # Set in production
    CREDENTIALS_ENCRYPTION_KEY_V2: str = ""  # For rotation

    # CSRF Protection
    CSRF_ENABLED: bool = True
    CSRF_COOKIE_NAME: str = "csrf_token"
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    CSRF_EXEMPT_PATHS: List[str] = []

    # Security Headers
    SECURITY_HEADERS_ENABLED: bool = True
    CSP_POLICY: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://api.openai.com https://api.anthropic.com; "
        "frame-ancestors 'none'"
    )
    ENABLE_HSTS: bool = True
    HSTS_MAX_AGE: int = 31536000  # 1 year

    # Content Sanitization
    SANITIZATION_ALLOWED_TAGS: List[str] = ["b", "i", "a", "code", "pre", "strong", "em"]
    SANITIZATION_ALLOWED_ATTRIBUTES: dict = {"a": ["href", "title"]}
    SANITIZATION_MAX_CONTENT_LENGTH: int = 10000

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    CORS_MAX_AGE: int = 3600
    CORS_ALLOW_CREDENTIALS: bool = True

    # LLM Providers
    VERTEX_AI_PROJECT: str = ""
    VERTEX_AI_LOCATION: str = "us-central1"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "causal-analysis"

    # Storage
    GCS_BUCKET_NAME: str = "causal-analysis-datasets"
    GCS_REPORTS_BUCKET_NAME: str = "causal-analysis-reports"
    LOCAL_STORAGE_PATH: str = "./storage"

    # Reporting
    REPORT_TEMPLATE_DIR: str = "./app/templates"
    REPORT_PDF_PAGE_SIZE: str = "A4"
    REPORT_PDF_MARGIN: str = "1cm"
    REPORT_SIGNED_URL_EXPIRATION_MINUTES: int = 60
    REPORT_ENABLE_PDF: bool = True
    REPORT_ENABLE_MARKDOWN: bool = True
    REPORT_ENABLE_PPTX: bool = True

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"

    # Feature flags / tools
    ENABLE_BETA_TOOLS: bool = False
    ENABLE_ALPHA_TOOLS: bool = False
    ENABLE_GES: bool = False
    ENABLE_FCI: bool = False
    ENABLE_DOUBLY_ROBUST: bool = False
    ENABLE_IV: bool = False
    TOOL_CONFIG_PATH: str = "./config/tools.yaml"

    # LLM routing
    LLM_PRIMARY_PROVIDER: str = "vertex"
    LLM_FALLBACK_PROVIDERS: List[str] = ["openai", "anthropic"]
    LLM_CIRCUIT_BREAKER_THRESHOLD: int = 5
    LLM_CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 300
    LLM_RETRY_MAX_ATTEMPTS: int = 4
    LLM_RETRY_BACKOFF_SECONDS: float = 1.0

    # LLM caching
    LLM_CACHE_ENABLED: bool = True
    LLM_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    LLM_CACHE_MAX_PROMPT_LENGTH: int = 10000

    # Cost-based routing
    LLM_COST_BASED_ROUTING_ENABLED: bool = True
    LLM_SIMPLE_PROMPT_MAX_LENGTH: int = 500

    # Agent retries
    AGENT_RETRY_MAX_ATTEMPTS: int = 3
    AGENT_RETRY_BACKOFF_SECONDS: float = 1.0
    AGENT_RETRY_MAX_BACKOFF_SECONDS: float = 8.0

    # Kaggle
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    KAGGLE_CIRCUIT_BREAKER_THRESHOLD: int = 3
    KAGGLE_CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 300
    KAGGLE_RETRY_MAX_ATTEMPTS: int = 4
    KAGGLE_RETRY_BACKOFF_SECONDS: float = 1.0

    # Data quality validation
    DATA_QUALITY_MIN_ROWS: int = 100
    DATA_QUALITY_WARN_ROWS: int = 500
    DATA_QUALITY_MAX_MISSING_PERCENT: float = 0.5
    DATA_QUALITY_WARN_MISSING_PERCENT: float = 0.5
    DATA_QUALITY_MIN_NUMERIC_COLUMNS: int = 2
    DATA_QUALITY_WARN_NUMERIC_COLUMNS: int = 3
    DATA_QUALITY_MAX_DUPLICATE_PERCENT: float = 0.3
    DATA_QUALITY_MAX_IMBALANCE_RATIO: float = 0.95

    # Bootstrap confidence intervals
    BOOTSTRAP_NUM_SIMULATIONS: int = 500  # Number of bootstrap iterations
    BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95  # CI confidence level (e.g., 0.95 for 95%)

    # Causal discovery cross-validation
    DISCOVERY_CV_FOLDS: int = 5  # Number of cross-validation folds
    DISCOVERY_STABILITY_THRESHOLD: float = 0.6  # Edge stability threshold (0-1)
    DISCOVERY_SUBSAMPLE_FRACTION: float = 0.8  # Fraction of data for each fold

    # Sensitivity analysis thresholds
    SENSITIVITY_RV_THRESHOLD: float = 0.1  # Robustness value threshold
    SENSITIVITY_EVALUE_THRESHOLD: float = 1.5  # E-value threshold

    # Rate limiting
    RATE_LIMIT_ANALYSES_PER_HOUR: int = 10
    RATE_LIMIT_ANALYSES_PER_DAY: int = 50
    RATE_LIMIT_CONCURRENT_ANALYSES: int = 3
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_STORAGE_URL: str = "redis://localhost:6379/3"

    # Admin
    ADMIN_USER_IDS: List[str] = []

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: List[str], info) -> List[str]:
        """Validate CORS origins for security.

        - Each origin must start with http:// or https://
        - No wildcards allowed in production
        - Reject 'null' origin
        """
        # Get environment from info.data (already validated fields)
        environment = info.data.get("ENVIRONMENT", "development")

        validated = []
        for origin in v:
            # Reject null origin (can be used in CSRF attacks)
            if origin.lower() == "null":
                raise ValueError("CORS origin 'null' is not allowed")

            # Reject wildcard origins in non-development environments
            if "*" in origin:
                if environment != "development":
                    raise ValueError(
                        f"Wildcard CORS origins are not allowed in {environment} environment: {origin}"
                    )

            # Must start with http:// or https:// (skip wildcard check for dev)
            if origin != "*" and not (origin.startswith("http://") or origin.startswith("https://")):
                raise ValueError(f"CORS origin must start with http:// or https://: {origin}")

            # Parse to validate URL format (skip for pure wildcard)
            if origin != "*":
                try:
                    parsed = urlparse(origin)
                    if not parsed.netloc:
                        raise ValueError(f"Invalid CORS origin URL: {origin}")
                except Exception:
                    raise ValueError(f"Invalid CORS origin URL format: {origin}")

            validated.append(origin)

        return validated

    @field_validator("CREDENTIALS_ENCRYPTION_KEY")
    @classmethod
    def validate_encryption_key(cls, v: str) -> str:
        """Validate encryption key has minimum length for security."""
        if len(v) < 16:
            raise ValueError(
                "CREDENTIALS_ENCRYPTION_KEY must be at least 16 characters"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
