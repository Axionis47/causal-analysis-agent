"""FastAPI application entry point."""

import sentry_sdk
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app import __version__
from app.api.v1.admin import router as admin_router
from app.api.v1.analyses import router as analyses_router
from app.api.v1.auth import router as auth_router
from app.api.v1.comments import router as comments_router
from app.api.v1.results import router as results_router
from app.api.v1.sharing import router as sharing_router
from app.api.v1.users import router as users_router
from app.api.v1.versions import router as versions_router
from app.core.config import settings
from app.core.csrf import CSRFMiddleware, get_csrf_token
from app.core.logging import configure_logging, get_release, sentry_before_send
from app.core.middleware import RequestIDMiddleware
from app.core.rate_limit import limiter
from app.core.security_headers import SecurityHeadersMiddleware
from app.db.database import _set_main_loop

configure_logging()

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        integrations=[FastApiIntegration()],
        before_send=sentry_before_send,
        release=get_release(),
    )
    sentry_sdk.set_tag("app_name", settings.APP_NAME)
    sentry_sdk.set_tag("environment", settings.ENVIRONMENT)
    sentry_sdk.set_tag("version", __version__)


OPENAPI_TAGS = [
    {
        "name": "auth",
        "description": "Authentication endpoints for user registration, login, and password management.",
    },
    {
        "name": "users",
        "description": "User profile management endpoints.",
    },
    {
        "name": "analyses",
        "description": "Core analysis endpoints for creating, managing, and monitoring causal analyses.",
    },
    {
        "name": "results",
        "description": "Endpoints for retrieving analysis results, reports, and visualizations.",
    },
    {
        "name": "versions",
        "description": "Version control endpoints for tracking and reverting analysis configurations.",
    },
    {
        "name": "sharing",
        "description": "Endpoints for creating and managing shareable links to analyses.",
    },
    {
        "name": "comments",
        "description": "Collaboration endpoints for adding and managing comments on analyses.",
    },
    {
        "name": "admin",
        "description": "Administrative endpoints for quota management, usage monitoring, and system statistics. Requires admin privileges.",
    },
]

API_DESCRIPTION = """
# Causal Analysis API

A production-grade multi-agent system for automated causal inference on tabular datasets.

## Overview

This API provides endpoints to:
- **Create analyses** from Kaggle datasets with automatic causal discovery
- **Stream progress** in real-time via Server-Sent Events
- **View results** including executive summaries, technical reports, and visualizations
- **Download reports** in PDF, Markdown, HTML, PowerPoint, or JSON formats
- **Collaborate** with comments, sharing, and version control
- **Manage quotas** and monitor usage (admin only)

## Authentication

All endpoints (except public share links) require Bearer token authentication:

```
Authorization: Bearer <your-access-token>
```

Obtain a token by:
1. Register: `POST /api/v1/auth/register`
2. Login: `POST /api/v1/auth/token`

Tokens expire after the configured duration (default: 60 minutes). Re-authenticate to get a new token.

## Rate Limiting

To ensure fair usage, the API enforces rate limits:
- **Hourly limit**: Maximum analyses per hour per user
- **Daily limit**: Maximum analyses per day per user
- **Concurrent limit**: Maximum simultaneous running analyses

Rate limit status is returned in response headers:
- `X-RateLimit-Limit-Hourly`
- `X-RateLimit-Remaining-Hourly`
- `X-RateLimit-Reset-Hourly`

## Error Responses

The API uses standard HTTP status codes:

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid input |
| 401  | Unauthorized - Missing or invalid token |
| 403  | Forbidden - Insufficient permissions |
| 404  | Not Found - Resource doesn't exist |
| 422  | Unprocessable Entity - Validation error |
| 429  | Too Many Requests - Rate limit exceeded |
| 500  | Internal Server Error |

## CSRF Protection

State-changing requests (POST, PUT, PATCH, DELETE) require a CSRF token:
1. Fetch token: `GET /api/v1/csrf-token`
2. Include in header: `X-CSRF-Token: <token>`

## Support

- Documentation: `/api/docs` (this page) or `/api/redoc`
- GitHub: https://github.com/anthropics/claude-code/issues
"""

app = FastAPI(
    title="Causal Analysis API",
    description=API_DESCRIPTION,
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=OPENAPI_TAGS,
    responses={
        400: {
            "description": "Bad Request - Invalid input parameters",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid request parameters"}
                }
            },
        },
        401: {
            "description": "Unauthorized - Missing or invalid Bearer token",
            "content": {
                "application/json": {
                    "example": {"detail": "Not authenticated"}
                }
            },
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "example": {"detail": "Not authorized to access this resource"}
                }
            },
        },
        404: {
            "description": "Not Found - Resource doesn't exist",
            "content": {
                "application/json": {
                    "example": {"detail": "Resource not found"}
                }
            },
        },
        429: {
            "description": "Too Many Requests - Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {"detail": "Rate limit exceeded"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error"}
                }
            },
        },
    },
)

# Add rate limiter to app state
app.state.limiter = limiter

# Add SlowAPI exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# Configure CORS with explicit settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=settings.CORS_MAX_AGE,
)

# Add CSRF protection middleware
app.add_middleware(CSRFMiddleware, redis_url=settings.REDIS_URL)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)

app.include_router(admin_router)
app.include_router(analyses_router)
app.include_router(auth_router)
app.include_router(comments_router)
app.include_router(results_router)
app.include_router(sharing_router)
app.include_router(users_router)
app.include_router(versions_router)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the main event loop reference for database sessions."""
    _set_main_loop()


def _serialize_validation_errors(errors: list) -> list:
    """Serialize validation errors to JSON-safe format."""
    result = []
    for error in errors:
        serialized = {}
        for key, value in error.items():
            if isinstance(value, bytes):
                serialized[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                    for v in value
                ]
            else:
                serialized[key] = value
        result.append(serialized)
    return result


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 for invalid Kaggle URLs, 422 for other validation errors."""
    errors = _serialize_validation_errors(exc.errors())
    for error in errors:
        loc = error.get("loc", [])
        if loc and loc[-1] == "kaggle_url":
            if request.url.path.endswith("/api/v1/analyses/preview"):
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={"detail": errors},
                )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid Kaggle URL"},
            )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/v1/status")
async def api_status() -> dict[str, str]:
    """API status endpoint."""
    return {
        "status": "operational",
        "version": "0.1.0",
    }


@app.get("/api/v1/csrf-token")
async def csrf_token_endpoint(request: Request, response: Response) -> dict[str, str]:
    """Get CSRF token for the current session.

    This endpoint returns the CSRF token that should be included in the
    X-CSRF-Token header for all state-changing requests (POST, PUT, PATCH, DELETE).

    If no CSRF cookie exists, this endpoint generates a new token,
    stores it in Redis, and sets the cookie.
    """
    return await get_csrf_token(request, response)
