"""CSRF protection middleware for FastAPI."""

from __future__ import annotations

import hashlib
import secrets
from typing import Callable

import redis.asyncio as redis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Middleware to protect against Cross-Site Request Forgery (CSRF) attacks.

    This middleware:
    - Generates a CSRF token for each session
    - Stores tokens in Redis with a configurable TTL
    - Validates the X-CSRF-Token header on state-changing requests (POST, PUT, PATCH, DELETE)
    - Sets the CSRF token in a cookie for frontend access

    Configuration is read from settings:
    - CSRF_ENABLED: Enable/disable CSRF protection
    - CSRF_COOKIE_NAME: Name of the CSRF cookie
    - CSRF_HEADER_NAME: Name of the header to check
    - CSRF_EXEMPT_PATHS: List of paths to exempt from CSRF validation
    """

    # State-changing HTTP methods that require CSRF validation
    PROTECTED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    # Token TTL in seconds (1 hour)
    TOKEN_TTL = 3600

    def __init__(
        self,
        app: ASGIApp,
        redis_url: str | None = None,
        exempt_paths: list[str] | None = None,
    ):
        """Initialize CSRF middleware.

        Args:
            app: The ASGI application
            redis_url: Redis connection URL for token storage
            exempt_paths: Additional paths to exempt from CSRF validation
        """
        super().__init__(app)
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis_client: redis.Redis | None = None

        # Get configuration from settings
        self.enabled = getattr(settings, "CSRF_ENABLED", True)
        self.cookie_name = getattr(settings, "CSRF_COOKIE_NAME", "csrf_token")
        self.header_name = getattr(settings, "CSRF_HEADER_NAME", "X-CSRF-Token")

        # Default exempt paths
        default_exempt = [
            "/health",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/token",
            "/api/v1/csrf-token",
        ]

        # Merge with settings and additional exempt paths
        settings_exempt = getattr(settings, "CSRF_EXEMPT_PATHS", [])
        self.exempt_paths = set(default_exempt + settings_exempt + (exempt_paths or []))

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis client connection."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis_client

    def _generate_token(self) -> str:
        """Generate a cryptographically secure CSRF token."""
        return secrets.token_urlsafe(32)

    def _get_session_id(self, request: Request) -> str:
        """Extract session ID from request for token storage key.

        Uses the JWT token or falls back to a hash of IP and User-Agent.
        """
        # Try to get session from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # Use first 32 chars of token as session identifier
            return f"csrf:{token[:32]}"

        # Fallback: use deterministic hash of IP + User-Agent
        ip = request.client.host if request.client else "unknown"
        ua = request.headers.get("User-Agent", "unknown")
        identifier = f"{ip}:{ua}"
        hashed = hashlib.sha256(identifier.encode()).hexdigest()[:32]
        return f"csrf:{hashed}"

    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from CSRF validation."""
        # Exact match
        if path in self.exempt_paths:
            return True

        # Prefix match for paths ending with /*
        for exempt_path in self.exempt_paths:
            if exempt_path.endswith("/*") and path.startswith(exempt_path[:-2]):
                return True

        return False

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Process request with CSRF protection.

        For GET requests: Generate and set CSRF token
        For state-changing requests: Validate CSRF token
        """
        # Skip if CSRF is disabled
        if not self.enabled:
            return await call_next(request)

        # Skip for exempt paths
        if self._is_exempt(request.url.path):
            return await call_next(request)

        # Skip for non-protected methods
        if request.method not in self.PROTECTED_METHODS:
            response = await call_next(request)

            # Set CSRF token cookie on GET requests if not already set
            if request.method == "GET":
                csrf_cookie = request.cookies.get(self.cookie_name)
                if not csrf_cookie:
                    token = self._generate_token()
                    session_id = self._get_session_id(request)

                    # Store in Redis
                    try:
                        redis_client = await self._get_redis()
                        await redis_client.setex(session_id, self.TOKEN_TTL, token)
                    except Exception as e:
                        logger.warning(f"Failed to store CSRF token in Redis: {e}")

                    # Set cookie
                    response.set_cookie(
                        key=self.cookie_name,
                        value=token,
                        httponly=False,  # Must be accessible by JavaScript
                        secure=settings.ENVIRONMENT != "development",
                        samesite="strict",
                        max_age=self.TOKEN_TTL,
                    )

            return response

        # Validate CSRF token for protected methods
        csrf_header = request.headers.get(self.header_name)
        csrf_cookie = request.cookies.get(self.cookie_name)

        # Both must be present
        if not csrf_header or not csrf_cookie:
            logger.warning(
                "CSRF validation failed: missing token",
                path=request.url.path,
                method=request.method,
                has_header=bool(csrf_header),
                has_cookie=bool(csrf_cookie),
            )
            return Response(
                content='{"detail": "CSRF token missing"}',
                status_code=403,
                media_type="application/json",
            )

        # Tokens must match
        if not secrets.compare_digest(csrf_header, csrf_cookie):
            logger.warning(
                "CSRF validation failed: token mismatch",
                path=request.url.path,
                method=request.method,
            )
            return Response(
                content='{"detail": "CSRF token invalid"}',
                status_code=403,
                media_type="application/json",
            )

        # Optionally validate against Redis stored token
        try:
            session_id = self._get_session_id(request)
            redis_client = await self._get_redis()
            stored_token = await redis_client.get(session_id)

            if stored_token and not secrets.compare_digest(csrf_header, stored_token):
                logger.warning(
                    "CSRF validation failed: Redis token mismatch",
                    path=request.url.path,
                )
                return Response(
                    content='{"detail": "CSRF token expired or invalid"}',
                    status_code=403,
                    media_type="application/json",
                )
        except Exception as e:
            # Log but don't fail if Redis is unavailable
            logger.warning(f"Failed to validate CSRF token in Redis: {e}")

        return await call_next(request)


# Token TTL for the get_csrf_token endpoint (must match middleware)
_TOKEN_TTL = 3600


def _get_session_id_for_request(request: Request) -> str:
    """Extract session ID from request for token storage key (standalone function).

    Uses the JWT token or falls back to a hash of IP and User-Agent.
    """
    # Try to get session from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # Use first 32 chars of token as session identifier
        return f"csrf:{token[:32]}"

    # Fallback: use deterministic hash of IP + User-Agent
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("User-Agent", "unknown")
    identifier = f"{ip}:{ua}"
    hashed = hashlib.sha256(identifier.encode()).hexdigest()[:32]
    return f"csrf:{hashed}"


# Endpoint to get CSRF token
async def get_csrf_token(request: Request, response: Response) -> dict[str, str]:
    """Get a CSRF token for the current session.

    This endpoint returns the CSRF token that should be included in the
    X-CSRF-Token header for all state-changing requests.

    If no CSRF cookie exists, this endpoint generates a new token,
    stores it in Redis, and sets the cookie.
    """
    cookie_name = getattr(settings, "CSRF_COOKIE_NAME", "csrf_token")
    existing_token = request.cookies.get(cookie_name, "")

    if existing_token:
        return {"csrf_token": existing_token}

    # Generate a new token
    token = secrets.token_urlsafe(32)
    session_id = _get_session_id_for_request(request)

    # Store in Redis
    try:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.setex(session_id, _TOKEN_TTL, token)
        await redis_client.aclose()
    except Exception as e:
        logger.warning(f"Failed to store CSRF token in Redis: {e}")

    # Set cookie on response
    response.set_cookie(
        key=cookie_name,
        value=token,
        httponly=False,  # Must be accessible by JavaScript
        secure=settings.ENVIRONMENT != "development",
        samesite="strict",
        max_age=_TOKEN_TTL,
    )

    return {"csrf_token": token}
