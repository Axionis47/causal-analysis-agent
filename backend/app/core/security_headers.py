"""Security headers middleware for FastAPI."""

from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all HTTP responses.

    This middleware adds the following security headers:
    - Content-Security-Policy (CSP)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Strict-Transport-Security (HSTS) - only in production
    - Referrer-Policy
    - Permissions-Policy

    Configuration is read from settings:
    - SECURITY_HEADERS_ENABLED: Enable/disable security headers
    - CSP_POLICY: Custom Content-Security-Policy
    - ENABLE_HSTS: Enable HSTS (only for HTTPS)
    - HSTS_MAX_AGE: Max age for HSTS header
    """

    # Default Content-Security-Policy
    DEFAULT_CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://api.openai.com https://api.anthropic.com; "
        "frame-ancestors 'none'"
    )

    def __init__(self, app: ASGIApp):
        """Initialize security headers middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

        # Get configuration from settings
        self.enabled = getattr(settings, "SECURITY_HEADERS_ENABLED", True)
        self.csp_policy = getattr(settings, "CSP_POLICY", self.DEFAULT_CSP)
        self.enable_hsts = getattr(settings, "ENABLE_HSTS", True)
        self.hsts_max_age = getattr(settings, "HSTS_MAX_AGE", 31536000)  # 1 year

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Process request and add security headers to response."""
        response = await call_next(request)

        if not self.enabled:
            return response

        # Content-Security-Policy
        response.headers["Content-Security-Policy"] = self.csp_policy

        # X-Frame-Options - prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options - prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection - legacy XSS protection for older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Strict-Transport-Security (HSTS)
        # Only enable in production and when HTTPS is in use
        if self.enable_hsts and settings.ENVIRONMENT != "development":
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains"
            )

        # Referrer-Policy - control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy - restrict browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response
