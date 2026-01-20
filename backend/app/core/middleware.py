"""Middleware for request tracing and logging context."""

from __future__ import annotations

import time
import uuid

import sentry_sdk
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.logging import bind_contextvars, get_logger


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach request IDs to responses and bind them to logging context."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.logger = get_logger(__name__)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        bind_contextvars(request_id=request_id)

        start_time = time.monotonic()
        self.logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            request_id=request_id,
            user_id=_safe_user_id(request),
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - start_time) * 1000
            _bind_sentry_user(request)
            self.logger.exception(
                "Request failed",
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=duration_ms,
                request_id=request_id,
                user_id=_safe_user_id(request),
            )
            raise

        response.headers["X-Request-ID"] = request_id
        duration_ms = (time.monotonic() - start_time) * 1000
        _bind_sentry_user(request)
        self.logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
            user_id=_safe_user_id(request),
        )
        return response


def _safe_user_id(request: Request) -> str | None:
    user_id = getattr(request.state, "user_id", None)
    return str(user_id) if user_id else None


def _bind_sentry_user(request: Request) -> None:
    user_id = _safe_user_id(request)
    if user_id:
        sentry_sdk.set_user({"id": user_id})
