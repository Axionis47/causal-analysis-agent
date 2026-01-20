"""Structured logging configuration and helpers."""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Mapping

import sentry_sdk
import structlog

from app import __version__
from app.core.config import settings

SENSITIVE_KEYS = [
    "api_key",
    "password",
    "token",
    "secret",
    "authorization",
    "kaggle_key",
    "openai_api_key",
    "anthropic_api_key",
]
_REDACTED_VALUE = "***REDACTED***"

_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_user_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)
_analysis_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("analysis_id", default=None)

_GLOBAL_CONTEXT = {
    "environment": settings.ENVIRONMENT,
    "app_name": settings.APP_NAME,
    "version": __version__,
}

LLM_CACHE_EVENTS = {
    "llm.cache.hit",
    "llm.cache.miss",
    "llm.cache.set",
    "llm.cache.error",
}

_configured = False


def bind_contextvars(
    *,
    request_id: str | None = None,
    user_id: str | None = None,
    analysis_id: str | None = None,
) -> None:
    """Bind request/user/analysis identifiers to context variables."""
    if request_id is not None:
        _request_id_ctx.set(str(request_id))
    if user_id is not None:
        _user_id_ctx.set(str(user_id))
    if analysis_id is not None:
        _analysis_id_ctx.set(str(analysis_id))


def get_log_context() -> dict[str, str]:
    """Return the active logging context derived from context variables."""
    context: dict[str, str] = {}
    request_id = _request_id_ctx.get()
    user_id = _user_id_ctx.get()
    analysis_id = _analysis_id_ctx.get()
    if request_id:
        context["request_id"] = request_id
    if user_id:
        context["user_id"] = user_id
    if analysis_id:
        context["analysis_id"] = analysis_id
    return context


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact sensitive fields in nested payloads."""
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str.lower() in SENSITIVE_KEYS:
                redacted[key_str] = _REDACTED_VALUE
            else:
                redacted[key_str] = redact_sensitive_data(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    return value


def filter_sensitive_data(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to redact sensitive values before rendering."""
    return redact_sensitive_data(event_dict)


def add_correlation_id(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach request/user/analysis IDs from context variables."""
    for key, value in get_log_context().items():
        event_dict.setdefault(key, value)
    return event_dict


def _resolve_log_level() -> int:
    level_name = (settings.LOG_LEVEL or "").strip().upper()
    if level_name:
        level = logging._nameToLevel.get(level_name)
        if level is not None:
            return level
    if settings.ENVIRONMENT.lower() == "development":
        return logging.DEBUG
    return logging.INFO


def configure_logging() -> None:
    """Configure structlog and stdlib logging for structured output."""
    global _configured
    if _configured:
        return
    _configured = True

    if not settings.ENABLE_STRUCTURED_LOGGING:
        logging.basicConfig(level=_resolve_log_level())
        return

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.LOG_FORMAT.lower() == "json"
        else structlog.dev.ConsoleRenderer()
    )
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain = [
        add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.format_exc_info,
        filter_sensitive_data,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=pre_chain,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(_resolve_log_level())
    logging.captureWarnings(True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            add_correlation_id,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            filter_sensitive_data,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with module and global context."""
    configure_logging()
    return structlog.get_logger(name).bind(module=name, **_GLOBAL_CONTEXT)


def log_llm_cache_event(event: str, **fields: Any) -> None:
    """Emit structured cache metrics for LLM responses."""
    logger = get_logger("app.llm.cache")
    logger.info(event, **fields)


def get_release() -> str:
    """Return the release identifier for Sentry reporting."""
    return os.getenv("GIT_COMMIT") or os.getenv("COMMIT_SHA") or __version__


def sentry_before_send(event: dict[str, Any], _hint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Redact sensitive keys from Sentry payloads."""
    if not isinstance(event, dict):
        return event
    return redact_sensitive_data(event)


def capture_exception(exc: BaseException) -> None:
    """Capture exceptions in Sentry with bound context."""
    context = get_log_context()
    with sentry_sdk.push_scope() as scope:
        for key, value in context.items():
            scope.set_tag(key, value)
        sentry_sdk.capture_exception(exc)
