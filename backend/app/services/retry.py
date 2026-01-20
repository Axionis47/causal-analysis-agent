"""Retry helpers for transient failures."""

from __future__ import annotations

import errno
from typing import Any, Callable, TypeVar

from tenacity import AsyncRetrying, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_TRANSIENT_ERRNO = {
    errno.EAGAIN,
    errno.EWOULDBLOCK,
    errno.EINTR,
    errno.ECONNRESET,
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ETIMEDOUT,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.ENETDOWN,
    errno.EPIPE,
    errno.EIO,
}
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_transient_os_error(exc: OSError) -> bool:
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        return False
    if exc.errno is None:
        return True
    return exc.errno in _TRANSIENT_ERRNO


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    return status if isinstance(status, int) else None


def is_transient_io_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError) and _is_transient_os_error(exc):
        return True
    return _status_code(exc) in _TRANSIENT_STATUS_CODES


def is_transient_tool_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, RuntimeError)):
        return True
    if isinstance(exc, OSError):
        return _is_transient_os_error(exc)
    return False


def is_transient_kaggle_error(exc: BaseException) -> bool:
    if is_transient_io_error(exc):
        return True
    try:
        from kaggle.rest import ApiException
    except Exception:
        ApiException = None
    if ApiException and isinstance(exc, ApiException):
        status = _status_code(exc)
        return status in _TRANSIENT_STATUS_CODES
    return False


def is_transient_error(exc: BaseException) -> bool:
    if is_transient_io_error(exc):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "timeout",
            "temporar",
            "rate limit",
            "too many requests",
            "connection",
            "unavailable",
            "network",
        )
    )


def _build_retryer(
    max_attempts: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    retry_on: Callable[[BaseException], bool],
) -> Retrying:
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_seconds, min=1, max=max_backoff_seconds),
        retry=retry_if_exception(retry_on),
        before_sleep=_log_retry_attempt,
    )


def _build_async_retryer(
    max_attempts: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    retry_on: Callable[[BaseException], bool],
) -> AsyncRetrying:
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_seconds, min=1, max=max_backoff_seconds),
        retry=retry_if_exception(retry_on),
        before_sleep=_log_retry_attempt,
    )


def _log_retry_attempt(retry_state) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    backoff = retry_state.next_action.sleep if retry_state.next_action else None
    logger.warning(
        "Retrying operation",
        attempt_number=retry_state.attempt_number,
        backoff_seconds=backoff,
        exception_type=type(exception).__name__ if exception else None,
    )


def retry_sync(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int | None = None,
    backoff_seconds: float | None = None,
    max_backoff_seconds: float | None = None,
    retry_on: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> T:
    retryer = _build_retryer(
        max_attempts or settings.AGENT_RETRY_MAX_ATTEMPTS,
        backoff_seconds or settings.AGENT_RETRY_BACKOFF_SECONDS,
        max_backoff_seconds or settings.AGENT_RETRY_MAX_BACKOFF_SECONDS,
        retry_on or is_transient_error,
    )
    for attempt in retryer:
        with attempt:
            return func(*args, **kwargs)
    raise RuntimeError("Retrying failed without raising an exception")


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int | None = None,
    backoff_seconds: float | None = None,
    max_backoff_seconds: float | None = None,
    retry_on: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> Any:
    retryer = _build_async_retryer(
        max_attempts or settings.AGENT_RETRY_MAX_ATTEMPTS,
        backoff_seconds or settings.AGENT_RETRY_BACKOFF_SECONDS,
        max_backoff_seconds or settings.AGENT_RETRY_MAX_BACKOFF_SECONDS,
        retry_on or is_transient_error,
    )
    async for attempt in retryer:
        with attempt:
            return await func(*args, **kwargs)
    raise RuntimeError("Retrying failed without raising an exception")
