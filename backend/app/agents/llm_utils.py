"""LLM retry helpers for agents."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeVar

from tenacity import RetryCallState, retry, retry_if_exception, retry_if_result, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.services.retry import is_transient_error

T = TypeVar("T")


def _is_retryable_result(result: Any) -> bool:
    return result is None


def _fallback(_retry_state: RetryCallState) -> None:
    return None


@retry(
    stop=stop_after_attempt(settings.LLM_RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=settings.LLM_RETRY_BACKOFF_SECONDS, min=1, max=8),
    retry=retry_if_exception(is_transient_error) | retry_if_result(_is_retryable_result),
    retry_error_callback=_fallback,
)
async def call_llm_with_retry(
    func: Callable[..., Awaitable[T | None]],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    return await func(*args, **kwargs)
