"""Routing layer for LLM providers with fallback and circuit breaker."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_session_context
from app.llm.cost_router import PromptComplexity, classify_prompt_complexity, select_model_for_complexity
from app.llm.providers import AnthropicProvider, MockProvider, OpenAIProvider, VertexAIProvider
from app.llm.types import LLMProvider, LLMResponse
from app.models.llm_log import LLMLog
from app.services.circuit_breaker import CircuitBreaker
from app.services.llm_cache import LLMCacheService
from app.services.quota_manager import quota_manager
from app.services.tracing import traced

logger = get_logger(__name__)


class LLMRouter:
    """Router that tries providers in order with fallback and retries."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {
            "vertex": VertexAIProvider(),
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "mock": MockProvider(),
        }
        self._breaker = CircuitBreaker(
            failure_threshold=settings.LLM_CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout_seconds=settings.LLM_CIRCUIT_BREAKER_TIMEOUT_SECONDS,
        )
        self._cache = LLMCacheService()
        self._background_tasks: set[asyncio.Task] = set()

    def _ordered_providers(self) -> list[LLMProvider]:
        primary = settings.LLM_PRIMARY_PROVIDER
        fallback = settings.LLM_FALLBACK_PROVIDERS
        order = [primary] + [p for p in fallback if p != primary]
        return [self._providers[name] for name in order if name in self._providers]

    async def complete(
        self,
        prompt: str,
        analysis_id: str | uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResponse | None:
        analysis_id = analysis_id or kwargs.pop("analysis_id", None)
        has_explicit_model = "model" in kwargs
        complexity = None if has_explicit_model else self._classify_complexity(prompt)
        for provider in self._ordered_providers():
            if not self._breaker.is_available(provider.name):
                logger.warning("LLM provider is circuit-broken", provider=provider.name)
                continue

            model_name = kwargs.get("model") if has_explicit_model else self._select_model(provider.name, complexity)
            cache_kwargs, provider_kwargs = self._prepare_kwargs(kwargs, model_name, "complete")
            resolved_model = self._resolve_model(provider, model_name, provider_kwargs)
            attempt_started = time.monotonic()

            cached = await self._maybe_get_cache(prompt, resolved_model, cache_kwargs, provider.name)
            if cached is not None:
                if isinstance(cached, LLMResponse):
                    latency_ms = self._latency_ms(attempt_started)
                    self._log_usage(provider.name, cached, cache_hit=True, model=resolved_model)
                    self._schedule_task(
                        self._log_llm_call(
                            analysis_id=analysis_id,
                            provider=provider.name,
                            model=resolved_model,
                            prompt=prompt,
                            response_text=cached.text,
                            prompt_tokens=cached.prompt_tokens,
                            completion_tokens=cached.completion_tokens,
                            total_tokens=cached.total_tokens,
                            cache_hit=True,
                            latency_ms=latency_ms,
                            error=None,
                            metadata={"response_type": "complete"},
                        )
                    )
                    return cached
                logger.warning(
                    "Cache returned structured payload for completion",
                    provider=provider.name,
                )

            try:
                response = await traced(f"llm.{provider.name}.complete", run_type="llm")(self._run_with_retries)(
                    provider.complete,
                    prompt,
                    **provider_kwargs,
                )
                latency_ms = self._latency_ms(attempt_started)
                self._breaker.record_success(provider.name)
                self._log_usage(provider.name, response, cache_hit=False, model=resolved_model)
                await self._maybe_set_cache(prompt, resolved_model, response, cache_kwargs, provider.name)
                prompt_tokens, completion_tokens, total_tokens, tokens_estimated = self._token_metrics(
                    prompt,
                    response.text,
                    response,
                )
                metadata = {"response_type": "complete", "tokens_estimated": tokens_estimated}
                self._schedule_task(
                    self._log_llm_call(
                        analysis_id=analysis_id,
                        provider=provider.name,
                        model=resolved_model,
                        prompt=prompt,
                        response_text=response.text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cache_hit=False,
                        latency_ms=latency_ms,
                        error=None,
                        metadata=metadata,
                    )
                )
                self._schedule_task(
                    self._track_llm_usage(
                        analysis_id=analysis_id,
                        tokens=total_tokens,
                        model=resolved_model,
                        cache_hit=False,
                    )
                )
                return response
            except Exception as exc:  # noqa: BLE001 - fallback behavior
                latency_ms = self._latency_ms(attempt_started)
                self._breaker.record_failure(provider.name)
                logger.exception("LLM provider failed", provider=provider.name, error=str(exc))
                self._schedule_task(
                    self._log_llm_call(
                        analysis_id=analysis_id,
                        provider=provider.name,
                        model=resolved_model,
                        prompt=prompt,
                        response_text="",
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        cache_hit=False,
                        latency_ms=latency_ms,
                        error=str(exc),
                        metadata={"response_type": "complete"},
                    )
                )
                continue
        return None

    async def structured_output(
        self,
        prompt: str,
        analysis_id: str | uuid.UUID | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        analysis_id = analysis_id or kwargs.pop("analysis_id", None)
        has_explicit_model = "model" in kwargs
        complexity = None if has_explicit_model else self._classify_complexity(prompt)
        for provider in self._ordered_providers():
            if not self._breaker.is_available(provider.name):
                logger.warning("LLM provider is circuit-broken", provider=provider.name)
                continue

            model_name = kwargs.get("model") if has_explicit_model else self._select_model(provider.name, complexity)
            cache_kwargs, provider_kwargs = self._prepare_kwargs(kwargs, model_name, "structured")
            resolved_model = self._resolve_model(provider, model_name, provider_kwargs)
            attempt_started = time.monotonic()

            cached = await self._maybe_get_cache(prompt, resolved_model, cache_kwargs, provider.name)
            if isinstance(cached, dict):
                latency_ms = self._latency_ms(attempt_started)
                response_text = json.dumps(cached, ensure_ascii=True)
                prompt_tokens, completion_tokens, total_tokens, tokens_estimated = self._token_metrics(
                    prompt,
                    response_text,
                    None,
                )
                metadata = {"response_type": "structured", "tokens_estimated": tokens_estimated}
                self._schedule_task(
                    self._log_llm_call(
                        analysis_id=analysis_id,
                        provider=provider.name,
                        model=resolved_model,
                        prompt=prompt,
                        response_text=response_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cache_hit=True,
                        latency_ms=latency_ms,
                        error=None,
                        metadata=metadata,
                    )
                )
                return cached

            try:
                response = await traced(f"llm.{provider.name}.structured", run_type="llm")(self._run_with_retries)(
                    provider.complete,
                    prompt,
                    **provider_kwargs,
                )
                latency_ms = self._latency_ms(attempt_started)
                self._log_usage(provider.name, response, cache_hit=False, model=resolved_model)
                response_text = response.text
                prompt_tokens, completion_tokens, total_tokens, tokens_estimated = self._token_metrics(
                    prompt,
                    response_text,
                    response,
                )
                metadata = {"response_type": "structured", "tokens_estimated": tokens_estimated}
                structured = self._parse_structured_response(provider.name, response)
                self._schedule_task(
                    self._log_llm_call(
                        analysis_id=analysis_id,
                        provider=provider.name,
                        model=resolved_model,
                        prompt=prompt,
                        response_text=response_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cache_hit=False,
                        latency_ms=latency_ms,
                        error=None if structured is not None else "structured output returned None",
                        metadata=metadata,
                    )
                )
                self._schedule_task(
                    self._track_llm_usage(
                        analysis_id=analysis_id,
                        tokens=total_tokens,
                        model=resolved_model,
                        cache_hit=False,
                    )
                )
                if structured is None:
                    self._breaker.record_failure(provider.name)
                    continue
                self._breaker.record_success(provider.name)
                await self._maybe_set_cache(prompt, resolved_model, structured, cache_kwargs, provider.name)
                return structured
            except Exception as exc:  # noqa: BLE001 - fallback behavior
                latency_ms = self._latency_ms(attempt_started)
                self._breaker.record_failure(provider.name)
                logger.exception("LLM provider failed", provider=provider.name, error=str(exc))
                self._schedule_task(
                    self._log_llm_call(
                        analysis_id=analysis_id,
                        provider=provider.name,
                        model=resolved_model,
                        prompt=prompt,
                        response_text="",
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        cache_hit=False,
                        latency_ms=latency_ms,
                        error=str(exc),
                        metadata={"response_type": "structured"},
                    )
                )
                continue
        return None

    async def _run_with_retries(self, func, prompt: str, **kwargs: Any):
        retryer = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(settings.LLM_RETRY_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=settings.LLM_RETRY_BACKOFF_SECONDS, min=1, max=8),
            retry=retry_if_exception_type(Exception),
        )
        async for attempt in retryer:
            with attempt:
                return await func(prompt, **kwargs)

    def _log_usage(self, provider: str, response: LLMResponse, cache_hit: bool, model: str | None) -> None:
        if (
            response.total_tokens is None
            and response.prompt_tokens is None
            and response.completion_tokens is None
        ):
            return
        total_tokens = response.total_tokens
        if total_tokens is None and response.prompt_tokens is not None and response.completion_tokens is not None:
            total_tokens = response.prompt_tokens + response.completion_tokens
        logger.info(
            "LLM usage",
            provider=provider,
            model=model,
            cache_hit=cache_hit,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=total_tokens,
        )

    def _classify_complexity(self, prompt: str) -> PromptComplexity | None:
        if not settings.LLM_COST_BASED_ROUTING_ENABLED:
            return None
        return classify_prompt_complexity(prompt)

    def _select_model(self, provider: str, complexity: PromptComplexity | None) -> str | None:
        if complexity is None:
            return None
        return select_model_for_complexity(complexity, provider)

    def _resolve_model(
        self,
        provider: LLMProvider,
        model_name: str | None,
        provider_kwargs: dict[str, Any],
    ) -> str:
        return model_name or provider_kwargs.get("model") or getattr(provider, "_model_name", provider.name)

    def _prepare_kwargs(
        self,
        kwargs: dict[str, Any],
        model_name: str | None,
        response_type: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_kwargs = dict(kwargs)
        base_kwargs.pop("analysis_id", None)
        provider_kwargs = dict(base_kwargs)
        provider_kwargs.pop("response_type", None)
        if model_name:
            provider_kwargs["model"] = model_name
        cache_kwargs = {"response_type": response_type}
        return cache_kwargs, provider_kwargs

    async def _maybe_get_cache(
        self,
        prompt: str,
        model: str,
        cache_kwargs: dict[str, Any],
        provider: str,
    ):
        if not settings.LLM_CACHE_ENABLED:
            return None
        return await self._cache.get_cached_response(prompt, model, provider=provider, **cache_kwargs)

    async def _maybe_set_cache(
        self,
        prompt: str,
        model: str,
        response: LLMResponse | dict,
        cache_kwargs: dict[str, Any],
        provider: str,
    ) -> None:
        if not settings.LLM_CACHE_ENABLED:
            return None
        await self._cache.set_cached_response(prompt, model, response, provider=provider, **cache_kwargs)
        return None

    def _parse_structured_response(self, provider: str, response: LLMResponse) -> dict[str, Any] | None:
        text = response.text.strip()

        # Strip markdown code blocks (Gemini often wraps JSON in ```json ... ```)
        if text.startswith("```"):
            # Find the end of the first line (```json or just ```)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove trailing ```
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "LLM provider returned non-JSON response",
                provider=provider,
                response_preview=text[:200] if text else "(empty)",
            )
            return None

    def _token_metrics(
        self,
        prompt: str,
        response_text: str,
        response: LLMResponse | None,
    ) -> tuple[int | None, int | None, int | None, bool]:
        prompt_tokens = response.prompt_tokens if response else None
        completion_tokens = response.completion_tokens if response else None
        total_tokens = response.total_tokens if response else None
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        if total_tokens is None:
            estimated_tokens = quota_manager.estimate_tokens_from_text(prompt + response_text)
            return prompt_tokens, completion_tokens, estimated_tokens, True
        return prompt_tokens, completion_tokens, total_tokens, False

    def _latency_ms(self, start_time: float) -> int:
        return int((time.monotonic() - start_time) * 1000)

    def _schedule_task(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._log_task_result)

    def _log_task_result(self, task: asyncio.Task) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.warning("Background task failed", error=str(exc))

    async def _log_llm_call(
        self,
        *,
        analysis_id: str | uuid.UUID | None,
        provider: str,
        model: str,
        prompt: str,
        response_text: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        cache_hit: bool,
        latency_ms: int | None,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        analysis_uuid = None
        if analysis_id:
            try:
                analysis_uuid = uuid.UUID(str(analysis_id))
            except ValueError:
                logger.warning("Invalid analysis_id for LLM log", analysis_id=str(analysis_id))
        estimated_cost = 0.0
        if total_tokens is not None and not cache_hit:
            estimated_cost = quota_manager.calculate_cost(total_tokens, model)
        async with get_session_context() as session:
            llm_log = LLMLog(
                analysis_id=analysis_uuid,
                provider=provider,
                model=model,
                prompt=prompt,
                response=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                error=error,
                metadata=metadata,
            )
            session.add(llm_log)
            await session.flush()

    async def _track_llm_usage(
        self,
        *,
        analysis_id: str | uuid.UUID | None,
        tokens: int | None,
        model: str,
        cache_hit: bool,
    ) -> None:
        if cache_hit or analysis_id is None or tokens is None:
            return None
        try:
            analysis_uuid = uuid.UUID(str(analysis_id))
        except ValueError:
            logger.warning("Invalid analysis_id for usage tracking", analysis_id=str(analysis_id))
            return None
        async with get_session_context() as session:
            await quota_manager.track_llm_usage(session, analysis_uuid, tokens, model)
        return None


async def complete_prompt(prompt: str, **kwargs: Any) -> LLMResponse | None:
    """Convenience helper for a shared router instance."""
    return await ROUTER.complete(prompt, **kwargs)


async def structured_prompt(prompt: str, **kwargs: Any) -> dict[str, Any] | None:
    """Convenience helper for a shared router instance."""
    return await ROUTER.structured_output(prompt, **kwargs)


ROUTER = LLMRouter()
