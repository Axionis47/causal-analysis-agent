"""Redis-backed cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import log_llm_cache_event
from app.llm.types import LLMResponse
from app.services.progress import get_redis

_CACHE_HITS_KEY = "llm_cache:hits"
_CACHE_MISSES_KEY = "llm_cache:misses"
_CACHE_ENTRY_PREFIX = "llm_cache:entry:"


class LLMCacheService:
    """Service for caching LLM responses in Redis."""

    async def get_cached_response(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> LLMResponse | dict | None:
        if not settings.LLM_CACHE_ENABLED:
            return None
        if len(prompt) > settings.LLM_CACHE_MAX_PROMPT_LENGTH:
            return None

        response_type = kwargs.get("response_type")
        cache_key = self._generate_cache_key(prompt, model, response_type=response_type)
        cache_key_hash = cache_key.split(":")[-1]
        provider_name = kwargs.get("provider", model)
        resolved_model = kwargs.get("model", model)
        redis: Redis | None = None
        try:
            redis = await get_redis()
            payload = await redis.get(cache_key)
            if payload:
                await redis.incr(_CACHE_HITS_KEY)
                log_llm_cache_event(
                    "llm.cache.hit",
                    cache_key_hash=cache_key_hash,
                    prompt_length=len(prompt),
                    model=resolved_model,
                    provider=provider_name,
                    ttl=None,
                )
                return self._deserialize_response(payload)
            await redis.incr(_CACHE_MISSES_KEY)
            log_llm_cache_event(
                "llm.cache.miss",
                cache_key_hash=cache_key_hash,
                prompt_length=len(prompt),
                model=resolved_model,
                provider=provider_name,
                ttl=None,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - cache failures should not block
            log_llm_cache_event(
                "llm.cache.error",
                cache_key_hash=cache_key_hash,
                prompt_length=len(prompt),
                model=resolved_model,
                provider=provider_name,
                ttl=None,
                error=str(exc),
            )
            return None
        finally:
            if redis is not None:
                await redis.close()

    async def set_cached_response(
        self,
        prompt: str,
        model: str,
        response: LLMResponse | dict,
        **kwargs: Any,
    ) -> None:
        if not settings.LLM_CACHE_ENABLED:
            return None
        if len(prompt) > settings.LLM_CACHE_MAX_PROMPT_LENGTH:
            return None

        ttl = int(kwargs.pop("ttl", settings.LLM_CACHE_TTL_SECONDS))
        response_type = kwargs.get("response_type")
        cache_key = self._generate_cache_key(prompt, model, response_type=response_type)
        cache_key_hash = cache_key.split(":")[-1]
        provider_name = kwargs.get("provider", model)
        resolved_model = kwargs.get("model", model)
        payload = self._serialize_response(response)
        redis: Redis | None = None
        try:
            redis = await get_redis()
            await redis.setex(cache_key, ttl, payload)
            log_llm_cache_event(
                "llm.cache.set",
                cache_key_hash=cache_key_hash,
                prompt_length=len(prompt),
                model=resolved_model,
                provider=provider_name,
                ttl=ttl,
            )
        except Exception as exc:  # noqa: BLE001 - cache failures should not block
            log_llm_cache_event(
                "llm.cache.error",
                cache_key_hash=cache_key_hash,
                prompt_length=len(prompt),
                model=resolved_model,
                provider=provider_name,
                ttl=ttl,
                error=str(exc),
            )
        finally:
            if redis is not None:
                await redis.close()
        return None

    def _generate_cache_key(self, prompt: str, model: str, **kwargs: Any) -> str:
        response_type = kwargs.get("response_type")
        payload = {
            "prompt": prompt,
            "model": model,
        }
        if response_type:
            payload["response_type"] = response_type
        cache_material = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
        digest = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
        return f"{_CACHE_ENTRY_PREFIX}{digest}"

    async def invalidate_cache(self, pattern: str | None = None) -> int:
        redis: Redis | None = None
        pattern = pattern or f"{_CACHE_ENTRY_PREFIX}*"
        deleted = 0
        try:
            redis = await get_redis()
            keys = []
            async for key in redis.scan_iter(match=pattern):
                if key in {_CACHE_HITS_KEY, _CACHE_MISSES_KEY}:
                    continue
                keys.append(key)
            if keys:
                deleted = await redis.delete(*keys)
        except Exception as exc:  # noqa: BLE001 - cache failures should not block
            log_llm_cache_event(
                "llm.cache.error",
                cache_key_hash="",
                prompt_length=0,
                model="",
                provider="",
                ttl=None,
                error=str(exc),
            )
        finally:
            if redis is not None:
                await redis.close()
        return int(deleted or 0)

    async def get_cache_stats(self) -> dict[str, int]:
        redis: Redis | None = None
        try:
            redis = await get_redis()
            hits = await redis.get(_CACHE_HITS_KEY)
            misses = await redis.get(_CACHE_MISSES_KEY)
            return {
                "hits": int(hits or 0),
                "misses": int(misses or 0),
            }
        except Exception:  # noqa: BLE001 - cache failures should not block
            return {"hits": 0, "misses": 0}
        finally:
            if redis is not None:
                await redis.close()

    def _serialize_response(self, response: LLMResponse | dict) -> str:
        cached_at = datetime.now(timezone.utc).isoformat()
        if isinstance(response, LLMResponse):
            payload = {
                "response_type": "complete",
                "text": response.text,
                "tokens": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                },
                "cached_at": cached_at,
            }
        else:
            payload = {
                "response_type": "structured",
                "text": json.dumps(response, ensure_ascii=True),
                "tokens": {},
                "cached_at": cached_at,
            }
        return json.dumps(payload, ensure_ascii=True)

    def _deserialize_response(self, payload: str) -> LLMResponse | dict | None:
        data = json.loads(payload)
        response_type = data.get("response_type")
        text = data.get("text") or ""
        if response_type == "structured":
            try:
                return json.loads(text) if text else {}
            except json.JSONDecodeError:
                return None
        tokens = data.get("tokens") or {}
        return LLMResponse(
            text=text,
            raw=None,
            prompt_tokens=tokens.get("prompt_tokens"),
            completion_tokens=tokens.get("completion_tokens"),
            total_tokens=tokens.get("total_tokens"),
        )


llm_cache_service = LLMCacheService()
