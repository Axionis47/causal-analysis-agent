"""Tests for LLM cache service."""

from __future__ import annotations

import fnmatch
from unittest.mock import AsyncMock

import pytest

from app.llm.types import LLMResponse
from app.services import llm_cache as llm_cache_module


class FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._expiry: dict[str, float] = {}
        self._now = 0.0

    def advance(self, seconds: float) -> None:
        self._now += seconds

    async def get(self, key: str):
        if key in self._expiry and self._now >= self._expiry[key]:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return None
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._store[key] = value
        self._expiry[key] = self._now + ttl
        return True

    async def incr(self, key: str):
        current = int(self._store.get(key, "0"))
        current += 1
        self._store[key] = str(current)
        return current

    async def scan_iter(self, match: str | None = None):
        for key in list(self._store.keys()):
            if match is None or fnmatch.fnmatch(key, match):
                yield key

    async def delete(self, *keys: str):
        deleted = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                deleted += 1
        return deleted

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_cache_key_generation_stable():
    service = llm_cache_module.LLMCacheService()
    key1 = service._generate_cache_key(
        "hello",
        "openai",
        response_type="complete",
        temperature=0.2,
    )
    key2 = service._generate_cache_key(
        "hello",
        "openai",
        temperature=0.2,
        response_type="complete",
    )
    assert key1 == key2


@pytest.mark.asyncio
async def test_cache_hit_and_miss(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(llm_cache_module, "get_redis", AsyncMock(return_value=fake_redis))
    service = llm_cache_module.LLMCacheService()

    miss = await service.get_cached_response("prompt", "openai", response_type="complete")
    assert miss is None

    response = LLMResponse(text="cached", prompt_tokens=1, completion_tokens=2, total_tokens=3)
    await service.set_cached_response("prompt", "openai", response, response_type="complete")

    hit = await service.get_cached_response("prompt", "openai", response_type="complete")
    assert isinstance(hit, LLMResponse)
    assert hit.text == "cached"
    assert hit.total_tokens == 3

    stats = await service.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_cache_ttl_expiration(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(llm_cache_module, "get_redis", AsyncMock(return_value=fake_redis))
    service = llm_cache_module.LLMCacheService()

    response = LLMResponse(text="cached", prompt_tokens=1, completion_tokens=1, total_tokens=2)
    await service.set_cached_response(
        "prompt",
        "openai",
        response,
        response_type="complete",
        ttl=1,
    )

    fake_redis.advance(2)
    expired = await service.get_cached_response("prompt", "openai", response_type="complete")
    assert expired is None


@pytest.mark.asyncio
async def test_cache_stats(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(llm_cache_module, "get_redis", AsyncMock(return_value=fake_redis))
    service = llm_cache_module.LLMCacheService()

    await service.get_cached_response("prompt", "openai", response_type="complete")
    stats = await service.get_cache_stats()
    assert stats == {"hits": 0, "misses": 1}
