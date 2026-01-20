"""Tests for enhanced LLM router behavior."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.llm import router as router_module
from app.llm.router import LLMRouter
from app.llm.types import LLMResponse
from app.models.llm_log import LLMLog


class FakeProvider:
    name = "openai"

    def __init__(self, text: str = "ok") -> None:
        self._text = text
        self.calls: list[dict] = []

    async def complete(self, prompt: str, **kwargs):  # noqa: ARG002 - test stub
        self.calls.append(dict(kwargs))
        return LLMResponse(
            text=self._text,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


class FakeCache:
    def __init__(self, response=None) -> None:
        self.response = response
        self.get_calls = []
        self.set_calls = []

    async def get_cached_response(self, prompt: str, model: str, **kwargs):
        self.get_calls.append((prompt, model, kwargs))
        return self.response

    async def set_cached_response(self, prompt: str, model: str, response, **kwargs):
        self.set_calls.append((prompt, model, response, kwargs))


@pytest.mark.asyncio
async def test_router_cache_hit_returns_cached_response(monkeypatch):
    router = LLMRouter()
    fake_provider = FakeProvider()
    cached_response = LLMResponse(text="cached", prompt_tokens=1, completion_tokens=1, total_tokens=2)

    router._cache = FakeCache(response=cached_response)
    router._ordered_providers = lambda: [fake_provider]
    monkeypatch.setattr(router._breaker, "is_available", lambda *_: True)
    monkeypatch.setattr(router._breaker, "record_success", lambda *_: None)
    monkeypatch.setattr(router._breaker, "record_failure", lambda *_: None)
    monkeypatch.setattr(router, "_schedule_task", lambda *_: None)
    monkeypatch.setattr(router, "_maybe_set_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(router_module.settings, "LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(router_module.settings, "LLM_COST_BASED_ROUTING_ENABLED", False)

    response = await router.complete("prompt")

    assert response == cached_response
    assert fake_provider.calls == []


@pytest.mark.asyncio
async def test_router_cache_miss_calls_provider_and_sets_cache(monkeypatch):
    router = LLMRouter()
    fake_provider = FakeProvider()
    router._cache = FakeCache(response=None)
    router._ordered_providers = lambda: [fake_provider]
    monkeypatch.setattr(router._breaker, "is_available", lambda *_: True)
    monkeypatch.setattr(router._breaker, "record_success", lambda *_: None)
    monkeypatch.setattr(router._breaker, "record_failure", lambda *_: None)
    monkeypatch.setattr(router, "_schedule_task", lambda *_: None)
    monkeypatch.setattr(router_module.settings, "LLM_CACHE_ENABLED", True)
    monkeypatch.setattr(router_module.settings, "LLM_COST_BASED_ROUTING_ENABLED", False)

    response = await router.complete("prompt")

    assert response is not None
    assert fake_provider.calls
    assert router._cache.set_calls


@pytest.mark.asyncio
async def test_router_cost_based_model_selection(monkeypatch):
    router = LLMRouter()
    fake_provider = FakeProvider()
    router._cache = FakeCache(response=None)
    router._ordered_providers = lambda: [fake_provider]
    monkeypatch.setattr(router._breaker, "is_available", lambda *_: True)
    monkeypatch.setattr(router._breaker, "record_success", lambda *_: None)
    monkeypatch.setattr(router._breaker, "record_failure", lambda *_: None)
    monkeypatch.setattr(router, "_schedule_task", lambda *_: None)
    monkeypatch.setattr(router_module.settings, "LLM_CACHE_ENABLED", False)
    monkeypatch.setattr(router_module.settings, "LLM_COST_BASED_ROUTING_ENABLED", True)

    await router.complete("Hello")

    assert fake_provider.calls[0]["model"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_router_cost_routing_disabled(monkeypatch):
    router = LLMRouter()
    fake_provider = FakeProvider()
    router._cache = FakeCache(response=None)
    router._ordered_providers = lambda: [fake_provider]
    monkeypatch.setattr(router._breaker, "is_available", lambda *_: True)
    monkeypatch.setattr(router._breaker, "record_success", lambda *_: None)
    monkeypatch.setattr(router._breaker, "record_failure", lambda *_: None)
    monkeypatch.setattr(router, "_schedule_task", lambda *_: None)
    monkeypatch.setattr(router_module.settings, "LLM_CACHE_ENABLED", False)
    monkeypatch.setattr(router_module.settings, "LLM_COST_BASED_ROUTING_ENABLED", False)

    await router.complete("Hello")

    assert "model" not in fake_provider.calls[0]


@pytest.mark.asyncio
async def test_router_creates_llm_log(db_session, monkeypatch):
    router = LLMRouter()
    fake_provider = FakeProvider()
    router._cache = FakeCache(response=None)
    router._ordered_providers = lambda: [fake_provider]

    @asynccontextmanager
    async def session_context():
        yield db_session

    monkeypatch.setattr(router_module, "get_session_context", session_context)
    monkeypatch.setattr(router._breaker, "is_available", lambda *_: True)
    monkeypatch.setattr(router._breaker, "record_success", lambda *_: None)
    monkeypatch.setattr(router._breaker, "record_failure", lambda *_: None)
    monkeypatch.setattr(router_module.settings, "LLM_CACHE_ENABLED", False)
    monkeypatch.setattr(router_module.settings, "LLM_COST_BASED_ROUTING_ENABLED", False)

    await router.complete("prompt")
    await asyncio.sleep(0.01)

    result = await db_session.execute(select(LLMLog))
    logs = result.scalars().all()
    assert logs
    assert logs[0].prompt == "prompt"
