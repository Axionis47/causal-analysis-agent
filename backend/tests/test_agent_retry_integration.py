"""Integration tests for retry and circuit breaker behavior."""

from __future__ import annotations

import errno
import pytest
import tenacity
import tenacity._asyncio

from app.agents import data_acquisition
from app.agents.data_acquisition import DataAcquisitionAgent, KAGGLE_BREAKER_KEY
from app.agents.llm_utils import call_llm_with_retry
from app.services import circuit_breaker
from app.services.retry import (
    is_transient_io_error,
    is_transient_kaggle_error,
    is_transient_tool_error,
    retry_async,
    retry_sync,
)
from tests.fixtures.agent_fixtures import create_analysis
from tests.fixtures.mock_helpers import make_session_context


@pytest.mark.asyncio
async def test_llm_retry_with_exponential_backoff(monkeypatch):
    attempts = {"count": 0}
    sleep_calls = []

    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("rate limit")
        return {"ok": True}

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(tenacity._asyncio, "sleep", fake_sleep)

    result = await call_llm_with_retry(flaky)

    assert result == {"ok": True}
    assert attempts["count"] == 3
    assert sleep_calls
    assert sleep_calls[0] <= sleep_calls[-1]


@pytest.mark.asyncio
async def test_kaggle_circuit_breaker_open(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    breaker = circuit_breaker.CircuitBreaker(threshold=1, timeout_seconds=60)
    breaker.record_failure(KAGGLE_BREAKER_KEY)

    async def fake_credentials(*_args, **_kwargs):
        return ("user", "key")

    monkeypatch.setattr(data_acquisition, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(data_acquisition, "get_kaggle_credentials", fake_credentials)
    monkeypatch.setattr(data_acquisition, "KAGGLE_BREAKER", breaker)

    state = {"analysis_id": str(analysis.id), "kaggle_url": analysis.kaggle_url}

    with pytest.raises(RuntimeError):
        await DataAcquisitionAgent()._run(state)

    partial = state.get(data_acquisition.PARTIAL_OUTPUTS_KEY)
    assert partial
    assert partial["retry_after_seconds"] is not None


def test_circuit_breaker_recovery(monkeypatch):
    current = {"value": 1000.0}

    def fake_time():
        return current["value"]

    monkeypatch.setattr(circuit_breaker.time, "time", fake_time)

    breaker = circuit_breaker.CircuitBreaker(threshold=1, timeout_seconds=5)
    breaker.record_failure("key")
    assert breaker.is_available("key") is False

    current["value"] += 6
    assert breaker.is_available("key") is True


def test_retry_exhaustion_raises(monkeypatch):
    def always_fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(tenacity.nap, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="boom"):
        retry_sync(always_fail, max_attempts=2, retry_on=lambda exc: True)


def test_transient_error_detection():
    assert is_transient_io_error(TimeoutError("timeout")) is True
    assert is_transient_io_error(OSError(errno.ECONNRESET, "reset")) is True
    assert is_transient_io_error(PermissionError("no")) is False

    err = RuntimeError("tool")
    assert is_transient_tool_error(err) is True

    api_like = RuntimeError("rate limit")
    api_like.status_code = 429
    assert is_transient_kaggle_error(api_like) is True


@pytest.mark.asyncio
async def test_sync_vs_async_retry(monkeypatch):
    sync_calls = {"count": 0}
    async_calls = {"count": 0}

    def flaky_sync():
        sync_calls["count"] += 1
        if sync_calls["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    async def flaky_async():
        async_calls["count"] += 1
        if async_calls["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    monkeypatch.setattr(tenacity.nap, "sleep", lambda *_args, **_kwargs: None)

    result_sync = retry_sync(flaky_sync, max_attempts=3, retry_on=lambda exc: True)
    assert result_sync == "ok"

    async def fake_sleep(duration):
        return None

    monkeypatch.setattr(tenacity._asyncio, "sleep", fake_sleep)
    result_async = await retry_async(flaky_async, max_attempts=3, retry_on=lambda exc: True)

    assert result_async == "ok"
    assert sync_calls["count"] == 2
    assert async_calls["count"] == 2
