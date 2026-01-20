"""Tests for structured logging and observability helpers."""

import asyncio
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.logging import bind_contextvars, capture_exception, get_logger
from app.main import app


@pytest.fixture(autouse=True)
def clear_log_context():
    """Ensure log context does not leak between tests."""
    bind_contextvars(request_id="", user_id="", analysis_id="")
    yield
    bind_contextvars(request_id="", user_id="", analysis_id="")


def _last_log_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    assert lines
    return lines[-1]


def test_structured_logger_creates_json_output(capsys):
    logger = get_logger("test.logging")
    logger.info("hello", foo="bar")

    captured = capsys.readouterr()
    output = captured.err or captured.out
    data = json.loads(_last_log_line(output))

    assert data["event"] == "hello"
    assert data["foo"] == "bar"
    assert "timestamp" in data


def test_sensitive_data_is_redacted(capsys):
    logger = get_logger("test.redaction")
    logger.info(
        "secret",
        api_key="abc",
        payload={"password": "pw", "token": "t"},
        nested=[{"authorization": "Bearer token"}],
    )

    captured = capsys.readouterr()
    output = captured.err or captured.out
    data = json.loads(_last_log_line(output))

    assert data["api_key"] == "***REDACTED***"
    assert data["payload"]["password"] == "***REDACTED***"
    assert data["payload"]["token"] == "***REDACTED***"
    assert data["nested"][0]["authorization"] == "***REDACTED***"


def test_context_binding_works(capsys):
    bind_contextvars(request_id="req-1", user_id="user-1", analysis_id="analysis-1")
    logger = get_logger("test.context")
    logger.info("context")

    captured = capsys.readouterr()
    output = captured.err or captured.out
    data = json.loads(_last_log_line(output))

    assert data["request_id"] == "req-1"
    assert data["user_id"] == "user-1"
    assert data["analysis_id"] == "analysis-1"


@pytest.mark.asyncio
async def test_correlation_id_propagates(capsys):
    bind_contextvars(request_id="req-async")
    logger = get_logger("test.async")
    capsys.readouterr()

    async def inner():
        logger.info("async")

    await asyncio.create_task(inner())

    captured = capsys.readouterr()
    output = captured.err or captured.out
    data = json.loads(_last_log_line(output))

    assert data["request_id"] == "req-async"


def test_sentry_captures_exceptions():
    bind_contextvars(request_id="req-2", user_id="user-2", analysis_id="analysis-2")

    with patch("app.core.logging.sentry_sdk.capture_exception") as mock_capture:
        with patch("app.core.logging.sentry_sdk.push_scope") as mock_scope:
            scope = mock_scope.return_value.__enter__.return_value
            try:
                raise ValueError("boom")
            except Exception as exc:  # noqa: BLE001
                capture_exception(exc)

    mock_capture.assert_called_once()
    scope.set_tag.assert_any_call("request_id", "req-2")
    scope.set_tag.assert_any_call("user_id", "user-2")
    scope.set_tag.assert_any_call("analysis_id", "analysis-2")


@pytest.mark.asyncio
async def test_request_id_middleware_adds_header():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]
