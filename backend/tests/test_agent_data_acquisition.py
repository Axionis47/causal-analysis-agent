"""Unit tests for DataAcquisitionAgent."""

from __future__ import annotations

import hashlib
import uuid

import pandas as pd
import pytest

from app.agents import data_acquisition as acquisition
from app.agents.data_acquisition import (
    DataAcquisitionAgent,
    KAGGLE_BREAKER_KEY,
    SUPPORTED_EXTENSIONS,
    _parse_kaggle_dataset_id,
    _sample_if_needed,
    _select_file,
)
from app.services.circuit_breaker import CircuitBreaker
from tests.fixtures.agent_fixtures import create_analysis, create_dataset
from tests.fixtures.mock_helpers import create_mock_dataframe, make_session_context


@pytest.mark.asyncio
async def test_data_acquisition_cache_hit_state():
    state = {
        "analysis_id": str(uuid.uuid4()),
        "kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset",
        "dataset_id": str(uuid.uuid4()),
    }

    result = await DataAcquisitionAgent()._run(state)

    assert result.outputs["cached"] is True


@pytest.mark.asyncio
async def test_data_acquisition_db_cache_hit(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    cached = await create_analysis(
        db_session,
        kaggle_url="https://www.kaggle.com/datasets/test/other",
    )
    kaggle_url = "https://www.kaggle.com/datasets/test/sample-dataset"
    cache_key = hashlib.sha256(kaggle_url.encode()).hexdigest()
    await create_dataset(
        db_session,
        analysis_id=cached.id,
        kaggle_url=kaggle_url,
        cache_key=cache_key,
    )

    state = {
        "analysis_id": str(analysis.id),
        "kaggle_url": kaggle_url,
    }

    monkeypatch.setattr(acquisition, "get_session_context", make_session_context(db_session))

    result = await DataAcquisitionAgent()._run(state)

    assert result.outputs["cached"] is True
    assert state["dataset_id"]


@pytest.mark.asyncio
async def test_data_acquisition_fresh_download_success(db_session, monkeypatch, tmp_path):
    analysis = await create_analysis(db_session)
    state = {
        "analysis_id": str(analysis.id),
        "kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset",
    }
    download_root = tmp_path / "download"
    download_root.mkdir()
    (download_root / "small.csv").write_text("a\n1\n")
    (download_root / "large.csv").write_text("a\n" + "1\n" * 50)

    async def fake_retry_async(func, *args, **kwargs):
        kwargs.pop("retry_on", None)
        return await func(*args, **kwargs)

    async def fake_credentials(*_args, **_kwargs):
        return ("user", "key")

    async def fake_download(*_args, **_kwargs):
        return download_root

    async def fake_upload_file(path, bucket_name, destination):  # noqa: ARG001
        return f"gs://test-bucket/{destination}"

    monkeypatch.setattr(acquisition, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(acquisition, "get_kaggle_credentials", fake_credentials)
    monkeypatch.setattr(acquisition, "_download_with_retry", fake_download)
    monkeypatch.setattr(
        acquisition,
        "_load_dataframe",
        lambda *_: create_mock_dataframe(rows=10),
    )
    monkeypatch.setattr(
        acquisition,
        "_sample_if_needed",
        lambda df: pd.DataFrame({"x": range(1_000_000)}),
    )
    monkeypatch.setattr(acquisition, "retry_async", fake_retry_async)
    monkeypatch.setattr(acquisition, "upload_file", fake_upload_file)
    monkeypatch.setattr(acquisition, "local_storage_root", lambda: tmp_path)
    monkeypatch.setattr(
        acquisition,
        "_write_dataframe",
        lambda df, path: path.write_text("data"),
    )
    monkeypatch.setattr(
        acquisition,
        "KAGGLE_BREAKER",
        CircuitBreaker(threshold=3, timeout_seconds=60),
    )

    result = await DataAcquisitionAgent()._run(state)

    assert result.outputs["file"] == "large.csv"
    assert state["dataset_id"]


def test_select_file_picks_largest_supported(tmp_path):
    sizes = {"a.csv": 10, "b.parquet": 50, "c.xlsx": 30}
    for name, size in sizes.items():
        (tmp_path / name).write_text("x" * size)
    selected = _select_file(tmp_path)

    assert selected is not None
    assert selected.name == "b.parquet"
    assert selected.suffix in SUPPORTED_EXTENSIONS


def test_sample_if_needed_stratified_sampling():
    rows = 1_000_002
    df = pd.DataFrame(
        {
            "group": ["A"] * 501_001 + ["B"] * 501_001,
            "value": list(range(rows)),
        }
    )

    sampled = _sample_if_needed(df)

    assert len(sampled) == 1_000_000
    counts = sampled["group"].value_counts().to_dict()
    assert counts["A"] == 500_000
    assert counts["B"] == 500_000


def test_sample_if_needed_random_sampling():
    rows = 1_000_010
    df = pd.DataFrame({"value": list(range(rows)), "other": list(range(rows))})

    sampled = _sample_if_needed(df)

    assert len(sampled) == 1_000_000


@pytest.mark.parametrize(
    "url,expected_id,is_competition",
    [
        ("https://www.kaggle.com/datasets/test/sample-dataset", "test/sample-dataset", False),
        ("https://www.kaggle.com/competitions/sample-competition", "sample-competition", True),
    ],
)
def test_kaggle_url_parsing(url, expected_id, is_competition):
    dataset_id, competition_flag = _parse_kaggle_dataset_id(url)

    assert dataset_id == expected_id
    assert competition_flag is is_competition


@pytest.mark.asyncio
async def test_data_acquisition_circuit_breaker_open(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    state = {"analysis_id": str(analysis.id), "kaggle_url": analysis.kaggle_url}
    breaker = CircuitBreaker(threshold=1, timeout_seconds=60)
    breaker.record_failure(KAGGLE_BREAKER_KEY)

    monkeypatch.setattr(acquisition, "get_session_context", make_session_context(db_session))

    async def fake_credentials(*_args, **_kwargs):
        return ("user", "key")

    monkeypatch.setattr(acquisition, "get_kaggle_credentials", fake_credentials)
    monkeypatch.setattr(acquisition, "KAGGLE_BREAKER", breaker)

    with pytest.raises(RuntimeError, match="circuit breaker is open"):
        await DataAcquisitionAgent()._run(state)

    partial = state.get(acquisition.PARTIAL_OUTPUTS_KEY)
    assert partial
    assert partial["circuit_breaker"] == "open"
    assert partial["retry_after_seconds"] is not None


@pytest.mark.asyncio
async def test_data_acquisition_retry_on_transient_errors(monkeypatch, tmp_path):
    attempts = {"count": 0, "backoff": None, "max_attempts": None}

    async def flaky_download(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            exc = RuntimeError("service unavailable")
            exc.status_code = 503
            raise exc
        return tmp_path

    async def fake_retry_async(func, *args, **kwargs):
        attempts["backoff"] = kwargs.get("backoff_seconds")
        attempts["max_attempts"] = kwargs.get("max_attempts")
        max_attempts = attempts["max_attempts"] or 1
        last_exc = None
        for _ in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - test retry loop
                last_exc = exc
                if kwargs.get("retry_on") and not kwargs["retry_on"](exc):
                    raise
        raise last_exc

    monkeypatch.setattr(acquisition, "_download_dataset", flaky_download)
    monkeypatch.setattr(acquisition, "retry_async", fake_retry_async)
    monkeypatch.setattr(
        acquisition,
        "KAGGLE_BREAKER",
        CircuitBreaker(threshold=3, timeout_seconds=60),
    )

    result = await acquisition._download_with_retry("dataset", "user", "key", False)

    assert result == tmp_path
    assert attempts["count"] == 3
    assert attempts["max_attempts"] == acquisition.settings.KAGGLE_RETRY_MAX_ATTEMPTS
    assert attempts["backoff"] == acquisition.settings.KAGGLE_RETRY_BACKOFF_SECONDS


@pytest.mark.asyncio
async def test_data_acquisition_unsupported_file_format(db_session, monkeypatch, tmp_path):
    analysis = await create_analysis(db_session)
    state = {"analysis_id": str(analysis.id), "kaggle_url": analysis.kaggle_url}

    download_root = tmp_path / "download"
    download_root.mkdir()
    (download_root / "data.txt").write_text("nope")

    monkeypatch.setattr(acquisition, "get_session_context", make_session_context(db_session))
    async def fake_credentials(*_args, **_kwargs):
        return ("user", "key")

    async def fake_download(*_args, **_kwargs):
        return download_root

    monkeypatch.setattr(acquisition, "get_kaggle_credentials", fake_credentials)
    monkeypatch.setattr(acquisition, "_download_with_retry", fake_download)
    monkeypatch.setattr(
        acquisition,
        "KAGGLE_BREAKER",
        CircuitBreaker(threshold=3, timeout_seconds=60),
    )

    with pytest.raises(ValueError, match="No supported dataset files"):
        await DataAcquisitionAgent()._run(state)


@pytest.mark.asyncio
async def test_data_acquisition_missing_credentials(db_session, monkeypatch):
    analysis = await create_analysis(db_session)
    state = {"analysis_id": str(analysis.id), "kaggle_url": analysis.kaggle_url}

    monkeypatch.setattr(acquisition, "get_session_context", make_session_context(db_session))
    async def fake_credentials(*_args, **_kwargs):
        return None

    monkeypatch.setattr(acquisition, "get_kaggle_credentials", fake_credentials)

    with pytest.raises(ValueError, match="credentials"):
        await DataAcquisitionAgent()._run(state)


def test_select_file_returns_none_when_unsupported(tmp_path):
    (tmp_path / "data.txt").write_text("nope")

    assert _select_file(tmp_path) is None
