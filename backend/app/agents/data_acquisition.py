"""Data acquisition agent for Kaggle downloads."""

from __future__ import annotations

import hashlib
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

from app.agents.base import AgentResult, BaseAgent, PARTIAL_OUTPUTS_KEY
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.dataset import dataset_crud
from app.db.database import get_session_context
from app.models.analysis_stage import StageType
from app.services.credentials import get_kaggle_credentials
from app.services.circuit_breaker import CircuitBreaker
from app.services.retry import is_transient_io_error, is_transient_kaggle_error, retry_async
from app.services.storage import ensure_local_dir, local_storage_root, upload_file
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}
logger = get_logger(__name__)
KAGGLE_BREAKER_KEY = "kaggle"
KAGGLE_BREAKER = CircuitBreaker(
    failure_threshold=settings.KAGGLE_CIRCUIT_BREAKER_THRESHOLD,
    recovery_timeout_seconds=settings.KAGGLE_CIRCUIT_BREAKER_TIMEOUT_SECONDS,
)


class DataAcquisitionAgent(BaseAgent):
    name = "Data Acquisition Agent"
    stage = StageType.DOWNLOAD

    async def _run(self, state):
        if state.get("dataset_id"):
            return AgentResult(state=state, outputs={"cached": True}, message="Dataset cached")

        analysis_id = state["analysis_id"]
        agent_logger = logger.bind(
            analysis_id=analysis_id,
            stage=self.stage.value,
            agent_name=self.name,
        )
        kaggle_url = state["kaggle_url"]
        cache_key = hashlib.sha256(kaggle_url.encode()).hexdigest()
        analysis_uuid = uuid.UUID(analysis_id)

        async with get_session_context() as session:
            existing = await dataset_crud.get_by_analysis(session, analysis_uuid)
            if existing:
                state["dataset_id"] = str(existing[0].id)
                return AgentResult(
                    state=state,
                    outputs={"dataset_id": str(existing[0].id), "cached": True},
                    message="Dataset already registered for analysis",
                )
            cached = await dataset_crud.get_cached(session, cache_key)
            if cached is not None:
                dataset = await dataset_crud.create(
                    session,
                    obj_in={
                        "analysis_id": analysis_uuid,
                        "kaggle_url": cached.kaggle_url,
                        "kaggle_dataset_id": cached.kaggle_dataset_id,
                        "files": cached.files,
                        "selected_file": cached.selected_file,
                        "selection_reasoning": "Reused cached dataset",
                        "gcs_path": cached.gcs_path,
                        "cache_key": cache_key,
                        "cache_expires_at": cached.cache_expires_at,
                        "downloaded_at": cached.downloaded_at,
                        "dataset_metadata": cached.dataset_metadata,
                        "characteristics": cached.characteristics,
                    },
                )
                state["dataset_id"] = str(dataset.id)
                return AgentResult(
                    state=state,
                    outputs={"dataset_id": str(dataset.id), "cached": True},
                    message="Reused cached dataset",
                )

        dataset_id, is_competition = _parse_kaggle_dataset_id(kaggle_url)
        if dataset_id is None:
            raise ValueError("Unable to parse Kaggle dataset ID")

        async with get_session_context() as session:
            credentials = await get_kaggle_credentials(session, None)
        if credentials is None:
            raise ValueError("Kaggle credentials not configured")

        username, api_key = credentials
        if not KAGGLE_BREAKER.is_available(KAGGLE_BREAKER_KEY):
            retry_after = KAGGLE_BREAKER.retry_after_seconds(KAGGLE_BREAKER_KEY)
            message = "Kaggle circuit breaker is open; skipping download until it resets"
            agent_logger.warning(
                message,
                retry_after_seconds=retry_after,
                breaker_key=KAGGLE_BREAKER_KEY,
            )
            state[PARTIAL_OUTPUTS_KEY] = {
                "failure_message": message,
                "retry_after_seconds": retry_after,
                "circuit_breaker": "open",
            }
            raise RuntimeError(message)
        downloaded_path = await _download_with_retry(
            dataset_id,
            username,
            api_key,
            is_competition,
            analysis_id=analysis_id,
            agent_name=self.name,
            stage=self.stage.value,
        )
        selected_file = _select_file(downloaded_path)
        if selected_file is None:
            raise ValueError("No supported dataset files found")

        df = _load_dataframe(selected_file)
        df = _sample_if_needed(df)
        agent_logger.info(
            "Selected dataset file",
            selected_file=selected_file.name,
            file_size=selected_file.stat().st_size,
            row_count=len(df),
        )

        local_root = local_storage_root() / analysis_id / "raw"
        ensure_local_dir(local_root)
        output_path = local_root / selected_file.name
        _write_dataframe(df, output_path)

        gcs_path = await retry_async(
            upload_file,
            output_path,
            settings.GCS_BUCKET_NAME,
            f"{analysis_id}/raw/{selected_file.name}",
            retry_on=is_transient_io_error,
        )

        metadata = {
            "row_count": len(df),
            "column_count": df.shape[1],
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "local_path": str(output_path),
        }

        files = [
            {
                "name": selected_file.name,
                "size_bytes": selected_file.stat().st_size,
                "format": selected_file.suffix.lstrip("."),
                "row_count": len(df),
                "column_count": df.shape[1],
            }
        ]

        async with get_session_context() as session:
            dataset = await dataset_crud.create(
                session,
                obj_in={
                    "analysis_id": analysis_uuid,
                    "kaggle_url": kaggle_url,
                    "kaggle_dataset_id": dataset_id,
                    "files": files,
                    "selected_file": selected_file.name,
                    "selection_reasoning": "Largest supported file selected automatically",
                    "gcs_path": gcs_path,
                    "cache_key": cache_key,
                    "cache_expires_at": datetime.now(timezone.utc) + timedelta(days=7),
                    "downloaded_at": datetime.now(timezone.utc),
                    "metadata": metadata,
                    "characteristics": {},
                },
            )

        state["dataset_id"] = str(dataset.id)
        return AgentResult(
            state=state,
            outputs={"dataset_id": str(dataset.id), "file": selected_file.name},
            message="Dataset downloaded",
        )


def _parse_kaggle_dataset_id(url: str) -> tuple[str | None, bool]:
    if "/datasets/" in url:
        match = re.search(r"kaggle\.com/datasets/([^/]+/[^/]+)", url)
        return (match.group(1), False) if match else (None, False)
    if "/competitions/" in url:
        match = re.search(r"kaggle\.com/competitions/([^/]+)", url)
        return (match.group(1), True) if match else (None, True)
    return (None, False)


async def _download_with_retry(
    dataset_id: str,
    username: str,
    api_key: str,
    is_competition: bool,
    *,
    analysis_id: str,
    agent_name: str,
    stage: str,
) -> Path:
    if not KAGGLE_BREAKER.is_available(KAGGLE_BREAKER_KEY):
        raise RuntimeError("Kaggle API circuit breaker open")
    retryer = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(settings.KAGGLE_RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.KAGGLE_RETRY_BACKOFF_SECONDS,
            min=1,
            max=settings.AGENT_RETRY_MAX_BACKOFF_SECONDS,
        ),
        retry=retry_if_exception(is_transient_kaggle_error),
    )
    try:
        async for attempt in retryer:
            with attempt:
                logger.info(
                    "Kaggle download attempt",
                    analysis_id=analysis_id,
                    stage=stage,
                    agent_name=agent_name,
                    dataset_id=dataset_id,
                    is_competition=is_competition,
                    attempt_number=attempt.retry_state.attempt_number,
                )
                result = await _download_dataset(
                    dataset_id,
                    username,
                    api_key,
                    is_competition,
                )
                KAGGLE_BREAKER.record_success(KAGGLE_BREAKER_KEY)
                return result
    except Exception as exc:  # noqa: BLE001 - retryer controls transient filtering
        if is_transient_kaggle_error(exc):
            KAGGLE_BREAKER.record_failure(KAGGLE_BREAKER_KEY)
        raise
    raise RuntimeError("Kaggle download retry loop exited without a result")


async def _download_dataset(dataset_id: str, username: str, api_key: str, is_competition: bool) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Kaggle SDK not installed") from exc

    import os

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key
    api = KaggleApi()
    api.authenticate()
    temp_dir = Path(tempfile.mkdtemp(prefix="kaggle_dataset_"))
    if is_competition:
        api.competition_download_files(dataset_id, path=str(temp_dir))
        _unzip_competition_files(temp_dir)
    else:
        api.dataset_download_files(dataset_id, path=str(temp_dir), unzip=True)
    return temp_dir


def _unzip_competition_files(temp_dir: Path) -> None:
    import zipfile

    for zip_path in temp_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir)
        zip_path.unlink(missing_ok=True)


def _select_file(root_path: Path) -> Path | None:
    candidates = [
        path for path in root_path.rglob("*") if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type")


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _sample_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 1_000_000:
        return df
    stratify_cols = [col for col in df.columns if df[col].nunique() <= 20]
    if stratify_cols:
        col = stratify_cols[0]
        return (
            df.groupby(col, group_keys=False)
            .apply(lambda group: group.sample(n=max(1, int(1_000_000 / df[col].nunique())), random_state=42))
            .reset_index(drop=True)
        )
    return df.sample(n=1_000_000, random_state=42)
