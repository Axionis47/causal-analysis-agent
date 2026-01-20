"""Mock utilities for agent tests."""

from __future__ import annotations

from contextlib import asynccontextmanager
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.orchestrator.state import AnalysisState


class MockKaggleApi:
    """Minimal Kaggle API mock that writes fake files to disk."""

    def __init__(
        self,
        dataset_files: dict[str, dict[str, str]] | None = None,
        competition_files: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.dataset_files = dataset_files or {
            "default": {"data.csv": "col\n1\n2\n"}
        }
        self.competition_files = competition_files or {
            "default": {"competition.csv": "col\n3\n4\n"}
        }
        self.authenticated = False
        self.calls: list[tuple[str, str]] = []

    def authenticate(self) -> None:
        self.authenticated = True

    def dataset_download_files(
        self,
        dataset: str,
        path: str,
        unzip: bool = True,  # noqa: ARG002
    ) -> None:
        self.calls.append(("dataset", dataset))
        self._write_files(path, self.dataset_files.get(dataset, self.dataset_files["default"]))

    def competition_download_files(self, dataset: str, path: str) -> None:
        self.calls.append(("competition", dataset))
        self._write_files(
            path,
            self.competition_files.get(dataset, self.competition_files["default"]),
        )

    def _write_files(self, path: str, files: dict[str, str]) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (root / name).write_text(content)


class MockLLMRouter:
    """Return predefined structured outputs for LLM calls."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> None:
        self.responses = responses if responses is not None else []
        self.prompts: list[str] = []

    async def structured_output(self, prompt: str) -> dict[str, Any] | None:
        self.prompts.append(prompt)
        if isinstance(self.responses, list):
            if not self.responses:
                return None
            return self.responses.pop(0)
        return self.responses


class MockGCSClient:
    """Minimal GCS client mock for uploads/downloads."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    async def upload_file(self, local_path: Path, bucket_name: str, destination: str) -> str:
        self.uploads.append((str(local_path), bucket_name, destination))
        return f"gs://{bucket_name}/{destination}"

    async def download_to_path(self, gcs_path: str, local_path: Path) -> Path:
        self.downloads.append((gcs_path, str(local_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("mock")
        return local_path


def create_mock_dataframe(
    *,
    rows: int = 100,
    include_categorical: bool = True,
    include_missing: bool = True,
    include_outliers: bool = False,
    include_datetime: bool = False,
    include_binary: bool = True,
) -> pd.DataFrame:
    """Create a dataframe with configurable characteristics."""
    rng = np.random.default_rng(42)
    data: dict[str, Any] = {}
    if include_binary:
        data["treatment"] = rng.integers(0, 2, size=rows)
    data["outcome"] = rng.normal(loc=0.0, scale=1.0, size=rows)
    data["feature"] = rng.normal(loc=5.0, scale=2.0, size=rows)
    if include_categorical:
        categories = np.where(np.arange(rows) % 2 == 0, "A", "B")
        data["group"] = pd.Series(categories, dtype="category")
    if include_datetime:
        data["event_time"] = pd.date_range("2020-01-01", periods=rows, freq="H")
    df = pd.DataFrame(data)
    if include_missing:
        df.loc[df.index[: max(1, rows // 5)], "feature"] = np.nan
    if include_outliers and rows >= 10:
        df.loc[df.index[-max(1, rows // 50):], "feature"] = 9999
    return df


def create_mock_state(
    *,
    analysis_id: uuid.UUID | None = None,
    dataset_id: uuid.UUID | None = None,
    kaggle_url: str = "https://www.kaggle.com/datasets/test/sample-dataset",
    analysis_types: list[str] | None = None,
) -> AnalysisState:
    """Create a base AnalysisState dict."""
    state: AnalysisState = {
        "analysis_id": str(analysis_id or uuid.uuid4()),
        "kaggle_url": kaggle_url,
    }
    if dataset_id is not None:
        state["dataset_id"] = str(dataset_id)
    if analysis_types is not None:
        state["analysis_types"] = analysis_types
    return state


def make_session_context(session):
    """Create a session context manager for patching get_session_context."""

    @asynccontextmanager
    async def _context():
        yield session

    return _context
