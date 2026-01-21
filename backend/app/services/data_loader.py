"""Dataset loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from app.models.dataset import Dataset

logger = logging.getLogger(__name__)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a file path.

    Supports CSV, Parquet, and Excel formats.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_preprocessed_dataframe(dataset: "Dataset") -> pd.DataFrame | None:
    """Load preprocessed DataFrame if available.

    Args:
        dataset: Dataset model instance with metadata

    Returns:
        Preprocessed DataFrame if available, None otherwise.
        Caller should fall back to raw data if None is returned.
    """
    if not dataset.dataset_metadata:
        return None

    preprocessed_path = dataset.dataset_metadata.get("preprocessed_path")
    if not preprocessed_path:
        return None

    pp_path = Path(preprocessed_path)
    if not pp_path.exists():
        # Try to download from GCS if path doesn't exist locally
        preprocessed_gcs_path = dataset.dataset_metadata.get("preprocessed_gcs_path")
        if preprocessed_gcs_path:
            try:
                # Lazy import to avoid circular dependencies
                from app.services.storage import download_to_path

                import asyncio

                # Run the async download synchronously
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we can't use run_until_complete
                    logger.warning(
                        "Cannot download preprocessed data in async context; falling back to raw data"
                    )
                    return None
                loop.run_until_complete(download_to_path(preprocessed_gcs_path, pp_path))
            except Exception as exc:
                logger.warning(
                    "Failed to download preprocessed data from GCS",
                    exc_info=exc,
                )
                return None

    if not pp_path.exists():
        return None

    logger.info("Loading preprocessed dataframe from %s", str(pp_path))
    return load_dataframe(pp_path)


async def load_preprocessed_dataframe_async(dataset: "Dataset") -> pd.DataFrame | None:
    """Async version of load_preprocessed_dataframe.

    Args:
        dataset: Dataset model instance with metadata

    Returns:
        Preprocessed DataFrame if available, None otherwise.
    """
    if not dataset.dataset_metadata:
        return None

    preprocessed_path = dataset.dataset_metadata.get("preprocessed_path")
    if not preprocessed_path:
        return None

    pp_path = Path(preprocessed_path)
    if not pp_path.exists():
        # Try to download from GCS
        preprocessed_gcs_path = dataset.dataset_metadata.get("preprocessed_gcs_path")
        if preprocessed_gcs_path:
            try:
                from app.services.storage import download_to_path

                await download_to_path(preprocessed_gcs_path, pp_path)
            except Exception as exc:
                logger.warning(
                    "Failed to download preprocessed data from GCS",
                    exc_info=exc,
                )
                return None

    if not pp_path.exists():
        return None

    logger.info("Loading preprocessed dataframe (async) from %s", str(pp_path))
    return load_dataframe(pp_path)
