"""Storage utilities for GCS and local filesystem."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def local_storage_root() -> Path:
    return Path(settings.LOCAL_STORAGE_PATH).expanduser().resolve()


def ensure_local_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_gcs_project() -> Optional[str]:
    """Get the GCP project ID from settings."""
    return settings.VERTEX_AI_PROJECT or None


def gcs_available() -> bool:
    if not settings.GCS_BUCKET_NAME:
        return False
    project = _get_gcs_project()
    if not project:
        return False
    try:
        from google.cloud import storage
        # Actually try to create a client to verify credentials are available
        storage.Client(project=project)
        return True
    except Exception:
        return False


async def upload_file(local_path: Path, bucket_name: str, destination: str) -> Optional[str]:
    if not bucket_name or not gcs_available():
        return None
    from google.cloud import storage

    project = _get_gcs_project()
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination}"


async def upload_to_gcs(local_path: Path, destination: str) -> Optional[str]:
    """Upload a file to the default GCS bucket.

    Convenience wrapper around upload_file that uses the configured bucket.
    """
    return await upload_file(local_path, settings.GCS_BUCKET_NAME, destination)


async def download_to_path(gcs_path: str, local_path: Path) -> Path:
    if not gcs_available():
        raise RuntimeError("GCS client not available")
    from google.cloud import storage

    if not gcs_path.startswith("gs://"):
        raise ValueError("Invalid GCS path")
    _, rest = gcs_path.split("gs://", 1)
    bucket_name, blob_name = rest.split("/", 1)

    project = _get_gcs_project()
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    ensure_local_dir(local_path.parent)
    blob.download_to_filename(str(local_path))
    return local_path


def infer_local_path(metadata: dict) -> Optional[Path]:
    local_path = metadata.get("local_path") if isinstance(metadata, dict) else None
    if local_path:
        return Path(local_path)
    return None


def generate_signed_url(gcs_path: str, expiration_minutes: int = 60) -> Optional[str]:
    if not gcs_available():
        return None
    if not gcs_path.startswith("gs://"):
        return None
    try:
        from google.cloud import storage

        _, rest = gcs_path.split("gs://", 1)
        bucket_name, blob_name = rest.split("/", 1)
        project = _get_gcs_project()
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        extension = Path(blob_name).suffix.lower()
        content_type = None
        if extension == ".pdf":
            content_type = "application/pdf"
        elif extension in {".md", ".markdown"}:
            content_type = "text/markdown"
        elif extension in {".html", ".htm"}:
            content_type = "text/html"
        elif extension == ".png":
            content_type = "image/png"

        response_disposition = None
        if extension in {".pdf", ".md", ".markdown", ".html", ".htm"}:
            response_disposition = f'attachment; filename="{Path(blob_name).name}"'

        return blob.generate_signed_url(
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
            response_type=content_type,
            response_disposition=response_disposition,
        )
    except Exception as exc:
        logger.warning("Failed to generate signed URL", error=str(exc))
        return None
