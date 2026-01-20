"""Integration tests for data quality preview endpoint."""

import uuid
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.auth import create_access_token
from app.db.database import get_async_session
from app.main import app


def create_test_token(user_id: uuid.UUID) -> str:
    return create_access_token({"user_id": user_id})


@pytest_asyncio.fixture
async def async_client(db_session):
    async def override_get_async_session():
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_async_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@patch("app.api.v1.analyses.get_kaggle_credentials", new_callable=AsyncMock)
@patch("app.api.v1.analyses._download_with_retry", new_callable=AsyncMock)
async def test_preview_valid_dataset(mock_download, mock_creds, async_client, tmp_path):
    user_id = uuid.uuid4()
    token = create_test_token(user_id)
    mock_creds.return_value = ("user", "key")

    df = pd.DataFrame({"a": range(1000), "b": range(1000, 2000), "c": range(2000, 3000)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    mock_download.return_value = tmp_path

    payload = {"kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset"}
    response = await async_client.post(
        "/api/v1/analyses/preview",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is True
    assert data["warnings"] == []
    assert data["dataset_info"]["row_count"] == 1000


@pytest.mark.asyncio
@patch("app.api.v1.analyses.get_kaggle_credentials", new_callable=AsyncMock)
@patch("app.api.v1.analyses._download_with_retry", new_callable=AsyncMock)
async def test_preview_invalid_dataset(mock_download, mock_creds, async_client, tmp_path):
    user_id = uuid.uuid4()
    token = create_test_token(user_id)
    mock_creds.return_value = ("user", "key")

    df = pd.DataFrame({"a": range(200), "b": range(200, 400), "c": range(400, 600)})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    mock_download.return_value = tmp_path

    payload = {"kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset"}
    response = await async_client.post(
        "/api/v1/analyses/preview",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is True
    assert data["warnings"]


@pytest.mark.asyncio
async def test_preview_invalid_kaggle_url(async_client):
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    payload = {"kaggle_url": "https://example.com/invalid"}
    response = await async_client.post(
        "/api/v1/analyses/preview",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_preview_requires_authentication(async_client):
    payload = {"kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset"}
    response = await async_client.post("/api/v1/analyses/preview", json=payload)

    assert response.status_code == 401
