"""Integration tests for analysis API endpoints."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.auth import create_access_token
from app.crud.analysis import analysis_crud
from app.db.database import get_async_session
from app.main import app
from app.models.analysis import AnalysisStatus


def create_test_token(user_id: uuid.UUID) -> str:
    """Create a JWT token for testing."""
    return create_access_token({"user_id": user_id})


@pytest_asyncio.fixture
async def async_client(db_session):
    """Create an async client with a test database session."""

    async def override_get_async_session():
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_async_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@patch("app.api.v1.analyses.run_analysis_task.delay")
async def test_create_analysis_success(mock_delay, async_client, db_session):
    """POST /api/v1/analyses creates an analysis and enqueues a task."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)
    mock_delay.return_value = Mock(id="task-id")

    payload = {
        "kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset",
        "config": {"key": "value"},
    }
    response = await async_client.post(
        "/api/v1/analyses",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["kaggle_url"] == payload["kaggle_url"]
    assert data["user_id"] == str(user_id)
    assert data["status"] == AnalysisStatus.PENDING.value

    analysis_id = uuid.UUID(data["id"])
    analysis = await analysis_crud.get(db_session, analysis_id)
    assert analysis is not None
    assert analysis.celery_task_id == "task-id"
    mock_delay.assert_called_once_with(str(analysis_id), payload["kaggle_url"])


@pytest.mark.asyncio
async def test_create_analysis_invalid_url(async_client):
    """POST /api/v1/analyses rejects invalid Kaggle URLs."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    payload = {"kaggle_url": "https://example.com/invalid"}
    response = await async_client.post(
        "/api/v1/analyses",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    assert "Invalid Kaggle URL" in response.text


@pytest.mark.asyncio
async def test_create_analysis_unauthorized(async_client):
    """POST /api/v1/analyses requires authentication."""
    payload = {"kaggle_url": "https://www.kaggle.com/datasets/test/sample-dataset"}
    response = await async_client.post("/api/v1/analyses", json=payload)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_analyses_success(async_client, db_session):
    """GET /api/v1/analyses lists analyses for the user."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": user_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/a",
            "config": {},
        },
    )
    await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": user_id,
            "kaggle_url": "https://www.kaggle.com/competitions/test/b",
            "config": {},
        },
    )
    await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": uuid.uuid4(),
            "kaggle_url": "https://www.kaggle.com/datasets/test/c",
            "config": {},
        },
    )

    response = await async_client.get(
        "/api/v1/analyses",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    assert data["skip"] == 0
    assert data["limit"] == 100


@pytest.mark.asyncio
async def test_list_analyses_pagination(async_client, db_session):
    """GET /api/v1/analyses supports pagination."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    for idx in range(3):
        await analysis_crud.create(
            db_session,
            obj_in={
                "user_id": user_id,
                "kaggle_url": f"https://www.kaggle.com/datasets/test/{idx}",
                "config": {},
            },
        )

    response = await async_client.get(
        "/api/v1/analyses?skip=1&limit=1",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["items"]) == 1
    assert data["skip"] == 1
    assert data["limit"] == 1


@pytest.mark.asyncio
async def test_list_analyses_unauthorized(async_client):
    """GET /api/v1/analyses requires authentication."""
    response = await async_client.get("/api/v1/analyses")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_analysis_by_id_success(async_client, db_session):
    """GET /api/v1/analyses/{id} returns analysis details."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": user_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/sample",
            "config": {},
        },
    )

    response = await async_client.get(
        f"/api/v1/analyses/{analysis.id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(analysis.id)
    assert data["user_id"] == str(user_id)


@pytest.mark.asyncio
async def test_get_analysis_by_id_not_found(async_client):
    """GET /api/v1/analyses/{id} returns 404 for missing analysis."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    response = await async_client.get(
        f"/api/v1/analyses/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_analysis_by_id_forbidden(async_client, db_session):
    """GET /api/v1/analyses/{id} enforces ownership."""
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    token = create_test_token(other_id)

    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": owner_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/forbidden",
            "config": {},
        },
    )

    response = await async_client.get(
        f"/api/v1/analyses/{analysis.id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_analysis_success(async_client, db_session):
    """DELETE /api/v1/analyses/{id} removes the analysis."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": user_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/delete",
            "config": {},
        },
    )

    response = await async_client.delete(
        f"/api/v1/analyses/{analysis.id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 204
    deleted = await analysis_crud.get(db_session, analysis.id)
    assert deleted is None


@pytest.mark.asyncio
async def test_delete_running_analysis_cancels_task(async_client, db_session):
    """DELETE /api/v1/analyses/{id} revokes a running task."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": user_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/running",
            "config": {},
            "status": AnalysisStatus.RUNNING,
            "celery_task_id": "celery-task-id",
        },
    )

    with patch("app.api.v1.analyses.celery_app.control.revoke") as revoke_mock:
        with patch(
            "app.api.v1.analyses.analysis_crud.update",
            new_callable=AsyncMock,
        ) as update_mock:
            update_mock.return_value = analysis
            response = await async_client.delete(
                f"/api/v1/analyses/{analysis.id}",
                headers={"Authorization": f"Bearer {token}"},
            )

    assert response.status_code == 204
    revoke_mock.assert_called_once_with("celery-task-id", terminate=True)
    assert update_mock.await_count >= 1
    update_kwargs = update_mock.call_args.kwargs
    assert update_kwargs["obj_in"]["status"] == AnalysisStatus.CANCELLED


@pytest.mark.asyncio
async def test_delete_analysis_not_found(async_client):
    """DELETE /api/v1/analyses/{id} returns 404 when missing."""
    user_id = uuid.uuid4()
    token = create_test_token(user_id)

    response = await async_client.delete(
        f"/api/v1/analyses/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_analysis_forbidden(async_client, db_session):
    """DELETE /api/v1/analyses/{id} enforces ownership."""
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    token = create_test_token(other_id)

    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": owner_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/forbidden-delete",
            "config": {},
        },
    )

    response = await async_client.delete(
        f"/api/v1/analyses/{analysis.id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
