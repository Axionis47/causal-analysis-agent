"""Tests for report generation and download endpoints."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.agents import reporting
from app.agents.reporting import _generate_markdown_report, _generate_pdf_report, _render_html_template
from app.api.v1 import results as results_api
from app.core.auth import create_access_token
from app.crud.analysis import analysis_crud
from app.db.database import get_async_session
from app.main import app
from app.models.generated_report import GeneratedReport, ReportFormat, ReportType


def create_test_token(user_id: uuid.UUID) -> str:
    return create_access_token({"user_id": user_id})


@pytest_asyncio.fixture
async def async_client(db_session):
    async def override_get_async_session():
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_async_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def _sample_payloads():
    summary = {
        "overview": "Summary overview.",
        "key_findings": [
            {
                "treatment": "Treatment",
                "outcome": "Outcome",
                "method": "iv",
                "ate": 0.42,
                "ci": [0.1, 0.7],
            }
        ],
        "recommendations": ["Review with stakeholders."],
    }
    technical = {
        "methodology": {"discovery_methods": ["pc"], "treatment_methods": ["iv"]},
        "sections": {"sample_size": {"total": 1000}},
        "extra_results": {"notes": "none"},
    }
    effects = [
        {
            "treatment": "Treatment",
            "outcome": "Outcome",
            "method": "iv",
            "ate": 0.42,
            "ci": [0.1, 0.7],
        }
    ]
    graphs = [
        {
            "method": "pc",
            "nodes": [{"name": "A", "node_type": "variable"}],
            "edges": [{"source": "A", "target": "B", "confidence": 0.8}],
            "confidence": 0.8,
        }
    ]
    validation = [
        {
            "method": "placebo",
            "passed": True,
            "confidence_score": 0.91,
            "details": {"recommendations": ["Proceed"]},
        }
    ]
    return summary, technical, effects, graphs, validation


def test_generate_markdown_report(tmp_path, monkeypatch):
    summary, technical, effects, graphs, validation = _sample_payloads()
    monkeypatch.setattr(reporting, "local_storage_root", lambda: tmp_path)

    path = _generate_markdown_report(
        analysis_id=uuid.uuid4(),
        kaggle_url="https://www.kaggle.com/datasets/test/sample",
        summary=summary,
        technical=technical,
        effects=effects,
        graphs=graphs,
        validation=validation,
        chart_urls=["https://example.com/chart.png"],
    )

    content = path.read_text(encoding="utf-8")
    assert "# Causal Analysis Report" in content
    assert "## Executive Summary" in content
    assert "## Treatment Effects" in content
    assert "| Treatment | Outcome | Method |" in content
    assert "## Causal Graphs" in content
    assert "## Validation Results" in content
    assert "![Chart 1](https://example.com/chart.png)" in content


def test_render_html_template(monkeypatch):
    summary, technical, effects, graphs, validation = _sample_payloads()
    template_dir = Path(__file__).resolve().parents[1] / "app" / "templates"
    monkeypatch.setattr(reporting.settings, "REPORT_TEMPLATE_DIR", str(template_dir))

    html = _render_html_template(
        analysis_id=uuid.uuid4(),
        kaggle_url="https://www.kaggle.com/datasets/test/sample",
        status="completed",
        summary=summary,
        technical=technical,
        effects=effects,
        graphs=graphs,
        validation=validation,
        visualizations={},
    )

    assert "Causal Analysis Report" in html
    assert "Executive Summary" in html
    assert "Treatment Effects" in html
    assert "Validation Results" in html


def test_generate_pdf_report(tmp_path, monkeypatch):
    summary, technical, effects, graphs, validation = _sample_payloads()
    monkeypatch.setattr(reporting, "local_storage_root", lambda: tmp_path)
    template_dir = Path(__file__).resolve().parents[1] / "app" / "templates"
    monkeypatch.setattr(reporting.settings, "REPORT_TEMPLATE_DIR", str(template_dir))

    class FakeHTML:
        def __init__(self, string, base_url=None):  # noqa: ARG002
            self.string = string

        def write_pdf(self, target, stylesheets=None):  # noqa: ARG002
            Path(target).write_bytes(b"%PDF-1.4 fake")

    class FakeCSS:
        def __init__(self, string):  # noqa: ARG002
            self.string = string

    fake_module = SimpleNamespace(HTML=FakeHTML, CSS=FakeCSS)
    monkeypatch.setitem(sys.modules, "weasyprint", fake_module)

    path = _generate_pdf_report(
        analysis_id=uuid.uuid4(),
        kaggle_url="https://www.kaggle.com/datasets/test/sample",
        status="completed",
        summary=summary,
        technical=technical,
        effects=effects,
        graphs=graphs,
        validation=validation,
        visualizations={},
    )

    assert path is not None
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.asyncio
async def test_download_endpoint_pdf(async_client, db_session, monkeypatch):
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
    report = GeneratedReport(
        analysis_id=analysis.id,
        report_type=ReportType.EXECUTIVE,
        format=ReportFormat.PDF,
        content=None,
        gcs_path="gs://bucket/report.pdf",
        generated_at=datetime.now(timezone.utc),
        file_size_bytes=123,
    )
    db_session.add(report)
    await db_session.commit()

    monkeypatch.setattr(results_api, "generate_signed_url", lambda *_args, **_kwargs: "https://signed-url")

    response = await async_client.get(
        f"/api/v1/results/{analysis.id}/download?format=pdf",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://signed-url"
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"] == 'attachment; filename="report.pdf"'


@pytest.mark.asyncio
async def test_download_endpoint_markdown(async_client, db_session, monkeypatch):
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
    report = GeneratedReport(
        analysis_id=analysis.id,
        report_type=ReportType.TECHNICAL,
        format=ReportFormat.MARKDOWN,
        content=None,
        gcs_path="gs://bucket/report.md",
        generated_at=datetime.now(timezone.utc),
        file_size_bytes=256,
    )
    db_session.add(report)
    await db_session.commit()

    monkeypatch.setattr(results_api, "generate_signed_url", lambda *_args, **_kwargs: "https://signed-url")

    response = await async_client.get(
        f"/api/v1/results/{analysis.id}/download?format=markdown",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://signed-url"
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="report.md"'


@pytest.mark.asyncio
async def test_download_endpoint_unauthorized(async_client, db_session):
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    token = create_test_token(other_id)
    analysis = await analysis_crud.create(
        db_session,
        obj_in={
            "user_id": owner_id,
            "kaggle_url": "https://www.kaggle.com/datasets/test/sample",
            "config": {},
        },
    )

    response = await async_client.get(
        f"/api/v1/results/{analysis.id}/download?format=pdf",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_download_endpoint_not_found(async_client, db_session):
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
        f"/api/v1/results/{analysis.id}/download?format=pdf",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404
