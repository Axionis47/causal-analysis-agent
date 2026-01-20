"""Unit tests for ReportGenerationAgent."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from app.agents import reporting
from app.agents.reporting import (
    ReportGenerationAgent,
    _executive_summary,
    _technical_summary,
    _validation_payload,
    _visualization_payload,
)
from app.crud.analysis import analysis_crud
from app.models.generated_report import GeneratedReport
from tests.fixtures.agent_fixtures import (
    create_analysis,
    create_causal_graph,
    create_treatment_effect,
    create_validation_result,
)
from tests.fixtures.mock_helpers import make_session_context


@pytest.mark.asyncio
async def test_reporting_cache_hit():
    state = {
        "analysis_id": str(uuid.uuid4()),
        "report_ids": [str(uuid.uuid4())],
    }

    result = await ReportGenerationAgent()._run(state)

    assert result.outputs["cached"] is True


@pytest.mark.asyncio
async def test_executive_summary_generation(monkeypatch):
    async def fake_llm(*_args, **_kwargs):
        return {"overview": "Summary", "key_findings": [], "recommendations": ["Next"]}

    monkeypatch.setattr(reporting, "call_llm_with_retry", fake_llm)

    payload = {"effects": []}
    result = await _executive_summary(payload)

    assert result["overview"] == "Summary"
    assert result["recommendations"] == ["Next"]


def test_technical_summary_sections():
    payload = {
        "graphs": [{"method": "pc"}],
        "effects": [{"method": "propensity_matching"}],
        "validation": [],
        "analysis_types": ["mediation", "heterogeneous", "time_varying", "iv"],
        "extra_results": {
            "mediation": {"effect": 0.1},
            "heterogeneous": {"cate_mean": 0.2},
            "time_varying": {"time_bins": []},
            "iv": [{"ate": 0.3}],
        },
    }

    summary = _technical_summary(payload)

    assert "methodology" in summary
    assert summary["sections"]["mediation"]
    assert summary["sections"]["heterogeneous"]
    assert summary["sections"]["time_varying"]
    assert summary["sections"]["iv"]


@pytest.mark.asyncio
async def test_visualization_payload(db_session):
    analysis = await create_analysis(db_session)
    await create_causal_graph(db_session, analysis_id=analysis.id)
    await create_treatment_effect(db_session, analysis_id=analysis.id, ate=0.25)

    analysis = await analysis_crud.get_with_relations(db_session, analysis.id)
    payload = _visualization_payload(analysis)

    assert payload["graphs"]
    assert payload["effect_plot"]


@pytest.mark.asyncio
async def test_validation_payload(db_session):
    analysis = await create_analysis(db_session)
    await create_validation_result(db_session, analysis_id=analysis.id)

    analysis = await analysis_crud.get_with_relations(db_session, analysis.id)
    payload = _validation_payload(analysis)

    assert payload["tests"]


@pytest.mark.asyncio
async def test_reports_generated_and_persisted(db_session, monkeypatch, tmp_path):
    analysis = await create_analysis(db_session)
    await create_causal_graph(db_session, analysis_id=analysis.id)
    await create_treatment_effect(db_session, analysis_id=analysis.id)
    await create_validation_result(db_session, analysis_id=analysis.id)

    async def fake_llm(*_args, **_kwargs):
        return {"overview": "Summary", "key_findings": [], "recommendations": []}

    async def fake_upload(path: Path, bucket_name: str, destination: str):  # noqa: ARG002
        return f"gs://bucket/{destination}"

    async def fake_upload_chart_images(*_args, **_kwargs):
        return []

    def fake_markdown_report(
        analysis_id: uuid.UUID,
        kaggle_url: str | None,  # noqa: ARG002
        summary: dict[str, Any],  # noqa: ARG002
        technical: dict[str, Any],  # noqa: ARG002
        effects: list[dict[str, Any]],  # noqa: ARG002
        graphs: list[dict[str, Any]],  # noqa: ARG002
        validation: list[dict[str, Any]],  # noqa: ARG002
        chart_urls: list[str] | None = None,  # noqa: ARG002
    ) -> Path:
        path = tmp_path / str(analysis_id) / "reports" / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Report", encoding="utf-8")
        return path

    def fake_pdf_report(
        analysis_id: uuid.UUID,
        kaggle_url: str | None,  # noqa: ARG002
        status: str | None,  # noqa: ARG002
        summary: dict[str, Any],  # noqa: ARG002
        technical: dict[str, Any],  # noqa: ARG002
        effects: list[dict[str, Any]],  # noqa: ARG002
        graphs: list[dict[str, Any]],  # noqa: ARG002
        validation: list[dict[str, Any]],  # noqa: ARG002
        visualizations: dict[str, Any],  # noqa: ARG002
    ) -> Path:
        path = tmp_path / str(analysis_id) / "reports" / "report.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 test")
        return path

    monkeypatch.setattr(reporting, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(reporting, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(reporting, "upload_file", fake_upload)
    monkeypatch.setattr(reporting, "local_storage_root", lambda: tmp_path)
    monkeypatch.setattr(reporting, "_generate_markdown_report", fake_markdown_report)
    monkeypatch.setattr(reporting, "_generate_pdf_report", fake_pdf_report)
    monkeypatch.setattr(reporting, "_upload_chart_images", fake_upload_chart_images)

    state = {"analysis_id": str(analysis.id)}
    result = await ReportGenerationAgent()._run(state)

    assert result.outputs["reports"] == 7
    html_path = tmp_path / str(analysis.id) / "reports" / "report.html"
    assert html_path.exists()

    stored = await db_session.execute(
        GeneratedReport.__table__.select().where(GeneratedReport.analysis_id == analysis.id)
    )
    assert len(stored.fetchall()) == 7


@pytest.mark.asyncio
async def test_llm_fallback_summary(monkeypatch):
    async def fake_llm(*_args, **_kwargs):
        return None

    monkeypatch.setattr(reporting, "call_llm_with_retry", fake_llm)

    payload = {"effects": []}
    summary = await _executive_summary(payload)

    assert summary["overview"]
    assert summary["recommendations"]


@pytest.mark.asyncio
async def test_empty_analysis_handles_gracefully(db_session, monkeypatch, tmp_path):
    analysis = await create_analysis(db_session)

    async def fake_llm(*_args, **_kwargs):
        return {"overview": "Summary", "key_findings": [], "recommendations": []}

    async def fake_upload(path: Path, bucket_name: str, destination: str):  # noqa: ARG002
        return f"gs://bucket/{destination}"

    async def fake_upload_chart_images(*_args, **_kwargs):
        return []

    def fake_markdown_report(
        analysis_id: uuid.UUID,
        kaggle_url: str | None,  # noqa: ARG002
        summary: dict[str, Any],  # noqa: ARG002
        technical: dict[str, Any],  # noqa: ARG002
        effects: list[dict[str, Any]],  # noqa: ARG002
        graphs: list[dict[str, Any]],  # noqa: ARG002
        validation: list[dict[str, Any]],  # noqa: ARG002
        chart_urls: list[str] | None = None,  # noqa: ARG002
    ) -> Path:
        path = tmp_path / str(analysis_id) / "reports" / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Report", encoding="utf-8")
        return path

    def fake_pdf_report(
        analysis_id: uuid.UUID,
        kaggle_url: str | None,  # noqa: ARG002
        status: str | None,  # noqa: ARG002
        summary: dict[str, Any],  # noqa: ARG002
        technical: dict[str, Any],  # noqa: ARG002
        effects: list[dict[str, Any]],  # noqa: ARG002
        graphs: list[dict[str, Any]],  # noqa: ARG002
        validation: list[dict[str, Any]],  # noqa: ARG002
        visualizations: dict[str, Any],  # noqa: ARG002
    ) -> Path:
        path = tmp_path / str(analysis_id) / "reports" / "report.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 test")
        return path

    monkeypatch.setattr(reporting, "get_session_context", make_session_context(db_session))
    monkeypatch.setattr(reporting, "call_llm_with_retry", fake_llm)
    monkeypatch.setattr(reporting, "upload_file", fake_upload)
    monkeypatch.setattr(reporting, "local_storage_root", lambda: tmp_path)
    monkeypatch.setattr(reporting, "_generate_markdown_report", fake_markdown_report)
    monkeypatch.setattr(reporting, "_generate_pdf_report", fake_pdf_report)
    monkeypatch.setattr(reporting, "_upload_chart_images", fake_upload_chart_images)

    state = {"analysis_id": str(analysis.id)}
    result = await ReportGenerationAgent()._run(state)

    assert result.outputs["reports"] == 7
