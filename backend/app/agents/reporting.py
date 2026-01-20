"""Report generation agent for executive and technical outputs."""

from __future__ import annotations

import base64
import inspect
import json
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from app import __version__
from app.agents.base import AgentResult, BaseAgent
from app.agents.llm_utils import call_llm_with_retry
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.db.database import get_session_context
from app.llm.router import ROUTER
from app.models.analysis_stage import StageType
from app.models.generated_report import GeneratedReport, ReportFormat, ReportType
from app.prompts.reporting import build_executive_summary_prompt
from app.services.pptx_generator import generate_and_upload_pptx
from app.services.storage import ensure_local_dir, generate_signed_url, local_storage_root, upload_file

logger = get_logger(__name__)


class ReportGenerationAgent(BaseAgent):
    name = "Report Generation Agent"
    stage = StageType.REPORTING

    async def _run(self, state):
        if state.get("report_ids"):
            return AgentResult(state=state, outputs={"cached": True}, message="Reports cached")

        agent_logger = logger.bind(
            analysis_id=state.get("analysis_id"),
            stage=self.stage.value,
            agent_name=self.name,
        )
        analysis_id = uuid.UUID(state["analysis_id"])
        async with get_session_context() as session:
            analysis = await analysis_crud.get_with_relations(session, analysis_id)
            if analysis is None:
                raise ValueError("Analysis not found")

        payload = _build_results_payload(analysis, state)
        start = time.monotonic()
        summary = await _executive_summary(payload, agent_instance=self, state=state)
        agent_logger.info(
            "Report generated",
            report_type="executive",
            size_bytes=len(json.dumps(summary)),
            generation_duration_ms=(time.monotonic() - start) * 1000,
        )
        start = time.monotonic()
        technical = _technical_summary(payload)
        agent_logger.info(
            "Report generated",
            report_type="technical",
            size_bytes=len(json.dumps(technical)),
            generation_duration_ms=(time.monotonic() - start) * 1000,
        )
        start = time.monotonic()
        visualizations = _visualization_payload(analysis)
        agent_logger.info(
            "Report generated",
            report_type="visualization",
            size_bytes=len(json.dumps(visualizations)),
            generation_duration_ms=(time.monotonic() - start) * 1000,
        )
        start = time.monotonic()
        validation = _validation_payload(analysis)
        agent_logger.info(
            "Report generated",
            report_type="validation",
            size_bytes=len(json.dumps(validation)),
            generation_duration_ms=(time.monotonic() - start) * 1000,
        )

        effects = payload.get("effects", [])
        graphs = payload.get("graphs", [])
        validation_results = payload.get("validation", [])

        report_ids = []
        async with get_session_context() as session:
            report_ids.append(
                await _store_report(session, analysis_id, ReportType.EXECUTIVE, ReportFormat.JSON, summary)
            )
            report_ids.append(
                await _store_report(session, analysis_id, ReportType.TECHNICAL, ReportFormat.JSON, technical)
            )
            report_ids.append(
                await _store_report(session, analysis_id, ReportType.VISUALIZATION, ReportFormat.JSON, visualizations)
            )
            report_ids.append(
                await _store_report(session, analysis_id, ReportType.VALIDATION, ReportFormat.JSON, validation)
            )

        reports_bucket = _resolve_reports_bucket(analysis)
        html_start = time.monotonic()
        html_path = _write_html_report(
            analysis_id,
            analysis.kaggle_url,
            analysis.status.value,
            summary,
            technical,
            effects,
            graphs,
            validation_results,
            visualizations,
        )
        agent_logger.info(
            "Report generated",
            report_type="html",
            size_bytes=html_path.stat().st_size if html_path.exists() else 0,
            generation_duration_ms=(time.monotonic() - html_start) * 1000,
        )
        gcs_path = await upload_file(
            html_path,
            bucket_name=reports_bucket,
            destination=f"{analysis_id}/report.html",
        )
        async with get_session_context() as session:
            html_report = GeneratedReport(
                analysis_id=analysis_id,
                report_type=ReportType.TECHNICAL,
                format=ReportFormat.HTML,
                content=None,
                gcs_path=gcs_path or str(html_path),
                generated_at=datetime.now(timezone.utc),
                file_size_bytes=html_path.stat().st_size if html_path.exists() else None,
            )
            session.add(html_report)
            await session.flush()
            report_ids.append(str(html_report.id))

        if settings.REPORT_ENABLE_MARKDOWN:
            markdown_start = time.monotonic()
            chart_urls = await _upload_chart_images(analysis_id, visualizations, reports_bucket)
            markdown_path = _generate_markdown_report(
                analysis_id,
                analysis.kaggle_url,
                summary,
                technical,
                effects,
                graphs,
                validation_results,
                chart_urls,
            )
            agent_logger.info(
                "Report generated",
                report_type="markdown",
                size_bytes=markdown_path.stat().st_size if markdown_path.exists() else 0,
                generation_duration_ms=(time.monotonic() - markdown_start) * 1000,
            )
            markdown_gcs_path = await upload_file(
                markdown_path,
                bucket_name=reports_bucket,
                destination=f"{analysis_id}/report.md",
            )
            async with get_session_context() as session:
                markdown_report = GeneratedReport(
                    analysis_id=analysis_id,
                    report_type=ReportType.TECHNICAL,
                    format=ReportFormat.MARKDOWN,
                    content=None,
                    gcs_path=markdown_gcs_path or str(markdown_path),
                    generated_at=datetime.now(timezone.utc),
                    file_size_bytes=markdown_path.stat().st_size if markdown_path.exists() else None,
                )
                session.add(markdown_report)
                await session.flush()
                report_ids.append(str(markdown_report.id))

        if settings.REPORT_ENABLE_PDF:
            pdf_start = time.monotonic()
            pdf_path = _generate_pdf_report(
                analysis_id,
                analysis.kaggle_url,
                analysis.status.value,
                summary,
                technical,
                effects,
                graphs,
                validation_results,
                visualizations,
            )
            if pdf_path:
                agent_logger.info(
                    "Report generated",
                    report_type="pdf",
                    size_bytes=pdf_path.stat().st_size if pdf_path.exists() else 0,
                    generation_duration_ms=(time.monotonic() - pdf_start) * 1000,
                )
                pdf_gcs_path = await upload_file(
                    pdf_path,
                    bucket_name=reports_bucket,
                    destination=f"{analysis_id}/report.pdf",
                )
                async with get_session_context() as session:
                    pdf_report = GeneratedReport(
                        analysis_id=analysis_id,
                        report_type=ReportType.EXECUTIVE,
                        format=ReportFormat.PDF,
                        content=None,
                        gcs_path=pdf_gcs_path or str(pdf_path),
                        generated_at=datetime.now(timezone.utc),
                        file_size_bytes=pdf_path.stat().st_size if pdf_path.exists() else None,
                    )
                    session.add(pdf_report)
                    await session.flush()
                    report_ids.append(str(pdf_report.id))
            else:
                agent_logger.warning("PDF report generation skipped")

        # Generate PPTX report
        if settings.REPORT_ENABLE_PPTX:
            pptx_start = time.monotonic()
            pptx_path, pptx_gcs_path = await generate_and_upload_pptx(
                analysis_id=analysis_id,
                dataset_name=analysis.kaggle_url,
                executive_summary=summary,
                technical_report=technical,
                effects=effects,
                graphs=graphs,
                validation_results=validation_results,
                bucket_name=reports_bucket,
            )
            if pptx_path:
                agent_logger.info(
                    "Report generated",
                    report_type="pptx",
                    size_bytes=pptx_path.stat().st_size if pptx_path.exists() else 0,
                    generation_duration_ms=(time.monotonic() - pptx_start) * 1000,
                )
                async with get_session_context() as session:
                    pptx_report = GeneratedReport(
                        analysis_id=analysis_id,
                        report_type=ReportType.EXECUTIVE,
                        format=ReportFormat.PPTX,
                        content=None,
                        gcs_path=pptx_gcs_path or str(pptx_path),
                        generated_at=datetime.now(timezone.utc),
                        file_size_bytes=pptx_path.stat().st_size if pptx_path.exists() else None,
                    )
                    session.add(pptx_report)
                    await session.flush()
                    report_ids.append(str(pptx_report.id))
            else:
                agent_logger.warning("PPTX report generation skipped")

        state["report_ids"] = report_ids
        return AgentResult(state=state, outputs={"reports": len(report_ids)}, message="Reports generated")


def _build_results_payload(analysis, state) -> dict[str, Any]:
    effects = [
        {
            "treatment": effect.treatment_variable,
            "outcome": effect.outcome_variable,
            "method": effect.method.value,
            "ate": effect.ate,
            "ci": [effect.ate_ci_lower, effect.ate_ci_upper],
        }
        for effect in analysis.treatment_effects
    ]
    graphs = [
        {
            "method": graph.method.value,
            "nodes": graph.nodes,
            "edges": graph.edges,
            "confidence": graph.confidence,
        }
        for graph in analysis.causal_graphs
    ]
    validation = [
        {
            "method": result.method,
            "passed": result.passed,
            "confidence_score": result.confidence_score,
            "details": result.details,
        }
        for result in analysis.validation_results
    ]
    return {
        "analysis_id": str(analysis.id),
        "kaggle_url": analysis.kaggle_url,
        "graphs": graphs,
        "effects": effects,
        "validation": validation,
        "extra_results": state.get("extra_results", {}),
        "analysis_types": state.get("analysis_types", []),
    }


async def _executive_summary(
    payload: dict[str, Any],
    agent_instance: ReportGenerationAgent | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_executive_summary_prompt(payload)
    analysis_id = state.get("analysis_id") if state else None
    response = await call_llm_with_retry(ROUTER.structured_output, prompt, analysis_id=analysis_id)

    # Track LLM usage if agent instance and state are provided
    if agent_instance and state:
        response_text = str(response) if response else ""
        estimated_tokens = agent_instance._estimate_tokens(prompt, response_text)
        await agent_instance._track_llm_usage(state, estimated_tokens, "gpt-4")

    if response:
        return response
    return {
        "overview": "Causal analysis completed.",
        "key_findings": payload.get("effects", []),
        "recommendations": ["Review results with domain experts."],
    }


def _technical_summary(payload: dict[str, Any]) -> dict[str, Any]:
    analysis_types = [item.lower() for item in payload.get("analysis_types", [])]
    extra = payload.get("extra_results", {})
    sections: dict[str, Any] = {}
    if "mediation" in analysis_types:
        sections["mediation"] = extra.get("mediation", {})
    if "heterogeneous" in analysis_types:
        sections["heterogeneous"] = extra.get("heterogeneous", {})
    if "time_varying" in analysis_types:
        sections["time_varying"] = extra.get("time_varying", {})
    if "iv" in analysis_types:
        sections["iv"] = extra.get("iv", {})
    return {
        "methodology": {
            "discovery_methods": [graph["method"] for graph in payload.get("graphs", [])],
            "treatment_methods": [effect["method"] for effect in payload.get("effects", [])],
        },
        "effects": payload.get("effects", []),
        "graphs": payload.get("graphs", []),
        "validation": payload.get("validation", []),
        "extra_results": payload.get("extra_results", {}),
        "sections": sections,
    }


def _visualization_payload(analysis) -> dict[str, Any]:
    graph_payloads = [
        {"nodes": graph.nodes, "edges": graph.edges, "graph_data": graph.graph_data}
        for graph in analysis.causal_graphs
    ]
    effects = [effect for effect in analysis.treatment_effects if effect.ate is not None]
    plot_data = {}
    if effects:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[f"{e.treatment_variable}->{e.outcome_variable}" for e in effects],
                    y=[e.ate for e in effects],
                )
            ]
        )
        plot_data = fig.to_plotly_json()
    return {"graphs": graph_payloads, "effect_plot": plot_data}


def _validation_payload(analysis) -> dict[str, Any]:
    return {
        "tests": [
            {
                "method": result.method,
                "passed": result.passed,
                "confidence_score": result.confidence_score,
                "details": result.details,
            }
            for result in analysis.validation_results
        ]
    }


async def _store_report(
    session,
    analysis_id: uuid.UUID,
    report_type: ReportType,
    format: ReportFormat,
    content: dict[str, Any],
) -> str:
    report = GeneratedReport(
        analysis_id=analysis_id,
        report_type=report_type,
        format=format,
        content=json.dumps(content),
        generated_at=datetime.now(timezone.utc),
        file_size_bytes=len(json.dumps(content)),
    )
    session.add(report)
    await session.flush()
    return str(report.id)


def _write_html_report(
    analysis_id: uuid.UUID,
    kaggle_url: str | None,
    status: str | None,
    summary: dict[str, Any],
    technical: dict[str, Any],
    effects: list[dict[str, Any]],
    graphs: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    visualizations: dict[str, Any],
) -> Path:
    root = local_storage_root() / str(analysis_id) / "reports"
    ensure_local_dir(root)
    path = root / "report.html"
    html = _render_html_template(
        analysis_id=analysis_id,
        kaggle_url=kaggle_url,
        status=status,
        summary=summary,
        technical=technical,
        effects=effects,
        graphs=graphs,
        validation=validation,
        visualizations=visualizations,
    )
    path.write_text(html, encoding="utf-8")
    return path


def _generate_markdown_report(
    analysis_id: uuid.UUID,
    kaggle_url: str | None,
    summary: dict[str, Any],
    technical: dict[str, Any],
    effects: list[dict[str, Any]],
    graphs: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    chart_urls: list[str] | None = None,
) -> Path:
    root = local_storage_root() / str(analysis_id) / "reports"
    ensure_local_dir(root)
    path = root / "report.md"
    generated_at = datetime.now(timezone.utc)

    summary_block = _normalize_summary(summary, effects)
    effect_rows = _normalize_effects(effects)
    graph_rows = _normalize_graphs(graphs)
    validation_rows = _normalize_validation(validation)

    lines: list[str] = [
        "# Causal Analysis Report",
        "",
        f"**Analysis ID**: {analysis_id}",
        f"**Dataset**: {kaggle_url or 'N/A'}",
        f"**Generated**: {_format_date(generated_at)}",
        "",
        "## Executive Summary",
        summary_block.get("overview") or "No overview available.",
        "",
        "### Key Findings",
    ]
    key_findings = summary_block.get("key_findings") or []
    if key_findings:
        lines.extend([f"- {finding}" for finding in key_findings])
    else:
        lines.append("- None recorded.")

    lines.extend(["", "### Recommendations"])
    recommendations = summary_block.get("recommendations") or []
    if recommendations:
        lines.extend([f"- {rec}" for rec in recommendations])
    else:
        lines.append("- None provided.")

    lines.extend(
        [
            "",
            "## Treatment Effects",
            "| Treatment | Outcome | Method | ATE | CI Lower | CI Upper |",
            "|---|---|---|---|---|---|",
        ]
    )
    if effect_rows:
        for effect in effect_rows:
            lines.append(
                "| {treatment} | {outcome} | {method} | {ate} | {ci_lower} | {ci_upper} |".format(
                    treatment=effect["treatment"],
                    outcome=effect["outcome"],
                    method=effect["method"],
                    ate=_format_number(effect["ate"]),
                    ci_lower=_format_number(effect["ci_lower"]),
                    ci_upper=_format_number(effect["ci_upper"]),
                )
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.append("")
    lines.append("## Causal Graphs")
    if graph_rows:
        for graph in graph_rows:
            lines.append(f"### {str(graph['method']).upper()} Discovery")
            lines.append(f"- Confidence: {_format_confidence(graph['confidence'])}")
            lines.append("- Nodes:")
            if graph["nodes"]:
                lines.extend(
                    [
                        f"  - {node['name']} ({node['node_type']})"
                        for node in graph["nodes"]
                    ]
                )
            else:
                lines.append("  - None")
            lines.append("")
            lines.append("| Source | Target | Confidence |")
            lines.append("|---|---|---|")
            if graph["edges"]:
                for edge in graph["edges"]:
                    lines.append(
                        "| {source} | {target} | {confidence} |".format(
                            source=edge["source"],
                            target=edge["target"],
                            confidence=_format_confidence(edge["confidence"]),
                        )
                    )
            else:
                lines.append("| N/A | N/A | N/A |")
            lines.append("")
    else:
        lines.append("No causal graphs available.")
        lines.append("")

    lines.append("## Validation Results")
    lines.append("| Method | Status | Confidence | Recommendations |")
    lines.append("|---|---|---|---|")
    if validation_rows:
        for result in validation_rows:
            status_icon = _status_icon(result["passed"])
            status_label = "Passed" if result["passed"] else "Failed"
            lines.append(
                "| {method} | {icon} {label} | {confidence} | {rec} |".format(
                    method=result["method"],
                    icon=status_icon,
                    label=status_label,
                    confidence=_format_confidence(result["confidence"]),
                    rec=result["recommendations"] or "N/A",
                )
            )
    else:
        lines.append("| N/A | N/A | N/A | N/A |")

    lines.append("")
    lines.append("## Visualizations")
    if chart_urls:
        for idx, url in enumerate(chart_urls, start=1):
            lines.append(f"![Chart {idx}]({url})")
    else:
        lines.append("No visualizations available.")

    lines.append("")
    lines.append("## Technical Details")
    lines.append("```json")
    lines.append(json.dumps(technical.get("sections", {}) or technical.get("extra_results", {}), indent=2))
    lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _render_html_template(
    *,
    analysis_id: uuid.UUID,
    kaggle_url: str | None,
    status: str | None,
    summary: dict[str, Any],
    technical: dict[str, Any],
    effects: list[dict[str, Any]],
    graphs: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    visualizations: dict[str, Any],
    generated_at: datetime | None = None,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    normalized_summary = _normalize_summary(summary, effects)
    normalized_effects = _normalize_effects(effects)
    normalized_graphs = _normalize_graphs(graphs)
    normalized_validation = _normalize_validation(validation)
    chart_images = _build_chart_images(visualizations)
    template_dir = Path(settings.REPORT_TEMPLATE_DIR).expanduser().resolve()
    env = _get_template_env(template_dir)
    env.filters["format_number"] = _format_number
    env.filters["format_confidence"] = _format_confidence
    env.filters["format_date"] = _format_date
    env.filters["status_icon"] = _status_icon
    context = {
        "analysis_id": str(analysis_id),
        "kaggle_url": kaggle_url,
        "generated_at": generated_at,
        "status": (status or "completed").replace("_", " ").title(),
        "summary": normalized_summary,
        "technical": technical or {},
        "effects": normalized_effects,
        "graphs": normalized_graphs,
        "validation": normalized_validation,
        "chart_images": chart_images,
        "system_version": f"{settings.APP_NAME} {__version__}",
        "page_size": settings.REPORT_PDF_PAGE_SIZE,
        "page_margin": settings.REPORT_PDF_MARGIN,
    }
    try:
        template = env.get_template("report.html")
    except TemplateNotFound:
        logger.warning("Report template not found", template_dir=str(template_dir))
        return _basic_html_report(normalized_summary, technical)
    try:
        return template.render(**context)
    except Exception:
        logger.exception("Failed to render report template")
        return _basic_html_report(normalized_summary, technical)


def _generate_pdf_report(
    analysis_id: uuid.UUID,
    kaggle_url: str | None,
    status: str | None,
    summary: dict[str, Any],
    technical: dict[str, Any],
    effects: list[dict[str, Any]],
    graphs: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    visualizations: dict[str, Any],
) -> Path | None:
    root = local_storage_root() / str(analysis_id) / "reports"
    ensure_local_dir(root)
    path = root / "report.pdf"
    html = _render_html_template(
        analysis_id=analysis_id,
        kaggle_url=kaggle_url,
        status=status,
        summary=summary,
        technical=technical,
        effects=effects,
        graphs=graphs,
        validation=validation,
        visualizations=visualizations,
    )
    try:
        from weasyprint import CSS, HTML
    except Exception as exc:
        logger.warning("WeasyPrint not available, skipping PDF generation", error=str(exc))
        return None
    try:
        stylesheet = CSS(
            string=f"@page {{ size: {settings.REPORT_PDF_PAGE_SIZE}; margin: {settings.REPORT_PDF_MARGIN}; }}"
        )
        html_doc = HTML(string=html, base_url=str(Path(settings.REPORT_TEMPLATE_DIR).resolve()))
        write_kwargs = {"target": str(path), "stylesheets": [stylesheet]}
        if "optimize_size" in inspect.signature(html_doc.write_pdf).parameters:
            write_kwargs["optimize_size"] = ("images", "fonts")
        html_doc.write_pdf(**write_kwargs)
    except Exception:
        logger.exception("PDF generation failed")
        return None
    return path


def _resolve_reports_bucket(analysis) -> str:
    if analysis.config:
        return (
            analysis.config.get("reports_bucket")
            or analysis.config.get("bucket_name")
            or analysis.config.get("gcs_reports_bucket")
            or settings.GCS_REPORTS_BUCKET_NAME
        )
    return settings.GCS_REPORTS_BUCKET_NAME


async def _upload_chart_images(
    analysis_id: uuid.UUID,
    visualizations: dict[str, Any],
    bucket_name: str | None,
) -> list[str]:
    if not visualizations or not bucket_name:
        return []
    charts_dir = local_storage_root() / str(analysis_id) / "reports" / "charts"
    ensure_local_dir(charts_dir)
    chart_urls: list[str] = []
    for name, fig in _extract_plotly_figures(visualizations):
        png_bytes = _plotly_to_png_bytes(fig)
        if not png_bytes:
            continue
        filename = f"{name}.png"
        chart_path = charts_dir / filename
        chart_path.write_bytes(png_bytes)
        try:
            gcs_path = await upload_file(
                chart_path,
                bucket_name=bucket_name,
                destination=f"{analysis_id}/charts/{filename}",
            )
        except Exception as exc:
            logger.warning("Chart upload failed", error=str(exc))
            gcs_path = None
        if gcs_path:
            signed_url = generate_signed_url(
                gcs_path,
                expiration_minutes=settings.REPORT_SIGNED_URL_EXPIRATION_MINUTES,
            )
            chart_urls.append(signed_url or gcs_path)
    return chart_urls


def _extract_plotly_figures(visualizations: dict[str, Any]) -> list[tuple[str, go.Figure]]:
    figures: list[tuple[str, go.Figure]] = []
    effect_plot = visualizations.get("effect_plot") if visualizations else None
    if effect_plot:
        try:
            figures.append(("treatment_effects", go.Figure(effect_plot)))
        except Exception as exc:
            logger.warning("Failed to build plotly figure", error=str(exc))
    return figures


def _plotly_to_png_bytes(fig: go.Figure) -> bytes | None:
    try:
        return pio.to_image(fig, format="png", scale=2)
    except Exception as exc:
        logger.warning("Plotly image export failed", error=str(exc))
        return None


def _build_chart_images(visualizations: dict[str, Any]) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    for name, fig in _extract_plotly_figures(visualizations):
        png_bytes = _plotly_to_png_bytes(fig)
        if not png_bytes:
            continue
        encoded = base64.b64encode(png_bytes).decode("ascii")
        charts.append({"data_uri": f"data:image/png;base64,{encoded}", "alt": f"{name} chart"})
    return charts


def _normalize_summary(summary: dict[str, Any], effects: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summary or {}
    key_findings = summary.get("key_findings") or []
    recommendations = summary.get("recommendations") or []
    return {
        "overview": summary.get("overview", ""),
        "key_findings": _format_key_findings(key_findings, effects),
        "recommendations": [str(rec) for rec in recommendations],
    }


def _format_key_findings(findings: list[Any], effects: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for finding in findings:
        if isinstance(finding, dict):
            treatment = finding.get("treatment") or finding.get("treatment_variable") or "N/A"
            outcome = finding.get("outcome") or finding.get("outcome_variable") or "N/A"
            ate = finding.get("ate")
            ci = finding.get("ci") or [finding.get("ci_lower"), finding.get("ci_upper")]
            ci_lower = ci[0] if len(ci) > 0 else None
            ci_upper = ci[1] if len(ci) > 1 else None
            formatted.append(
                "{treatment} -> {outcome} (ATE: {ate}, CI: {lower} to {upper})".format(
                    treatment=treatment,
                    outcome=outcome,
                    ate=_format_number(ate),
                    lower=_format_number(ci_lower),
                    upper=_format_number(ci_upper),
                )
            )
        else:
            formatted.append(str(finding))
    if not formatted and effects:
        for effect in _normalize_effects(effects):
            formatted.append(
                "{treatment} -> {outcome} (ATE: {ate})".format(
                    treatment=effect["treatment"],
                    outcome=effect["outcome"],
                    ate=_format_number(effect["ate"]),
                )
            )
    return formatted


def _normalize_effects(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for effect in effects or []:
        ci = effect.get("ci")
        ci_lower = None
        ci_upper = None
        if isinstance(ci, (list, tuple)):
            ci_lower = ci[0] if len(ci) > 0 else None
            ci_upper = ci[1] if len(ci) > 1 else None
        raw_ci_lower = effect.get("ci_lower")
        if raw_ci_lower is None:
            raw_ci_lower = effect.get("ate_ci_lower")
        raw_ci_upper = effect.get("ci_upper")
        if raw_ci_upper is None:
            raw_ci_upper = effect.get("ate_ci_upper")
        ci_lower = ci_lower if ci_lower is not None else raw_ci_lower
        ci_upper = ci_upper if ci_upper is not None else raw_ci_upper
        normalized.append(
            {
                "treatment": effect.get("treatment") or effect.get("treatment_variable") or "N/A",
                "outcome": effect.get("outcome") or effect.get("outcome_variable") or "N/A",
                "method": effect.get("method") or "N/A",
                "ate": effect.get("ate"),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    return normalized


def _normalize_graphs(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for graph in graphs or []:
        nodes = []
        for node in graph.get("nodes") or []:
            if isinstance(node, dict):
                nodes.append(
                    {
                        "name": node.get("name") or node.get("id") or str(node),
                        "node_type": node.get("node_type") or node.get("type") or "unknown",
                    }
                )
            else:
                nodes.append({"name": str(node), "node_type": "unknown"})
        edges = []
        for edge in graph.get("edges") or []:
            if isinstance(edge, dict):
                edges.append(
                    {
                        "source": edge.get("source") or edge.get("from") or "N/A",
                        "target": edge.get("target") or edge.get("to") or "N/A",
                        "confidence": edge.get("confidence")
                        if edge.get("confidence") is not None
                        else edge.get("weight"),
                    }
                )
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edges.append({"source": edge[0], "target": edge[1], "confidence": None})
            else:
                edges.append({"source": str(edge), "target": "N/A", "confidence": None})
        normalized.append(
            {
                "method": graph.get("method") or "unknown",
                "confidence": graph.get("confidence"),
                "nodes": nodes,
                "edges": edges,
            }
        )
    return normalized


def _normalize_validation(validation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for result in validation or []:
        recommendations = result.get("recommendations")
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        if not recommendations:
            recommendations = details.get("recommendations")
        if isinstance(recommendations, list):
            recommendations_text = "; ".join(str(item) for item in recommendations)
        elif recommendations:
            recommendations_text = str(recommendations)
        else:
            recommendations_text = ""
        normalized.append(
            {
                "method": result.get("method") or "N/A",
                "passed": bool(result.get("passed")),
                "confidence": result.get("confidence_score")
                if result.get("confidence_score") is not None
                else result.get("confidence"),
                "recommendations": recommendations_text or None,
                "status_icon": _status_icon(bool(result.get("passed"))),
            }
        )
    return normalized


def _format_number(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_confidence(value: Any) -> str:
    return _format_number(value, decimals=2)


def _format_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


def _status_icon(passed: bool) -> str:
    return "✅" if passed else "❌"


def _basic_html_report(summary: dict[str, Any], technical: dict[str, Any]) -> str:
    return f"""
    <html>
      <head><title>Causal Analysis Report</title></head>
      <body>
        <h1>Executive Summary</h1>
        <pre>{json.dumps(summary, indent=2)}</pre>
        <h2>Technical Details</h2>
        <pre>{json.dumps(technical, indent=2)}</pre>
      </body>
    </html>
    """


@lru_cache(maxsize=1)
def _get_template_env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
