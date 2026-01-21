"""LangGraph orchestrator entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.crud.analysis import analysis_crud
from app.db.database import get_session_context
from app.models.analysis import AnalysisStatus
from app.orchestrator.graph import build_graph
from app.orchestrator.state import AnalysisState
from app.services.tracing import traced

if TYPE_CHECKING:
    pass  # Agents imported at runtime to avoid circular imports


@traced("analysis.run", run_type="chain")
async def run_causal_analysis(
    analysis_id: str,
    kaggle_url: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AnalysisState:
    """Run the full causal analysis workflow."""
    async with get_session_context() as session:
        analysis_uuid = uuid.UUID(analysis_id)
        analysis = await analysis_crud.get(session, analysis_uuid)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found")
        await analysis_crud.update(
            session,
            db_obj=analysis,
            obj_in={
                "status": AnalysisStatus.RUNNING,
                "started_at": datetime.now(timezone.utc),
            },
        )
        config = analysis.config or {}

    analysis_types = config.get("analysis_types") or ["treatment_effects", "causal_discovery"]
    if isinstance(analysis_types, str):
        analysis_types = [analysis_types]

    state: AnalysisState = {
        "analysis_id": analysis_id,
        "kaggle_url": kaggle_url,
        "analysis_types": analysis_types,
        "config": config,
        "progress_percent": 0,
        "errors": [],
        "extra_results": {},
    }

    # Import agents here to avoid circular imports
    # (agents.base imports orchestrator.state, which would trigger this __init__)
    from app.agents.data_acquisition import DataAcquisitionAgent
    from app.agents.discovery import CausalDiscoveryAgent
    from app.agents.eda import EDAAgent
    from app.agents.reporting import ReportGenerationAgent
    from app.agents.treatment import TreatmentEffectsAgent
    from app.agents.validation import ValidationAgent

    data_agent = DataAcquisitionAgent()
    eda_agent = EDAAgent()
    discovery_agent = CausalDiscoveryAgent()
    treatment_agent = TreatmentEffectsAgent()
    validation_agent = ValidationAgent()
    report_agent = ReportGenerationAgent()

    graph = build_graph(
        data_agent.run,
        eda_agent.run,
        discovery_agent.run,
        treatment_agent.run,
        validation_agent.run,
        report_agent.run,
    )

    compiled = graph.compile(checkpointer=checkpointer)
    result: AnalysisState = await compiled.ainvoke(state)

    async with get_session_context() as session:
        analysis = await analysis_crud.get(session, analysis_uuid)
        if analysis is None:
            return result

        errors = result.get("errors", [])
        if errors:
            # Critical errors exist - mark analysis as failed
            error_message = "; ".join(errors) if errors else "Unknown error"
            await analysis_crud.update(
                session,
                db_obj=analysis,
                obj_in={
                    "status": AnalysisStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": error_message,
                },
            )
        else:
            await analysis_crud.update(
                session,
                db_obj=analysis,
                obj_in={
                    "status": AnalysisStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                },
            )
    return result


async def mark_analysis_failed(analysis_id: str, error_message: str) -> None:
    async with get_session_context() as session:
        analysis = await analysis_crud.get(session, uuid.UUID(analysis_id))
        if analysis is None:
            return
        await analysis_crud.update(
            session,
            db_obj=analysis,
            obj_in={
                "status": AnalysisStatus.FAILED,
                "error_message": error_message,
            },
        )
