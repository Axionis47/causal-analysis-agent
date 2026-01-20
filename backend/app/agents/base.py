"""Base agent utilities for stage tracking and progress updates."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from app.core.logging import bind_contextvars, get_logger
from app.db.database import get_session_context
from app.models.analysis_stage import StageType
from app.orchestrator.state import AnalysisState
from app.services.quota_manager import quota_manager
from app.services.stages import complete_stage, fail_stage, start_stage
from app.services.tracing import traced

logger = get_logger(__name__)

PARTIAL_OUTPUTS_KEY = "partial_outputs"


@dataclass
class AgentResult:
    state: AnalysisState
    outputs: dict[str, Any]
    message: str


class AgentFailure(Exception):
    """Agent failure that can carry partial outputs for persistence."""

    def __init__(
        self,
        message: str,
        *,
        partial_result: Optional[AgentResult] = None,
        partial_outputs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result
        self.partial_outputs = partial_outputs


class BaseAgent:
    name: str = "base"
    stage: StageType

    async def _track_llm_usage(
        self,
        state: AnalysisState,
        tokens_used: int,
        model: str,
    ) -> None:
        """
        Track LLM token usage and update cost for the analysis.

        Args:
            state: Current analysis state (must contain 'analysis_id')
            tokens_used: Number of tokens consumed
            model: Model name for pricing calculation
        """
        analysis_id = state.get("analysis_id")
        if not analysis_id:
            logger.warning(
                "Cannot track LLM usage",
                agent_name=self.name,
                stage=self.stage.value,
                tokens_used=tokens_used,
                model=model,
            )
            return

        try:
            analysis_uuid = uuid.UUID(analysis_id)
            async with get_session_context() as session:
                await quota_manager.track_llm_usage(
                    session,
                    analysis_uuid,
                    tokens_used,
                    model,
                )
        except Exception as exc:
            logger.warning(
                "Failed to track LLM usage",
                analysis_id=analysis_id,
                tokens_used=tokens_used,
                model=model,
                error=str(exc),
                agent_name=self.name,
                stage=self.stage.value,
            )

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count when actual count is unavailable.

        Uses a simple heuristic of ~4 characters per token.

        Args:
            prompt: Input prompt text
            response: Response text

        Returns:
            Estimated total token count
        """
        return (len(prompt) + len(response)) // 4

    async def run(self, state: AnalysisState) -> AnalysisState:
        analysis_id = state["analysis_id"]
        bind_contextvars(analysis_id=analysis_id)
        existing_partial = state.get(PARTIAL_OUTPUTS_KEY)
        if isinstance(existing_partial, dict):
            existing_partial.pop(self.name, None)
        state_snapshot = copy.deepcopy(state)
        async with get_session_context() as session:
            stage_row = await start_stage(
                session,
                analysis_id=analysis_id,
                stage=self.stage,
                agent_name=self.name,
                message=f"Starting {self.name}...",
            )
            stage_id = stage_row.id
        try:
            result = await traced(self.name)(self._run)(state)
            async with get_session_context() as session:
                await complete_stage(
                    session,
                    stage_id,
                    outputs=result.outputs,
                    message=result.message,
                )
            return result.state
        except Exception as exc:  # noqa: BLE001 - agent failure path
            logger.exception(
                "Agent failed",
                analysis_id=analysis_id,
                stage=self.stage.value,
                agent_name=self.name,
            )
            if isinstance(exc, AgentFailure) and exc.partial_result is not None:
                state = exc.partial_result.state
            partial_outputs = _extract_partial_outputs(state, exc, state_snapshot, self.name)
            if partial_outputs is not None:
                existing = state.get(PARTIAL_OUTPUTS_KEY)
                if isinstance(existing, dict):
                    existing[self.name] = partial_outputs
                    state[PARTIAL_OUTPUTS_KEY] = existing
                else:
                    state[PARTIAL_OUTPUTS_KEY] = {self.name: partial_outputs}
            async with get_session_context() as session:
                await fail_stage(
                    session,
                    stage_id,
                    error=str(exc),
                    message=f"{self.name} failed",
                    outputs=partial_outputs,
                )
            state.setdefault("errors", []).append(str(exc))
            return state

    async def _run(self, state: AnalysisState) -> AgentResult:
        raise NotImplementedError


def _extract_partial_outputs(
    state: AnalysisState,
    exc: Exception,
    snapshot: AnalysisState | None = None,
    agent_name: str | None = None,
) -> dict[str, Any] | None:
    if isinstance(exc, AgentFailure):
        if exc.partial_result is not None:
            return exc.partial_result.outputs
        if exc.partial_outputs is not None:
            return exc.partial_outputs
    candidate = state.get(PARTIAL_OUTPUTS_KEY)
    if isinstance(candidate, dict):
        if agent_name and agent_name in candidate and isinstance(candidate[agent_name], dict):
            return candidate[agent_name]
        if all(isinstance(value, dict) for value in candidate.values()):
            return None
        return candidate
    if snapshot is None:
        return None
    excluded = {"analysis_id", "kaggle_url", "errors", "progress_percent", PARTIAL_OUTPUTS_KEY}
    partial: dict[str, Any] = {}
    for key, value in state.items():
        if key in excluded:
            continue
        if key not in snapshot or snapshot[key] != value:
            partial[key] = value
    return partial or None
