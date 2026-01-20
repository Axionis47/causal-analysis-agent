"""API routes for analysis operations."""

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Annotated, AsyncIterator

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.data_acquisition import (
    _download_with_retry,
    _parse_kaggle_dataset_id,
    _select_file,
)
from app.api.dependencies import check_analysis_quota, get_rate_limit_headers
from app.core.auth import get_current_user, get_current_user_optional
from app.core.config import settings
from app.crud.analysis import analysis_crud
from app.crud.analysis_version import analysis_version_crud
from app.crud.audit_log import log_analysis_action
from app.db.database import get_async_session
from app.models.audit_log import AuditAction
from app.main import limiter
from app.models.analysis import Analysis, AnalysisStatus
from app.schemas.analysis import (
    AnalysisCreate,
    AnalysisListResponse,
    AnalysisResponse,
    DataQualityPreviewRequest,
    DataQualityPreviewResponse,
    HypothesisTestRequest,
)
from app.schemas.preprocessing import (
    PreprocessingConfig,
    PreprocessingPreviewRequest,
    PreprocessingPreviewResponse,
    PreprocessingStepSchema,
)
from app.services.comparison import compare_analyses
from app.services.data_preprocessing import DataPreprocessor
from app.services.credentials import get_kaggle_credentials
from app.services.data_quality import DataQualityValidator
from app.services.progress import ProgressEvent, stream_progress
from app.services.quota_manager import quota_manager
from app.tasks.analysis import run_analysis_task
from app.worker import celery_app

router = APIRouter(prefix="/api/v1/analyses", tags=["analyses"])
logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 30.0
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _load_preview_dataframe(path: Path, sample_rows: int) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=sample_rows)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=sample_rows)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(sample_rows)
    raise ValueError("Unsupported file type")


def _estimate_duration_seconds(file_size_bytes: int | None) -> int | None:
    if not file_size_bytes:
        return None
    return int(max(30, min(3600, file_size_bytes / 1_000_000 * 5)))


async def _analysis_event_generator(analysis: Analysis) -> AsyncIterator[dict[str, str]]:
    initial = ProgressEvent(
        analysis_id=str(analysis.id),
        stage="init",
        status=analysis.status.value,
        progress_percent=0,
        message="Connecting to progress stream",
    )
    yield {"event": "progress", "data": initial.to_json()}

    progress_iter = stream_progress(str(analysis.id))
    heartbeat_task = asyncio.create_task(asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS))
    progress_task = asyncio.create_task(progress_iter.__anext__())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.PROGRESS_STREAM_TIMEOUT
    sent_disconnect = False
    last_status = None

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            done, _ = await asyncio.wait_for(
                asyncio.wait(
                    {progress_task, heartbeat_task},
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                timeout=remaining,
            )
            if heartbeat_task in done:
                yield {"comment": "heartbeat"}
                heartbeat_task = asyncio.create_task(asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS))
            if progress_task in done:
                try:
                    event = progress_task.result()
                except StopAsyncIteration:
                    break
                yield {"event": "progress", "data": event.to_json()}
                last_status = event.status
                progress_task = asyncio.create_task(progress_iter.__anext__())
        if not sent_disconnect:
            reason = "completed" if last_status in TERMINAL_STATUSES else "stream_end"
            yield {
                "event": "disconnected",
                "data": json.dumps(
                    {"analysis_id": str(analysis.id), "reason": reason}
                ),
            }
            sent_disconnect = True
    except asyncio.TimeoutError:
        logger.info(
            "Progress stream timed out",
            extra={"analysis_id": str(analysis.id)},
        )
        yield {
            "event": "disconnected",
            "data": json.dumps(
                {"analysis_id": str(analysis.id), "reason": "timeout"}
            ),
        }
        sent_disconnect = True
    except asyncio.CancelledError:
        logger.info(
            "Progress stream cancelled by client",
            extra={"analysis_id": str(analysis.id)},
        )
        raise
    except Exception as exc:
        logger.exception(
            "Progress stream error",
            exc_info=exc,
            extra={"analysis_id": str(analysis.id)},
        )
        yield {
            "event": "disconnected",
            "data": json.dumps({"analysis_id": str(analysis.id), "reason": "error"}),
        }
        sent_disconnect = True
    finally:
        for task in (heartbeat_task, progress_task):
            task.cancel()
        await asyncio.gather(heartbeat_task, progress_task, return_exceptions=True)
        await progress_iter.aclose()


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new causal analysis",
    responses={
        201: {
            "description": "Analysis created successfully and queued for processing",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
                        "status": "pending",
                        "config": {"override_quality_warnings": False},
                        "created_at": "2026-01-20T12:00:00Z",
                        "llm_tokens_used": 0,
                        "estimated_cost": 0.0,
                    }
                }
            },
        },
        400: {"description": "Invalid Kaggle URL format"},
        401: {"description": "Authentication required - missing or invalid Bearer token"},
        429: {
            "description": "Rate limit exceeded - hourly, daily, or concurrent limit reached",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded. Try again in 3600 seconds.",
                        "retry_after": 3600,
                    }
                }
            },
        },
        500: {"description": "Database error or failed to enqueue analysis task"},
    },
)
@limiter.limit(f"{settings.RATE_LIMIT_ANALYSES_PER_HOUR}/hour")
async def create_analysis(
    request: Request,
    analysis_in: AnalysisCreate,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: Annotated[uuid.UUID, Depends(check_analysis_quota)],
) -> AnalysisResponse:
    """
    Create a new causal analysis and enqueue background processing.

    This endpoint creates a new analysis record, initializes version tracking (v1),
    and queues a Celery task to run the multi-agent analysis pipeline.

    **Rate Limiting:**
    - Maximum analyses per hour: Configurable via RATE_LIMIT_ANALYSES_PER_HOUR
    - Maximum analyses per day: Configurable via RATE_LIMIT_ANALYSES_PER_DAY
    - Maximum concurrent running analyses: Configurable via RATE_LIMIT_CONCURRENT_ANALYSES

    **Rate Limit Headers in Response:**
    - `X-RateLimit-Limit-Hourly`: Maximum analyses allowed per hour
    - `X-RateLimit-Remaining-Hourly`: Remaining analyses this hour
    - `X-RateLimit-Limit-Daily`: Maximum analyses allowed per day
    - `X-RateLimit-Remaining-Daily`: Remaining analyses today
    - `X-RateLimit-Reset-Hourly`: Unix timestamp when hourly limit resets

    **Request Example:**
    ```json
    {
        "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
        "config": {
            "override_quality_warnings": false,
            "treatment_variable": "species",
            "outcome_variable": "sepal_length",
            "analysis_types": ["treatment_effects", "causal_discovery"]
        }
    }
    ```

    **Workflow:**
    1. Validates Kaggle URL format
    2. Creates analysis record in database
    3. Creates initial version (v1) for configuration tracking
    4. Logs creation action for audit trail
    5. Queues Celery task for background processing
    6. Returns analysis object with rate limit headers
    """
    # Note: Concurrent analysis limit and rate limits are checked atomically
    # in the check_analysis_quota dependency via quota_manager.reserve_analysis_quota()

    # Extract request metadata for audit logging
    client_ip = request.client.host if request.client else None
    user_agent_header = request.headers.get("user-agent")
    request_id = request.headers.get("x-request-id")

    try:
        config = analysis_in.config.model_dump() if analysis_in.config else {}
        analysis = await analysis_crud.create(
            db,
            obj_in={
                "user_id": user_id,
                "kaggle_url": analysis_in.kaggle_url,
                "config": config,
                "llm_tokens_used": 0,
                "estimated_cost": 0.0,
            },
        )
        # Create initial version (v1) for the new analysis
        await analysis_version_crud.create_version(
            db,
            analysis_id=analysis.id,
            config=config,
            changed_by=user_id,
            change_summary="Initial configuration",
            is_manual_snapshot=False,
            auto_commit=True,
        )
        # Log the creation action
        await log_analysis_action(
            db,
            analysis_id=analysis.id,
            action=AuditAction.CREATE,
            user_id=user_id,
            changes={
                "kaggle_url": analysis_in.kaggle_url,
                "config": config,
            },
            ip_address=client_ip,
            user_agent=user_agent_header,
            request_id=request_id,
            description=f"Created analysis for {analysis_in.kaggle_url}",
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while creating analysis",
        ) from exc

    try:
        task = run_analysis_task.delay(str(analysis.id), analysis.kaggle_url)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue analysis task",
        ) from exc

    analysis = await analysis_crud.update(
        db,
        db_obj=analysis,
        obj_in={"celery_task_id": task.id},
    )

    # Note: Rate limit increment is handled atomically in check_analysis_quota dependency
    # via quota_manager.reserve_analysis_quota() - no separate record_analysis_started() needed

    # Add rate limit headers to response
    rate_limit_headers = await get_rate_limit_headers(user_id)
    for header_name, header_value in rate_limit_headers.items():
        response.headers[header_name] = header_value

    return analysis


@router.get(
    "",
    response_model=AnalysisListResponse,
    summary="List user's analyses",
    responses={
        200: {
            "description": "Paginated list of analyses",
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
                                "status": "completed",
                                "created_at": "2026-01-20T12:00:00Z",
                            }
                        ],
                        "total": 1,
                        "skip": 0,
                        "limit": 100,
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
    },
)
async def list_analyses(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> AnalysisListResponse:
    """
    List all analyses for the current authenticated user with pagination.

    Returns analyses ordered by creation date (newest first). Use `skip` and `limit`
    parameters to paginate through large result sets.
    """
    total_result = await db.execute(
        select(func.count()).select_from(Analysis).where(Analysis.user_id == user_id)
    )
    total = total_result.scalar_one()
    items = await analysis_crud.get_by_user(db, user_id, skip=skip, limit=limit)
    return AnalysisListResponse(items=items, total=total, skip=skip, limit=limit)


@router.get(
    "/{id}",
    response_model=AnalysisResponse,
    summary="Get analysis details",
    responses={
        200: {"description": "Analysis details with all related entities"},
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def get_analysis(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> AnalysisResponse:
    """
    Get detailed information about a specific analysis.

    Returns the analysis with all related entities loaded, including:
    - Datasets associated with the analysis
    - Causal graphs discovered
    - Treatment effect estimates
    - Validation results
    - Generated reports

    Only the analysis owner can access this endpoint.
    """
    analysis = await analysis_crud.get_with_relations(db, id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if analysis.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return analysis


@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an analysis",
    responses={
        204: {"description": "Analysis deleted successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to delete this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def delete_analysis(
    id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> Response:
    """
    Delete an analysis and cancel any running background task.

    This endpoint:
    1. Cancels any running Celery task associated with the analysis
    2. Logs the deletion action for audit purposes
    3. Marks the analysis as cancelled
    4. Permanently deletes the analysis record

    **Warning:** This action is irreversible. All associated data including
    datasets, graphs, reports, versions, and comments will be deleted.
    """
    analysis = await analysis_crud.get(db, id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if analysis.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    # Extract request metadata for audit logging
    client_ip = request.client.host if request.client else None
    user_agent_header = request.headers.get("user-agent")
    request_id = request.headers.get("x-request-id")

    if analysis.status == AnalysisStatus.RUNNING and analysis.celery_task_id:
        celery_app.control.revoke(analysis.celery_task_id, terminate=True)

    # Log the deletion before actually deleting
    await log_analysis_action(
        db,
        analysis_id=id,
        action=AuditAction.DELETE,
        user_id=user_id,
        changes={
            "kaggle_url": analysis.kaggle_url,
            "status": analysis.status.value,
        },
        ip_address=client_ip,
        user_agent=user_agent_header,
        request_id=request_id,
        description=f"Deleted analysis for {analysis.kaggle_url}",
    )

    await analysis_crud.update(
        db,
        db_obj=analysis,
        obj_in={"status": AnalysisStatus.CANCELLED},
    )
    await analysis_crud.delete(db, id=id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{id}/stream",
    summary="Stream analysis progress",
    responses={
        200: {
            "description": "Server-Sent Events stream of progress updates",
            "content": {
                "text/event-stream": {
                    "example": 'event: progress\ndata: {"analysis_id":"550e8400-e29b-41d4-a716-446655440000","stage":"eda","status":"running","progress_percent":35,"message":"Analyzing columns","error":null}\n\n'
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
    },
)
async def stream_analysis(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user_optional),
) -> EventSourceResponse:
    """
    Stream real-time analysis progress updates via Server-Sent Events (SSE).

    **Event Types:**

    1. **progress** - Analysis progress update
       ```json
       {
           "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
           "stage": "eda",
           "status": "running",
           "progress_percent": 35,
           "message": "Analyzing column distributions",
           "error": null
       }
       ```

    2. **disconnected** - Stream ending notification
       ```json
       {
           "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
           "reason": "completed"
       }
       ```
       Reasons: `completed`, `timeout`, `error`, `stream_end`

    3. **heartbeat** - Keep-alive comment (every 30 seconds)

    **Analysis Stages:**
    - `init`: Initializing analysis
    - `data_acquisition`: Downloading dataset from Kaggle
    - `eda`: Exploratory data analysis
    - `causal_discovery`: Running causal discovery algorithms (PC, GES, FCI)
    - `treatment_effects`: Estimating treatment effects (PSM, DR, IV)
    - `validation`: Running refutation tests
    - `report_generation`: Generating reports

    **Reconnection Strategy:**
    Clients should implement exponential backoff with a maximum of 5 retries.
    The stream times out after PROGRESS_STREAM_TIMEOUT seconds (default: 30 minutes).

    **Example JavaScript Client:**
    ```javascript
    const eventSource = new EventSource('/api/v1/analyses/{id}/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateProgress(data.progress_percent, data.message);
    };
    eventSource.addEventListener('disconnected', (event) => {
        const data = JSON.parse(event.data);
        if (data.reason === 'completed') {
            fetchResults();
        }
        eventSource.close();
    });
    ```
    """
    analysis = await analysis_crud.get(db, id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if analysis.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return EventSourceResponse(_analysis_event_generator(analysis))


@router.post(
    "/{id}/hypothesis-test",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run hypothesis test",
    responses={
        201: {
            "description": "New analysis created with hypothesis test configuration",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440001",
                        "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
                        "status": "pending",
                        "config": {
                            "parent_analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                            "hypothesis_test": {
                                "treatment": "species",
                                "outcome": "sepal_length",
                                "confounders": ["petal_length", "petal_width"],
                            },
                        },
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access the original analysis"},
        404: {"description": "Original analysis not found"},
        500: {"description": "Failed to enqueue hypothesis test task"},
    },
)
async def run_hypothesis_test(
    id: uuid.UUID,
    request: HypothesisTestRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> AnalysisResponse:
    """
    Run a hypothesis test by cloning an analysis with modified variables.

    This creates a new analysis using the same dataset but with different
    treatment, outcome, and/or confounder variables. Useful for:

    - Testing alternative causal hypotheses
    - Comparing effects of different treatments
    - Sensitivity analysis with different confounders

    **Request Example:**
    ```json
    {
        "treatment": "education_level",
        "outcome": "income",
        "confounders": ["age", "gender", "location"],
        "analysis_types": ["treatment_effects"]
    }
    ```

    **Workflow:**
    1. Clones configuration from the original analysis
    2. Updates treatment, outcome, and confounder variables
    3. Links new analysis to parent via `parent_analysis_id`
    4. Queues new analysis task

    The new analysis can be compared with the original using the
    `/api/v1/analyses/compare` endpoint.
    """
    # Get original analysis
    original = await analysis_crud.get_with_relations(db, id)
    if original is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if original.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    # Clone config with hypothesis parameters
    new_config = dict(original.config) if original.config else {}
    new_config["parent_analysis_id"] = str(original.id)
    new_config["hypothesis_test"] = {
        "treatment": request.treatment,
        "outcome": request.outcome,
        "confounders": request.confounders,
    }
    new_config["treatment_variable"] = request.treatment
    new_config["outcome_variable"] = request.outcome
    new_config["confounders"] = request.confounders
    new_config["analysis_types"] = request.analysis_types

    # Create new analysis
    analysis = await analysis_crud.create(
        db,
        obj_in={
            "user_id": user_id,
            "kaggle_url": original.kaggle_url,
            "config": new_config,
            "llm_tokens_used": 0,
            "estimated_cost": 0.0,
        },
    )

    # Enqueue task
    try:
        task = run_analysis_task.delay(str(analysis.id), original.kaggle_url)
        analysis = await analysis_crud.update(
            db,
            db_obj=analysis,
            obj_in={"celery_task_id": task.id},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue hypothesis test task",
        ) from exc

    logger.info(
        "Hypothesis test started",
        original_analysis_id=str(original.id),
        new_analysis_id=str(analysis.id),
        treatment=request.treatment,
        outcome=request.outcome,
    )

    return analysis


@router.get(
    "/compare",
    summary="Compare multiple analyses",
    responses={
        200: {
            "description": "Comparison metrics for the specified analyses",
            "content": {
                "application/json": {
                    "example": {
                        "analyses": [
                            {"id": "...", "kaggle_url": "...", "status": "completed"},
                            {"id": "...", "kaggle_url": "...", "status": "completed"},
                        ],
                        "comparison_metrics": {
                            "graph_similarity": 0.85,
                            "ate_differences": [
                                {
                                    "treatment": "x",
                                    "outcome": "y",
                                    "ate_1": 0.5,
                                    "ate_2": 0.52,
                                    "difference": 0.02,
                                }
                            ],
                            "confidence_deltas": {"analysis_1": 0.8, "analysis_2": 0.85},
                        },
                        "summary": {
                            "num_analyses": 2,
                            "graph_similarity_percent": 85.0,
                            "agreement_level": "high",
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid analysis IDs or wrong number of analyses (need 2-3)"},
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access one or more analyses"},
        404: {"description": "One or more analyses not found"},
    },
)
async def compare_analyses_endpoint(
    ids: str = Query(..., description="Comma-separated analysis IDs (2-3)", example="id1,id2,id3"),
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> dict:
    """
    Compare 2-3 analyses and return detailed comparison metrics.

    **Comparison Metrics:**

    - **graph_similarity**: Jaccard similarity between causal graphs (0-1)
      - 1.0 = identical graphs
      - 0.0 = completely different graphs

    - **ate_differences**: Differences in Average Treatment Effect estimates
      - Shows ATE from each analysis for matching treatment-outcome pairs
      - Computes absolute difference

    - **confidence_deltas**: Differences in confidence scores
      - Based on validation test results and refutation passes

    **Agreement Levels:**
    - `high`: Graph similarity > 0.7 and ATE differences < 0.1
    - `moderate`: Graph similarity 0.4-0.7 or ATE differences 0.1-0.3
    - `low`: Graph similarity < 0.4 or ATE differences > 0.3

    **Use Cases:**
    - Compare hypothesis test results with original analysis
    - Validate findings across different analysis configurations
    - Assess robustness of causal conclusions
    """
    analysis_ids = [id.strip() for id in ids.split(",") if id.strip()]

    if len(analysis_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 analysis IDs required",
        )
    if len(analysis_ids) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 3 analyses can be compared",
        )

    # Fetch all analyses
    analyses_data = []
    for aid in analysis_ids:
        try:
            analysis_uuid = uuid.UUID(aid)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid analysis ID: {aid}",
            )

        analysis = await analysis_crud.get_with_relations(db, analysis_uuid)
        if analysis is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis not found: {aid}",
            )
        if analysis.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized to access analysis: {aid}",
            )

        # Convert to dict for comparison service
        analyses_data.append({
            "id": str(analysis.id),
            "kaggle_url": analysis.kaggle_url,
            "status": analysis.status.value,
            "treatment_effects": [
                {
                    "treatment": e.treatment_variable,
                    "outcome": e.outcome_variable,
                    "method": e.method.value,
                    "ate": e.ate,
                    "ci_lower": e.ate_ci_lower,
                    "ci_upper": e.ate_ci_upper,
                }
                for e in analysis.treatment_effects
            ],
            "causal_graphs": [
                {
                    "method": g.method.value,
                    "nodes": g.nodes,
                    "edges": g.edges,
                    "confidence": g.confidence,
                }
                for g in analysis.causal_graphs
            ],
            "confidence_score": (
                analysis.causal_graphs[0].confidence
                if analysis.causal_graphs
                else None
            ),
            "created_at": analysis.created_at.isoformat(),
        })

    # Run comparison
    result = compare_analyses(analyses_data)

    return result


@router.post(
    "/preview",
    response_model=DataQualityPreviewResponse,
    summary="Preview data quality",
    responses={
        200: {
            "description": "Data quality assessment with validation warnings",
            "content": {
                "application/json": {
                    "example": {
                        "passed": True,
                        "warnings": [
                            {
                                "level": "warning",
                                "code": "HIGH_MISSING_RATE",
                                "message": "Column 'age' has 15% missing values",
                                "column": "age",
                            }
                        ],
                        "can_proceed_with_override": True,
                        "dataset_info": {
                            "row_count": 1000,
                            "column_count": 10,
                            "columns_preview": ["id", "age", "income", "education"],
                        },
                        "estimated_duration_seconds": 300,
                    }
                }
            },
        },
        400: {
            "description": "Invalid Kaggle URL or credentials not configured",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_url": {"value": {"detail": "Unable to parse Kaggle dataset ID"}},
                        "no_credentials": {"value": {"detail": "Kaggle credentials not configured"}},
                        "download_failed": {"value": {"detail": "Failed to download Kaggle dataset"}},
                    }
                }
            },
        },
        401: {"description": "Authentication required"},
        422: {
            "description": "Unsupported file type or unreadable dataset",
            "content": {
                "application/json": {
                    "example": {"detail": "Unsupported dataset file type"}
                }
            },
        },
    },
)
async def preview_data_quality(
    request: Request,
    preview_request: DataQualityPreviewRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> DataQualityPreviewResponse:
    """
    Preview dataset quality before creating an analysis.

    Downloads a sample of the Kaggle dataset and runs validation checks to identify
    potential issues before committing to a full analysis.

    **Quality Checks Performed:**
    - Minimum row count (100 rows required)
    - Minimum numeric columns (2 required for causal analysis)
    - Missing value rate per column
    - Duplicate rows detection
    - Data type inference

    **Warning Levels:**
    - `error`: Analysis cannot proceed (e.g., < 100 rows)
    - `warning`: Analysis can proceed with caution (e.g., high missing rate)
    - `info`: Informational notice (e.g., detected data types)

    **Override Behavior:**
    If `can_proceed_with_override` is `true`, you can create an analysis despite
    warnings by setting `override_quality_warnings: true` in the analysis config.

    **Request Example:**
    ```json
    {
        "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
        "sample_rows": 10000
    }
    ```

    **Supported File Types:** CSV, Parquet, Excel (.xlsx, .xls)
    """
    logger.info(
        "Data quality preview requested",
        extra={"user_id": str(user_id), "kaggle_url": preview_request.kaggle_url},
    )
    logger.info(
        "Preview conversion tracking",
        extra={
            "metric_name": "preview_to_analysis_conversion",
            "event": "preview_requested",
            "user_id": str(user_id),
        },
    )

    dataset_id, is_competition = _parse_kaggle_dataset_id(preview_request.kaggle_url)
    if dataset_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to parse Kaggle dataset ID",
        )

    credentials = await get_kaggle_credentials(db, None)
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kaggle credentials not configured",
        )

    username, api_key = credentials
    temp_dir: Path | None = None
    try:
        temp_dir = await _download_with_retry(
            dataset_id,
            username,
            api_key,
            is_competition,
            analysis_id=str(user_id),
            agent_name="DataQualityPreview",
            stage="preview",
        )
    except Exception as exc:  # noqa: BLE001 - surface download issues as 400
        logger.warning(
            "Kaggle download failed during preview",
            extra={"user_id": str(user_id), "kaggle_url": preview_request.kaggle_url},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to download Kaggle dataset",
        ) from exc

    try:
        selected_file = _select_file(temp_dir)
        if selected_file is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported dataset file type",
            )
        sample_rows = preview_request.sample_rows or 10000
        file_size_bytes = selected_file.stat().st_size
        df = _load_preview_dataframe(selected_file, sample_rows)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - pandas parsing errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported or unreadable dataset file",
        ) from exc
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    validator = DataQualityValidator()
    validation_result = await validator.validate_dataframe(df)
    warning_count = len(validation_result.warnings)
    error_count = len(
        [item for item in validation_result.warnings if item.get("level") == "error"]
    )

    dataset_info = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns_preview": [str(col) for col in list(df.columns)[:10]],
    }

    logger.info(
        "Data quality preview completed",
        extra={
            "user_id": str(user_id),
            "kaggle_url": preview_request.kaggle_url,
            "passed": validation_result.passed,
            "warning_count": warning_count,
            "error_count": error_count,
        },
    )

    return DataQualityPreviewResponse(
        passed=validation_result.passed,
        warnings=validation_result.warnings,
        can_proceed_with_override=validation_result.can_proceed_with_override,
        dataset_info=dataset_info,
        estimated_duration_seconds=_estimate_duration_seconds(file_size_bytes),
    )


@router.post(
    "/{id}/preview-preprocessing",
    response_model=PreprocessingPreviewResponse,
    summary="Preview preprocessing transformations",
    responses={
        200: {
            "description": "Preview of preprocessing transformations",
            "content": {
                "application/json": {
                    "example": {
                        "original_shape": [1000, 10],
                        "preprocessed_shape": [950, 12],
                        "steps_applied": [
                            {
                                "step_type": "imputation",
                                "affected_columns": ["age", "income"],
                                "parameters": {"strategy": "median"},
                                "statistics": {"rows_affected": 50},
                            },
                            {
                                "step_type": "encoding",
                                "affected_columns": ["gender"],
                                "parameters": {"method": "one_hot"},
                                "statistics": {"new_columns": ["gender_M", "gender_F"]},
                            },
                        ],
                        "sample_data": {
                            "rows": [{"age": 25, "income": 50000, "gender_M": 1, "gender_F": 0}]
                        },
                        "column_changes": {
                            "added": ["gender_M", "gender_F"],
                            "removed": ["gender"],
                            "modified": ["age", "income"],
                        },
                    }
                }
            },
        },
        400: {"description": "No dataset associated with analysis or file not found"},
        401: {"description": "Authentication required"},
        403: {"description": "Not authorized to access this analysis"},
        404: {"description": "Analysis not found"},
        422: {"description": "Failed to load dataset or preprocessing failed"},
    },
)
async def preview_preprocessing(
    id: uuid.UUID,
    preview_request: PreprocessingPreviewRequest,
    db: AsyncSession = Depends(get_async_session),
    user_id: uuid.UUID = Depends(get_current_user),
) -> PreprocessingPreviewResponse:
    """
    Preview preprocessing transformations without saving results.

    Applies the specified preprocessing configuration to the analysis dataset
    and returns a preview of the transformed data. Use this to understand
    how preprocessing will affect your data before running a full analysis.

    **Preprocessing Options:**

    - **imputation**: Fill missing values
      - `strategy`: "mean", "median", "mode", "constant"
      - `fill_value`: Value for constant strategy

    - **outlier_handling**: Handle outliers
      - `method`: "clip", "remove", "winsorize"
      - `threshold`: Number of standard deviations

    - **encoding**: Encode categorical variables
      - `method`: "one_hot", "label", "target"

    - **scaling**: Scale numeric features
      - `method`: "standard", "minmax", "robust"

    - **feature_engineering**: Create new features
      - `interactions`: Create interaction terms
      - `polynomial`: Create polynomial features

    **Request Example:**
    ```json
    {
        "config": {
            "imputation": {"strategy": "median"},
            "outlier_handling": {"method": "winsorize", "threshold": 3},
            "encoding": {"method": "one_hot"},
            "scaling": {"method": "standard"}
        }
    }
    ```

    **Response includes:**
    - Original vs preprocessed data dimensions
    - Steps applied with parameters and statistics
    - Sample of transformed data (first 10 rows)
    - Summary of column changes (added, removed, modified)
    """
    from app.crud.dataset import dataset_crud
    from app.services.data_loader import load_dataframe
    from app.services.storage import infer_local_path

    # Get the analysis
    analysis = await analysis_crud.get_with_relations(db, id)
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if analysis.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    # Get the dataset
    if not analysis.datasets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No dataset associated with this analysis",
        )

    dataset = analysis.datasets[0]
    local_path = infer_local_path(dataset.metadata)
    if not local_path or not local_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset file not found locally",
        )

    # Load the dataframe
    try:
        df = load_dataframe(local_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to load dataset: {str(exc)}",
        ) from exc

    # Build columns_info from dataframe
    columns_info = []
    for col in df.columns:
        series = df[col]
        dtype = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        if pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
        columns_info.append({
            "name": col,
            "inferred_type": dtype,
            "missing_rate": float(series.isna().mean()),
        })

    # Run preprocessing with provided config (don't save results)
    config = preview_request.config.model_dump()
    preprocessor = DataPreprocessor(config=config)

    try:
        df_preprocessed, steps = preprocessor.preprocess(
            df,
            columns_info,
            quality_issues=[],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Preprocessing failed: {str(exc)}",
        ) from exc

    # Build response
    steps_schemas = [
        PreprocessingStepSchema(
            step_type=step.step_type,
            affected_columns=step.affected_columns,
            parameters=step.parameters,
            statistics=step.statistics,
        )
        for step in steps
    ]

    # Determine column changes
    original_cols = set(df.columns)
    preprocessed_cols = set(df_preprocessed.columns)
    column_changes = {
        "added": list(preprocessed_cols - original_cols),
        "removed": list(original_cols - preprocessed_cols),
        "modified": [
            col for col in original_cols & preprocessed_cols
            if not df[col].equals(df_preprocessed[col])
        ] if len(df) == len(df_preprocessed) else list(original_cols & preprocessed_cols),
    }

    # Convert sample data to serializable format
    sample_data = df_preprocessed.head(10).to_dict(orient="records")

    return PreprocessingPreviewResponse(
        original_shape=[len(df), df.shape[1]],
        preprocessed_shape=[len(df_preprocessed), df_preprocessed.shape[1]],
        steps_applied=steps_schemas,
        sample_data={"rows": sample_data},
        column_changes=column_changes,
    )
