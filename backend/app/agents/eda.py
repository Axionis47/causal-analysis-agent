"""Exploratory data analysis agent with LLM interpretation."""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from app.agents.base import AgentFailure, AgentResult, BaseAgent
from app.agents.llm_utils import call_llm_with_retry
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.crud.dataset import dataset_crud
from app.db.database import get_session_context
from app.llm.router import ROUTER
from app.models.analysis_stage import StageType
from app.models.data_understanding import DataUnderstanding
from app.prompts.eda import build_eda_prompt
from app.services.data_quality import DataQualityValidator
from app.services.data_loader import load_dataframe
from app.services.data_preprocessing import DataPreprocessor, PreprocessingStep
from app.services.retry import is_transient_io_error, retry_async
from app.services.storage import download_to_path, infer_local_path, local_storage_root, upload_to_gcs

logger = get_logger(__name__)


class EDAAgent(BaseAgent):
    name = "EDA Agent"
    stage = StageType.EDA

    async def _run(self, state):
        if state.get("data_characteristics"):
            return AgentResult(
                state=state,
                outputs={"cached": True},
                message="EDA reused from state",
            )

        agent_logger = logger.bind(
            analysis_id=state.get("analysis_id"),
            stage=self.stage.value,
            agent_name=self.name,
        )
        dataset_id = state.get("dataset_id")
        if not dataset_id:
            raise ValueError("dataset_id missing from state")
        dataset_uuid = uuid.UUID(dataset_id)

        async with get_session_context() as session:
            dataset = await dataset_crud.get_with_understanding(session, dataset_uuid)
            if dataset is None:
                raise ValueError("Dataset not found")
            if dataset.data_understanding is not None:
                state["data_characteristics"] = _build_characteristics(dataset.data_understanding)
                state["analysis_types"] = _augment_analysis_types(
                    state.get("analysis_types", []),
                    dataset.data_understanding.columns,
                )
                return AgentResult(
                    state=state,
                    outputs={"cached": True},
                    message="EDA reused from database",
                )

        df = await retry_async(_load_dataframe, dataset, retry_on=is_transient_io_error)
        validation_result = None
        quality_warnings: list[dict[str, Any]] = []
        try:
            validator = DataQualityValidator()
            validation_result = await validator.validate_dataframe(df)
            quality_warnings = validation_result.warnings
            state["quality_warnings"] = quality_warnings

            override_warnings = bool(
                state.get("config", {}).get("override_quality_warnings", False)
            )

            # Auto-resolvable warnings are those that preprocessing will fix automatically
            # These should not block the analysis
            auto_resolvable_metrics = {"variance", "duplicates"}
            blocking_warnings = [
                w for w in quality_warnings
                if w.get("metric") not in auto_resolvable_metrics
            ]
            auto_resolved_warnings = [
                w for w in quality_warnings
                if w.get("metric") in auto_resolvable_metrics
            ]

            if auto_resolved_warnings:
                agent_logger.info(
                    "Auto-resolvable warnings detected (will be fixed in preprocessing)",
                    auto_resolved_count=len(auto_resolved_warnings),
                    metrics=[w.get("metric") for w in auto_resolved_warnings[:5]],
                )

            if blocking_warnings:
                if override_warnings:
                    agent_logger.info(
                        "Override enabled for data quality warnings",
                        warning_count=len(blocking_warnings),
                    )
                else:
                    agent_logger.warning(
                        "Data quality warnings detected",
                        warning_count=len(blocking_warnings),
                    )
            if validation_result and not validation_result.passed:
                if override_warnings:
                    agent_logger.warning(
                        "Data quality errors detected; override ignored",
                        error_count=len(
                            [item for item in quality_warnings if item.get("level") == "error"]
                        ),
                    )
                raise AgentFailure(
                    f"Data quality validation failed: {quality_warnings}",
                    partial_outputs={
                        "quality_warnings": quality_warnings,
                        "passed": validation_result.passed,
                        "can_proceed_with_override": validation_result.can_proceed_with_override,
                    },
                )
            # Only block on warnings that cannot be auto-resolved
            if blocking_warnings and not override_warnings:
                raise AgentFailure(
                    f"Data quality warnings require override: {blocking_warnings}",
                    partial_outputs={
                        "quality_warnings": quality_warnings,
                        "passed": validation_result.passed,
                        "can_proceed_with_override": validation_result.can_proceed_with_override,
                    },
                )
        except AgentFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - validation should not block EDA
            agent_logger.warning(
                "Data quality validation failed; continuing EDA",
                error=str(exc),
            )
        state.setdefault("quality_warnings", quality_warnings)
        summary_stats = _summary_stats(df)
        columns_info = _columns_info(df)
        quality_issues = _quality_issues(df)
        severity_distribution = Counter(issue.get("severity", "unknown") for issue in quality_issues)
        agent_logger.info(
            "Data quality issues detected",
            issue_count=len(quality_issues),
            severity_distribution=dict(severity_distribution),
        )
        correlations = _top_correlations(df)

        treatment_candidates, outcome_candidates, confounder_candidates = _heuristic_candidates(df)
        llm_payload = await _llm_suggestions(
            summary_stats, columns_info, correlations, quality_issues,
            agent_instance=self, state=state,
        )
        if llm_payload:
            treatment_candidates = _normalize_candidates(
                llm_payload.get("treatment_candidates", treatment_candidates)
            )
            outcome_candidates = _normalize_candidates(
                llm_payload.get("outcome_candidates", outcome_candidates)
            )
            confounder_candidates = _normalize_candidates(
                llm_payload.get("confounder_candidates", confounder_candidates)
            )
            quality_issues = llm_payload.get("data_quality_issues", quality_issues)

        recommended_preprocessing = _preprocessing_recommendations(columns_info, quality_issues)
        data_characteristics = {
            "summary_stats": summary_stats,
            "columns": columns_info,
            "quality_issues": quality_issues,
            "quality_warnings": quality_warnings,
            "correlations": correlations,
            "recommended_preprocessing": recommended_preprocessing,
            "treatment_candidates": treatment_candidates,
            "outcome_candidates": outcome_candidates,
            "confounder_candidates": confounder_candidates,
        }
        state["analysis_types"] = _augment_analysis_types(state.get("analysis_types", []), columns_info)
        await _persist_analysis_types(state["analysis_id"], state["analysis_types"])

        # Apply preprocessing if enabled
        enable_preprocessing = state.get("config", {}).get("enable_preprocessing", True)
        preprocessing_applied = False
        preprocessing_steps: list[PreprocessingStep] = []
        preprocessed_row_count = None
        preprocessed_column_count = None
        preprocessing_config: dict[str, Any] = {}

        if enable_preprocessing:
            preprocessing_config = state.get("config", {}).get("preprocessing_config", {})
            preprocessor = DataPreprocessor(config=preprocessing_config)

            df_preprocessed, preprocessing_steps = preprocessor.preprocess(
                df,
                columns_info,
                quality_issues,
                treatment_candidates=treatment_candidates,
                outcome_candidates=outcome_candidates,
            )
            agent_logger.info(
                "Preprocessing applied",
                step_count=len(preprocessing_steps),
                original_shape=df.shape,
                preprocessed_shape=df_preprocessed.shape,
            )

            # Store preprocessed dataset
            preprocessed_filename = f"{dataset.selected_file or 'dataset'}_preprocessed.parquet"
            preprocessed_path = local_storage_root() / str(dataset.analysis_id) / preprocessed_filename
            preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
            df_preprocessed.to_parquet(preprocessed_path, index=False)

            # Upload to GCS
            preprocessed_gcs_path = await upload_to_gcs(
                preprocessed_path,
                f"datasets/{dataset.analysis_id}/{preprocessed_filename}",
            )

            # Update dataset metadata with preprocessed paths
            dataset_metadata_dict = dict(dataset.dataset_metadata) if dataset.dataset_metadata else {}
            dataset_metadata_dict["preprocessed_path"] = str(preprocessed_path)
            dataset_metadata_dict["preprocessed_gcs_path"] = preprocessed_gcs_path

            preprocessing_applied = True
            preprocessed_row_count = len(df_preprocessed)
            preprocessed_column_count = df_preprocessed.shape[1]
            preprocessing_config = preprocessor.config

            # Update state with preprocessed dataset info
            state["preprocessed_dataset_path"] = str(preprocessed_path)
            state["preprocessing_applied"] = True

            # Update dataset metadata in database
            async with get_session_context() as session:
                ds = await dataset_crud.get(session, dataset_uuid)
                if ds is not None:
                    ds.dataset_metadata = dataset_metadata_dict

        # Serialize preprocessing steps for storage
        steps_serialized = [step.to_dict() for step in preprocessing_steps]

        async with get_session_context() as session:
            understanding = DataUnderstanding(
                dataset_id=dataset_uuid,
                summary_stats=summary_stats,
                columns=columns_info,
                quality_issues=quality_issues,
                quality_warnings=quality_warnings,
                recommended_preprocessing=recommended_preprocessing,
                treatment_candidates=treatment_candidates,
                outcome_candidates=outcome_candidates,
                confounder_candidates=confounder_candidates,
                preprocessing_applied=preprocessing_applied,
                preprocessing_steps=steps_serialized,
                preprocessed_row_count=preprocessed_row_count,
                preprocessed_column_count=preprocessed_column_count,
                preprocessing_config=preprocessing_config,
            )
            session.add(understanding)
            dataset = await dataset_crud.get(session, dataset_uuid)
            if dataset is not None:
                dataset.characteristics = data_characteristics

        state["data_characteristics"] = data_characteristics
        return AgentResult(
            state=state,
            outputs={"row_count": summary_stats.get("row_count")},
            message="EDA complete",
        )


async def _load_dataframe(dataset) -> pd.DataFrame:
    local_path = infer_local_path(dataset.dataset_metadata)
    if local_path and local_path.exists():
        return load_dataframe(local_path)
    if dataset.gcs_path:
        local_root = local_storage_root() / str(dataset.analysis_id) / "cache"
        local_path = local_root / (dataset.selected_file or "dataset.csv")
        await download_to_path(dataset.gcs_path, local_path)
        return load_dataframe(local_path)
    raise ValueError("No dataset path available")


def _summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    missing_percent = float(df.isna().mean().mean()) if len(df) else 0.0
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_percent": missing_percent,
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _columns_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    info = []
    for col in df.columns:
        series = df[col]
        dtype = _dtype_label(series)
        entry: dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "missing_percent": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if dtype == "numerical":
            entry["distribution_summary"] = {
                "mean": float(series.mean(skipna=True)),
                "std": float(series.std(skipna=True)),
                "min": float(series.min(skipna=True)),
                "max": float(series.max(skipna=True)),
                "q1": float(series.quantile(0.25)),
                "median": float(series.quantile(0.5)),
                "q3": float(series.quantile(0.75)),
            }
        else:
            entry["distribution_summary"] = {
                "top_values": series.value_counts(dropna=True).head(5).to_dict()
            }
        info.append(entry)
    return info


def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    unique_count = series.nunique(dropna=True)
    if unique_count <= 20:
        return "categorical"
    return "text"


def _quality_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing = float(series.isna().mean())
        if missing > 0.2:
            issues.append(
                {
                    "type": "high_missing",
                    "severity": "high" if missing > 0.5 else "medium",
                    "description": f"{col} has {missing:.0%} missing values",
                    "affected_columns": [col],
                }
            )
        if pd.api.types.is_numeric_dtype(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            if len(outliers) / max(len(series), 1) > 0.01:
                issues.append(
                    {
                        "type": "outliers",
                        "severity": "medium",
                        "description": f"{col} has potential outliers",
                        "affected_columns": [col],
                    }
                )
    return issues


def _top_correlations(df: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    corr = numeric.corr().abs()
    results: list[dict[str, Any]] = []
    for col in corr.columns:
        for other in corr.index:
            if col >= other:
                continue
            value = corr.loc[other, col]
            if value >= 0.6:
                results.append({"a": other, "b": col, "value": float(value)})
    results.sort(key=lambda item: item["value"], reverse=True)
    return results[:10]


def _normalize_candidates(candidates: list[str | dict[str, Any]]) -> list[str]:
    """Normalize candidate list to simple column name strings.

    The LLM may return either:
    - Simple strings: ["Age", "Sex", "Fare"]
    - Rich dicts: [{"column_name": "Age", "reason": "...", "related_columns": [...]}]

    This function extracts just the column names as strings for database storage.
    """
    if not candidates:
        return []
    result = []
    for item in candidates:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and "column_name" in item:
            result.append(str(item["column_name"]))
        elif isinstance(item, dict) and "name" in item:
            result.append(str(item["name"]))
    return result


def _heuristic_candidates(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    treatment_candidates = [
        col for col in df.columns if "treat" in col.lower() or df[col].nunique() == 2
    ]
    outcome_candidates = [
        col
        for col in df.columns
        if any(token in col.lower() for token in ["outcome", "target", "label", "y"])
    ]
    confounder_candidates = [col for col in df.columns if col not in treatment_candidates]
    return (
        list(dict.fromkeys(treatment_candidates))[:3],
        list(dict.fromkeys(outcome_candidates))[:3],
        list(dict.fromkeys(confounder_candidates))[:5],
    )


async def _llm_suggestions(
    summary_stats: dict[str, Any],
    columns: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    agent_instance: EDAAgent | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = build_eda_prompt(summary_stats, columns, correlations, quality_issues)
    analysis_id = state.get("analysis_id") if state else None
    result = await call_llm_with_retry(ROUTER.structured_output, prompt, analysis_id=analysis_id)

    if agent_instance and state:
        response_text = str(result) if result else ""
        estimated_tokens = agent_instance._estimate_tokens(prompt, response_text)
        logger.info(
            "LLM suggestion call",
            analysis_id=state.get("analysis_id"),
            stage=agent_instance.stage.value,
            agent_name=agent_instance.name,
            prompt_length=len(prompt),
            response_length=len(response_text),
            estimated_tokens=estimated_tokens,
        )

        if result:
            await agent_instance._track_llm_usage(state, estimated_tokens, "gpt-4")

    return result


def _preprocessing_recommendations(
    columns: list[dict[str, Any]], quality_issues: list[dict[str, Any]]
) -> list[str]:
    steps: list[str] = []
    if any(issue["type"] == "high_missing" for issue in quality_issues):
        steps.append("impute_missing")
    if any(col["dtype"] in {"categorical", "text"} for col in columns):
        steps.append("encode_categorical")
    if any(col["dtype"] == "numerical" for col in columns):
        steps.append("scale_numeric")
    return steps


def _build_characteristics(understanding: DataUnderstanding) -> dict[str, Any]:
    return {
        "summary_stats": understanding.summary_stats,
        "columns": understanding.columns,
        "quality_issues": understanding.quality_issues,
        "quality_warnings": understanding.quality_warnings,
        "recommended_preprocessing": understanding.recommended_preprocessing,
        "treatment_candidates": understanding.treatment_candidates,
        "outcome_candidates": understanding.outcome_candidates,
        "confounder_candidates": understanding.confounder_candidates,
        "preprocessing_applied": understanding.preprocessing_applied,
        "preprocessing_steps": understanding.preprocessing_steps,
        "preprocessed_row_count": understanding.preprocessed_row_count,
        "preprocessed_column_count": understanding.preprocessed_column_count,
    }


def _augment_analysis_types(existing: list[str], columns: list[dict[str, Any]]) -> list[str]:
    analysis_types = list({item.lower() for item in existing}) if existing else []
    if any(col.get("dtype") == "datetime" for col in columns):
        if "time_varying" not in analysis_types:
            analysis_types.append("time_varying")
    if any("mediator" in col.get("name", "").lower() for col in columns):
        if "mediation" not in analysis_types:
            analysis_types.append("mediation")
    return analysis_types


async def _persist_analysis_types(analysis_id: str, analysis_types: list[str]) -> None:
    analysis_uuid = uuid.UUID(analysis_id)
    async with get_session_context() as session:
        analysis = await analysis_crud.get(session, analysis_uuid)
        if analysis is None:
            return
        config = analysis.config or {}
        config["analysis_types"] = analysis_types
        await analysis_crud.update(session, db_obj=analysis, obj_in={"config": config})
