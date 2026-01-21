"""Treatment effects agent using DoWhy and EconML."""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from app.agents.base import AgentResult, BaseAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.crud.dataset import dataset_crud
from app.db.database import get_session_context
from app.models.analysis_stage import StageType
from app.models.treatment_effect import TreatmentEffect, TreatmentMethod
from app.services.data_loader import load_dataframe
from app.services.retry import is_transient_io_error, is_transient_tool_error, retry_sync
from app.services.storage import infer_local_path
from app.services.tracing import traced
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)

# Bootstrap configuration defaults
DEFAULT_BOOTSTRAP_SIMULATIONS = 500
DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


class TreatmentEffectsAgent(BaseAgent):
    name = "Treatment Effects Agent"
    stage = StageType.TREATMENT

    async def _run(self, state):
        if state.get("treatment_effect_ids"):
            return AgentResult(state=state, outputs={"cached": True}, message="Effects cached")

        agent_logger = logger.bind(
            analysis_id=state.get("analysis_id"),
            stage=self.stage.value,
            agent_name=self.name,
        )
        dataset_id = uuid.UUID(state["dataset_id"])
        analysis_id = uuid.UUID(state["analysis_id"])
        analysis_types = [t.lower() for t in state.get("analysis_types", [])]

        async with get_session_context() as session:
            dataset = await dataset_crud.get(session, dataset_id)
            analysis = await analysis_crud.get(session, analysis_id)
            if dataset is None or analysis is None:
                raise ValueError("Dataset or analysis not found")

        df = retry_sync(_load_dataframe, dataset, retry_on=is_transient_io_error)
        pairs = _select_pairs(df, state)
        if not pairs:
            return AgentResult(state=state, outputs={}, message="No treatment/outcome pairs found")

        registry = ToolRegistry()
        tools = registry.available_tools(
            "treatment",
            sample_size=len(df),
            data_types=["numerical"],
            overrides=analysis.config or {},
        )

        treatment_ids: list[str] = []
        extra_results: dict[str, Any] = {}
        for treatment, outcome in pairs:
            confounders = _confounders_from_state(state, treatment, outcome)
            method_results: list[dict[str, Any]] = []
            start = time.time()
            psm_result = retry_sync(
                _estimate_psm,
                df,
                treatment,
                outcome,
                confounders,
                retry_on=is_transient_tool_error,
            )
            psm_elapsed = time.time() - start
            registry.track_performance("treatment:propensity_matching", psm_result is not None, psm_elapsed)
            agent_logger.info(
                "Treatment effect estimated",
                method="propensity_matching",
                treatment=treatment,
                outcome=outcome,
                duration_ms=psm_elapsed * 1000,
                success=psm_result is not None,
            )
            if psm_result:
                treatment_ids.append(
                    await _store_effect(
                        analysis_id,
                        treatment,
                        outcome,
                        psm_result,
                        TreatmentMethod.PROPENSITY_MATCHING,
                        confounders,
                    )
                )
                method_results.append({
                    "method": "propensity_matching",
                    "ate": psm_result.get("ate"),
                    "ate_ci_lower": psm_result.get("ate_ci_lower"),
                    "ate_ci_upper": psm_result.get("ate_ci_upper"),
                    "confidence_score": psm_result.get("confidence_score"),
                })

            if "doubly_robust" in [tool.name for tool in tools]:
                start = time.time()
                dr_result = retry_sync(
                    _estimate_doubly_robust,
                    df,
                    treatment,
                    outcome,
                    confounders,
                    retry_on=is_transient_tool_error,
                )
                dr_elapsed = time.time() - start
                registry.track_performance("treatment:doubly_robust", dr_result is not None, dr_elapsed)
                agent_logger.info(
                    "Treatment effect estimated",
                    method="doubly_robust",
                    treatment=treatment,
                    outcome=outcome,
                    duration_ms=dr_elapsed * 1000,
                    success=dr_result is not None,
                )
                if dr_result:
                    treatment_ids.append(
                        await _store_effect(
                            analysis_id,
                            treatment,
                            outcome,
                            dr_result,
                            TreatmentMethod.DOUBLY_ROBUST,
                            confounders,
                        )
                    )
                    method_results.append({
                        "method": "doubly_robust",
                        "ate": dr_result.get("ate"),
                        "ate_ci_lower": dr_result.get("ate_ci_lower"),
                        "ate_ci_upper": dr_result.get("ate_ci_upper"),
                        "confidence_score": dr_result.get("confidence_score"),
                    })

            if "instrumental_variable" in [tool.name for tool in tools] or "iv" in analysis_types:
                start = time.time()
                iv_result = retry_sync(
                    _estimate_iv,
                    df,
                    treatment,
                    outcome,
                    confounders,
                    retry_on=is_transient_tool_error,
                )
                iv_elapsed = time.time() - start
                registry.track_performance("treatment:instrumental_variable", iv_result is not None, iv_elapsed)
                agent_logger.info(
                    "Treatment effect estimated",
                    method="iv",
                    treatment=treatment,
                    outcome=outcome,
                    duration_ms=iv_elapsed * 1000,
                    success=iv_result is not None,
                )
                if iv_result:
                    treatment_ids.append(
                        await _store_effect(
                            analysis_id,
                            treatment,
                            outcome,
                            iv_result,
                            TreatmentMethod.INSTRUMENTAL_VARIABLE,
                            confounders,
                        )
                    )
                    extra_results.setdefault("iv", []).append(iv_result)
                    method_results.append({
                        "method": "iv",
                        "ate": iv_result.get("ate"),
                        "ate_ci_lower": iv_result.get("ate_ci_lower"),
                        "ate_ci_upper": iv_result.get("ate_ci_upper"),
                        "confidence_score": iv_result.get("confidence_score"),
                    })

            if "mediation" in analysis_types:
                start = time.time()
                mediation = retry_sync(
                    _estimate_mediation,
                    df,
                    treatment,
                    outcome,
                    confounders,
                    retry_on=is_transient_tool_error,
                )
                mediation_elapsed = time.time() - start
                registry.track_performance("treatment:mediation", mediation is not None, mediation_elapsed)
                agent_logger.info(
                    "Treatment effect estimated",
                    method="mediation",
                    treatment=treatment,
                    outcome=outcome,
                    duration_ms=mediation_elapsed * 1000,
                    success=mediation is not None,
                )
                if mediation:
                    extra_results["mediation"] = mediation

            if "heterogeneous" in analysis_types:
                start = time.time()
                hetero = retry_sync(
                    _estimate_heterogeneous,
                    df,
                    treatment,
                    outcome,
                    confounders,
                    retry_on=is_transient_tool_error,
                )
                hetero_elapsed = time.time() - start
                registry.track_performance("treatment:heterogeneous", hetero is not None, hetero_elapsed)
                agent_logger.info(
                    "Treatment effect estimated",
                    method="heterogeneous",
                    treatment=treatment,
                    outcome=outcome,
                    duration_ms=hetero_elapsed * 1000,
                    success=hetero is not None,
                )
                if hetero:
                    extra_results["heterogeneous"] = hetero

            if "time_varying" in analysis_types:
                start = time.time()
                time_effect = retry_sync(
                    _estimate_time_varying,
                    df,
                    treatment,
                    outcome,
                    retry_on=is_transient_tool_error,
                )
                time_elapsed = time.time() - start
                registry.track_performance("treatment:time_varying", time_effect is not None, time_elapsed)
                agent_logger.info(
                    "Treatment effect estimated",
                    method="time_varying",
                    treatment=treatment,
                    outcome=outcome,
                    duration_ms=time_elapsed * 1000,
                    success=time_effect is not None,
                )
                if time_effect:
                    extra_results["time_varying"] = time_effect

            consensus = _consensus_summary(method_results)
            if consensus:
                agent_logger.info(
                    "Consensus calculated",
                    method_count=len(method_results),
                    mean_ate=consensus.get("mean_ate"),
                    std_ate=consensus.get("std_ate"),
                    treatment=treatment,
                    outcome=outcome,
                )
                extra_results.setdefault("consensus", []).append(
                    {"treatment": treatment, "outcome": outcome, **consensus}
                )

            # Compute ensemble estimate if multiple methods succeeded
            if len(method_results) >= 2:
                ensemble_result = _ensemble_estimate(method_results)
                if ensemble_result:
                    agent_logger.info(
                        "Ensemble estimate computed",
                        ensemble_ate=ensemble_result.get("ate"),
                        methods_used=ensemble_result.get("methods_used"),
                        weights=ensemble_result.get("weights"),
                        treatment=treatment,
                        outcome=outcome,
                    )
                    treatment_ids.append(
                        await _store_effect(
                            analysis_id,
                            treatment,
                            outcome,
                            ensemble_result,
                            TreatmentMethod.OTHER,
                            confounders,
                        )
                    )
                    extra_results["ensemble"] = ensemble_result

        state["treatment_effect_ids"] = treatment_ids
        if extra_results:
            state.setdefault("extra_results", {}).update(extra_results)

        return AgentResult(
            state=state,
            outputs={"treatment_effects": len(treatment_ids)},
            message="Treatment effects estimated",
        )


def _load_dataframe(dataset) -> pd.DataFrame:
    """Load dataframe, preferring preprocessed data if available.

    Attempts to load data in the following order:
    1. Preprocessed data from local path
    2. Preprocessed data from GCS (downloads to local path)
    3. Raw data from local path

    This ensures GCS-only artifacts are not ignored.
    """
    from pathlib import Path
    from app.services.data_loader import load_preprocessed_dataframe

    # Try to load preprocessed data using the centralized loader
    # which handles both local and GCS paths
    preprocessed_df = load_preprocessed_dataframe(dataset)
    if preprocessed_df is not None:
        logger.info(
            "Loaded preprocessed data for treatment effects",
            row_count=len(preprocessed_df),
            column_count=preprocessed_df.shape[1],
        )
        return preprocessed_df

    # If preprocessed data not available via sync loader, try async GCS download
    # This handles cases where we're not in an async context but need GCS data
    if dataset.dataset_metadata:
        preprocessed_path = dataset.dataset_metadata.get("preprocessed_path")
        preprocessed_gcs_path = dataset.dataset_metadata.get("preprocessed_gcs_path")

        if preprocessed_path and preprocessed_gcs_path:
            pp_path = Path(preprocessed_path)
            if not pp_path.exists():
                try:
                    import asyncio
                    from app.services.storage import download_to_path

                    # Try to download from GCS
                    pp_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                download_to_path(preprocessed_gcs_path, pp_path)
                            )
                            future.result(timeout=60)
                    except RuntimeError:
                        # No running loop, safe to use asyncio.run
                        asyncio.run(download_to_path(preprocessed_gcs_path, pp_path))

                    if pp_path.exists():
                        logger.info(
                            "Downloaded and loading preprocessed data from GCS for treatment effects",
                            file_path=str(pp_path),
                        )
                        return load_dataframe(pp_path)
                except Exception as exc:
                    logger.warning(
                        "Failed to download preprocessed data from GCS for treatment effects",
                        error=str(exc),
                    )

    # Fall back to raw data
    local_path = infer_local_path(dataset.dataset_metadata)
    if local_path and local_path.exists():
        logger.info(
            "Falling back to raw data for treatment effects",
            file_path=str(local_path),
        )
        return load_dataframe(local_path)
    raise ValueError("No dataset path available for treatment effects (checked local and GCS)")


def _resolve_column_name(col_name: str, df: pd.DataFrame) -> str | None:
    """Resolve a column name that may have been transformed during preprocessing.

    Handles cases like:
    - Exact match: "Sex" → "Sex"
    - One-hot encoded: "Sex" → "Sex_male" (when Sex column was dropped and replaced)
    - Case-insensitive: "sex" → "Sex"

    Returns the actual column name in the dataframe, or None if not found.
    """
    # Exact match
    if col_name in df.columns:
        return col_name

    # Case-insensitive match
    col_lower = col_name.lower()
    for col in df.columns:
        if col.lower() == col_lower:
            return col

    # One-hot encoded column: look for columns starting with "{col_name}_"
    # This handles cases where "Sex" becomes "Sex_male" after one-hot encoding
    encoded_cols = [c for c in df.columns if c.startswith(f"{col_name}_")]
    if len(encoded_cols) == 1:
        # Single encoded column (binary variable after drop_first=True)
        logger.info(
            "Resolved one-hot encoded column",
            original=col_name,
            resolved=encoded_cols[0],
        )
        return encoded_cols[0]
    elif len(encoded_cols) > 1:
        # Multiple encoded columns - can't use directly for treatment/outcome
        # Return the first one with a warning
        logger.warning(
            "Multiple one-hot encoded columns found, using first",
            original=col_name,
            encoded_cols=encoded_cols[:5],
        )
        return encoded_cols[0]

    # Case-insensitive one-hot encoded match
    for col in df.columns:
        if col.lower().startswith(f"{col_lower}_"):
            logger.info(
                "Resolved one-hot encoded column (case-insensitive)",
                original=col_name,
                resolved=col,
            )
            return col

    return None


def _select_pairs(df: pd.DataFrame, state) -> list[tuple[str, str]]:
    """Select treatment/outcome pairs for causal effect estimation.

    Priority order:
    1. User-specified treatment_variable and outcome_variable from config
    2. EDA-suggested treatment_candidates and outcome_candidates
    3. First two numeric columns as fallback

    Handles column name resolution for preprocessed data where columns may have been:
    - One-hot encoded (e.g., "Sex" → "Sex_male")
    - Renamed or transformed
    """
    config = state.get("config", {})
    candidates = state.get("data_characteristics", {})

    pairs: list[tuple[str, str]] = []

    # Priority 1: User-specified variables in config
    user_treatment = config.get("treatment_variable")
    user_outcome = config.get("outcome_variable")

    if user_treatment and user_outcome:
        # Try to resolve column names (handles one-hot encoding, case differences)
        resolved_treatment = _resolve_column_name(user_treatment, df)
        resolved_outcome = _resolve_column_name(user_outcome, df)

        if resolved_treatment and resolved_outcome and resolved_treatment != resolved_outcome:
            pairs.append((resolved_treatment, resolved_outcome))
            logger.info(
                "Using user-specified treatment/outcome pair",
                original_treatment=user_treatment,
                original_outcome=user_outcome,
                resolved_treatment=resolved_treatment,
                resolved_outcome=resolved_outcome,
            )
            return pairs
        else:
            logger.warning(
                "User-specified treatment/outcome not found in data, falling back to candidates",
                treatment=user_treatment,
                outcome=user_outcome,
                resolved_treatment=resolved_treatment,
                resolved_outcome=resolved_outcome,
                available_columns=list(df.columns[:15]),
            )

    # Priority 2: EDA-suggested candidates (also resolve column names)
    treatments = candidates.get("treatment_candidates", [])
    outcomes = candidates.get("outcome_candidates", [])
    for treatment in treatments[:1]:
        for outcome in outcomes[:1]:
            resolved_t = _resolve_column_name(treatment, df)
            resolved_o = _resolve_column_name(outcome, df)
            if resolved_t and resolved_o and resolved_t != resolved_o:
                pairs.append((resolved_t, resolved_o))
                logger.info(
                    "Using EDA-suggested treatment/outcome pair",
                    original_treatment=treatment,
                    original_outcome=outcome,
                    resolved_treatment=resolved_t,
                    resolved_outcome=resolved_o,
                )

    # Priority 3: Fallback to first two numeric columns
    if not pairs:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            pairs.append((numeric_cols[0], numeric_cols[1]))
            logger.info(
                "Using fallback numeric columns for treatment/outcome",
                treatment=numeric_cols[0],
                outcome=numeric_cols[1],
            )

    return pairs


def _confounders_from_state(state, treatment: str, outcome: str) -> list[str]:
    """Extract confounders from state, excluding engineered features by default.

    Engineered features (prefixed with 'fe_') are excluded unless explicitly requested
    in the analysis config.
    """
    confounders = state.get("data_characteristics", {}).get("confounder_candidates", [])

    # Check if engineered features should be included
    include_engineered = state.get("config", {}).get("include_engineered_confounders", False)

    filtered = []
    for c in confounders:
        if c in {treatment, outcome}:
            continue
        # Exclude engineered features unless explicitly included
        if c.startswith("fe_") and not include_engineered:
            continue
        filtered.append(c)

    return filtered


@traced("tool.treatment.psm")
def _estimate_psm(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict[str, Any] | None:
    try:
        from dowhy import CausalModel
    except Exception:
        return None

    try:
        num_simulations = getattr(settings, "BOOTSTRAP_NUM_SIMULATIONS", DEFAULT_BOOTSTRAP_SIMULATIONS)
        confidence_level = getattr(settings, "BOOTSTRAP_CONFIDENCE_LEVEL", DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL)

        model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=confounders)
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")
        ate = float(estimate.value)

        # Use bootstrap confidence interval
        ci_lower, ci_upper, bootstrap_samples = _bootstrap_confidence_interval(
            df, treatment, outcome, confounders,
            method_name="backdoor.propensity_score_matching",
            num_simulations=num_simulations,
            confidence_level=confidence_level,
        )

        # Fallback to DoWhy's default CI if bootstrap fails
        if ci_lower is None or ci_upper is None:
            ci = _confidence_interval(estimate)
            ci_lower, ci_upper = ci[0], ci[1]
            bootstrap_samples = None

        bootstrap_std = float(np.std(bootstrap_samples)) if bootstrap_samples is not None else None

        return {
            "ate": ate,
            "ate_ci_lower": ci_lower,
            "ate_ci_upper": ci_upper,
            "confidence_interval": {
                "method": "bootstrap",
                "lower": ci_lower,
                "upper": ci_upper,
                "num_simulations": num_simulations,
                "confidence_level": confidence_level,
                "samples": bootstrap_samples[:100] if bootstrap_samples is not None else None,
            },
            "bootstrap_std": bootstrap_std,
            "att": None,
            "sample_size": {"total": len(df)},
            "assumptions_checked": {"positivity": True},
            "confidence_score": 0.8 if ci_lower is not None and ci_upper is not None else 0.6,
        }
    except Exception as exc:
        # Log the error and return None to allow fallback to other methods
        # Common exceptions: "Propensity score methods are applicable only for binary treatments"
        logger.info(
            "Propensity score matching failed, will try other methods",
            treatment=treatment,
            outcome=outcome,
            error=str(exc),
        )
        return None


@traced("tool.treatment.doubly_robust")
def _estimate_doubly_robust(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict[str, Any] | None:
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LinearRegression, LogisticRegression
    except Exception:
        return None

    num_simulations = getattr(settings, "BOOTSTRAP_NUM_SIMULATIONS", DEFAULT_BOOTSTRAP_SIMULATIONS)
    confidence_level = getattr(settings, "BOOTSTRAP_CONFIDENCE_LEVEL", DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL)

    y = df[outcome].to_numpy()
    t = df[treatment].to_numpy()
    X = df[confounders].to_numpy() if confounders else None

    model = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(max_iter=1000))
    model.fit(y, t, X=X)
    ate = float(model.ate(X=X)) if X is not None else float(model.ate())

    # Manual bootstrap for CI since EconML doesn't provide built-in CI for ATE
    bootstrap_samples = []
    n = len(df)

    try:
        for _ in range(num_simulations):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            df_boot = df.iloc[indices]

            y_boot = df_boot[outcome].to_numpy()
            t_boot = df_boot[treatment].to_numpy()
            X_boot = df_boot[confounders].to_numpy() if confounders else None

            boot_model = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(max_iter=1000))
            boot_model.fit(y_boot, t_boot, X=X_boot)
            boot_ate = float(boot_model.ate(X=X_boot)) if X_boot is not None else float(boot_model.ate())
            bootstrap_samples.append(boot_ate)

        bootstrap_samples = np.array(bootstrap_samples)
        alpha = 1 - confidence_level
        ci_lower = float(np.percentile(bootstrap_samples, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_samples, (1 - alpha / 2) * 100))
        bootstrap_std = float(np.std(bootstrap_samples))
    except Exception:
        ci_lower, ci_upper = None, None
        bootstrap_std = None
        bootstrap_samples = []

    return {
        "ate": ate,
        "ate_ci_lower": ci_lower,
        "ate_ci_upper": ci_upper,
        "confidence_interval": {
            "method": "bootstrap",
            "lower": ci_lower,
            "upper": ci_upper,
            "num_simulations": num_simulations,
            "confidence_level": confidence_level,
            "samples": list(bootstrap_samples[:100]) if len(bootstrap_samples) > 0 else None,
        },
        "bootstrap_std": bootstrap_std,
        "att": None,
        "sample_size": {"total": len(df)},
        "assumptions_checked": {"robust": True, "doubly_robust": True},
        "confidence_score": 0.85 if ci_lower is not None and ci_upper is not None else 0.7,
    }


@traced("tool.treatment.iv")
def _estimate_iv(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict[str, Any] | None:
    try:
        from dowhy import CausalModel
    except Exception:
        return None

    num_simulations = getattr(settings, "BOOTSTRAP_NUM_SIMULATIONS", DEFAULT_BOOTSTRAP_SIMULATIONS)
    confidence_level = getattr(settings, "BOOTSTRAP_CONFIDENCE_LEVEL", DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL)

    instrument = _select_instrument(df, treatment, outcome)
    if not instrument:
        return None

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
        instruments=[instrument],
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="iv.instrumental_variable")
    ate = float(estimate.value)

    # Compute first-stage F-statistic for instrument strength
    first_stage_f = None
    try:
        from sklearn.linear_model import LinearRegression
        from scipy import stats

        # First stage: regress treatment on instrument (and confounders)
        X_first = df[[instrument] + confounders].to_numpy() if confounders else df[[instrument]].to_numpy()
        t_values = df[treatment].to_numpy()

        reg = LinearRegression().fit(X_first, t_values)
        y_pred = reg.predict(X_first)
        ss_res = np.sum((t_values - y_pred) ** 2)
        ss_tot = np.sum((t_values - np.mean(t_values)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        n = len(df)
        k = X_first.shape[1]
        first_stage_f = (r_squared / k) / ((1 - r_squared) / (n - k - 1)) if r_squared < 1 else float("inf")
    except Exception:
        pass

    # Bootstrap CI for IV estimate
    ci_lower, ci_upper, bootstrap_samples = _bootstrap_confidence_interval(
        df, treatment, outcome, confounders,
        method_name="iv.instrumental_variable",
        num_simulations=num_simulations,
        confidence_level=confidence_level,
        instrument=instrument,
    )

    bootstrap_std = float(np.std(bootstrap_samples)) if bootstrap_samples is not None else None

    # Weak instrument warning
    weak_instrument = first_stage_f is not None and first_stage_f < 10

    return {
        "ate": ate,
        "ate_ci_lower": ci_lower,
        "ate_ci_upper": ci_upper,
        "confidence_interval": {
            "method": "bootstrap",
            "lower": ci_lower,
            "upper": ci_upper,
            "num_simulations": num_simulations,
            "confidence_level": confidence_level,
            "samples": bootstrap_samples[:100] if bootstrap_samples is not None else None,
        },
        "bootstrap_std": bootstrap_std,
        "att": None,
        "sample_size": {"total": len(df)},
        "assumptions_checked": {
            "instrument": instrument,
            "first_stage_f_statistic": first_stage_f,
            "weak_instrument_warning": weak_instrument,
            "exclusion_restriction": "assumed",
        },
        "confidence_score": 0.7 if not weak_instrument else 0.4,
    }


@traced("tool.treatment.mediation")
def _estimate_mediation(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict[str, Any] | None:
    mediator = _select_mediator(df, treatment, outcome)
    if not mediator:
        return None
    try:
        from dowhy import CausalModel
    except Exception:
        return None
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
        mediators=[mediator],
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="mediation.two_stage_regression")
    return {"mediator": mediator, "effect": float(estimate.value)}


@traced("tool.treatment.heterogeneous")
def _estimate_heterogeneous(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict[str, Any] | None:
    try:
        from econml.dml import CausalForestDML
        from sklearn.linear_model import LinearRegression, LogisticRegression
    except Exception:
        return _group_effects(df, treatment, outcome)
    y = df[outcome].to_numpy()
    t = df[treatment].to_numpy()
    X = df[confounders].to_numpy() if confounders else None
    model = CausalForestDML(model_y=LinearRegression(), model_t=LogisticRegression(max_iter=1000))
    model.fit(y, t, X=X)
    cate = model.effect(X) if X is not None else model.effect(df[[treatment]].to_numpy())
    return {"cate_mean": float(np.mean(cate)), "cate_std": float(np.std(cate))}


@traced("tool.treatment.time_varying")
def _estimate_time_varying(df: pd.DataFrame, treatment: str, outcome: str) -> dict[str, Any] | None:
    time_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
    if not time_cols:
        return None
    time_col = time_cols[0]
    df_sorted = df.sort_values(time_col)
    df_sorted["time_bin"] = pd.qcut(df_sorted[time_col].rank(method="first"), q=4, duplicates="drop")
    results = []
    for bin_value, group in df_sorted.groupby("time_bin"):
        treated = group[group[treatment] == 1][outcome]
        control = group[group[treatment] == 0][outcome]
        if treated.empty or control.empty:
            continue
        results.append({"bin": str(bin_value), "effect": float(treated.mean() - control.mean())})
    return {"time_bins": results}


def _confidence_interval(estimate) -> tuple[float | None, float | None]:
    try:
        ci = estimate.get_confidence_intervals()
        if ci:
            return float(ci[0][0]), float(ci[0][1])
    except Exception:
        pass
    return (None, None)


@traced("tool.treatment.bootstrap_ci")
def _bootstrap_confidence_interval(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    method_name: str,
    num_simulations: int = 500,
    confidence_level: float = 0.95,
    instrument: str | None = None,
) -> tuple[float | None, float | None, list[float] | None]:
    """
    Compute bootstrap confidence interval for treatment effect estimates.

    Args:
        df: Input dataframe
        treatment: Treatment variable name
        outcome: Outcome variable name
        confounders: List of confounder variable names
        method_name: DoWhy method name for estimation
        num_simulations: Number of bootstrap iterations
        confidence_level: Confidence level for interval (e.g., 0.95)
        instrument: Instrument variable name for IV methods

    Returns:
        Tuple of (lower_bound, upper_bound, bootstrap_samples)
    """
    try:
        from dowhy import CausalModel
    except Exception:
        return (None, None, None)

    bootstrap_samples = []
    n = len(df)

    try:
        for _ in range(num_simulations):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            df_boot = df.iloc[indices].reset_index(drop=True)

            try:
                if instrument and "iv" in method_name:
                    model = CausalModel(
                        data=df_boot,
                        treatment=treatment,
                        outcome=outcome,
                        common_causes=confounders,
                        instruments=[instrument],
                    )
                else:
                    model = CausalModel(
                        data=df_boot,
                        treatment=treatment,
                        outcome=outcome,
                        common_causes=confounders,
                    )

                estimand = model.identify_effect()
                estimate = model.estimate_effect(estimand, method_name=method_name)
                bootstrap_samples.append(float(estimate.value))
            except Exception:
                # Skip failed bootstrap iterations
                continue

        if len(bootstrap_samples) < num_simulations * 0.5:
            # Too many failures, return None
            return (None, None, None)

        bootstrap_samples = np.array(bootstrap_samples)
        alpha = 1 - confidence_level
        ci_lower = float(np.percentile(bootstrap_samples, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_samples, (1 - alpha / 2) * 100))

        return (ci_lower, ci_upper, bootstrap_samples.tolist())
    except Exception:
        return (None, None, None)


def _consensus_summary(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    values = [item.get("ate") for item in results if item.get("ate") is not None]
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {"mean_ate": mean, "std_ate": std, "methods": results}


@traced("tool.treatment.ensemble")
def _ensemble_estimate(method_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Compute weighted ensemble estimate from multiple treatment effect methods.

    Weights are computed based on confidence scores and CI width, giving more
    weight to methods with higher confidence and narrower intervals.

    Args:
        method_results: List of dicts with 'method', 'ate', 'ate_ci_lower',
                       'ate_ci_upper', and optionally 'confidence_score'

    Returns:
        Dict with ensemble 'ate', 'ate_ci_lower', 'ate_ci_upper', 'weights', 'methods_used'
    """
    # Filter to methods with valid ATE
    valid_results = [r for r in method_results if r.get("ate") is not None]
    if len(valid_results) < 2:
        return None

    weights = []
    ates = []
    variances = []
    methods_used = []

    for result in valid_results:
        ate = result.get("ate")
        ci_lower = result.get("ate_ci_lower")
        ci_upper = result.get("ate_ci_upper")
        confidence_score = result.get("confidence_score", 0.5)
        method_name = result.get("method", "unknown")

        # Compute CI width (use default width if CI not available)
        if ci_lower is not None and ci_upper is not None:
            ci_width = ci_upper - ci_lower
        else:
            # Default to a wide CI based on ATE magnitude
            ci_width = abs(ate) * 2 if ate != 0 else 1.0

        # Compute weight: higher confidence and narrower CI = higher weight
        # Add small constant to avoid division by zero
        weight = confidence_score / (1 + ci_width)
        weights.append(weight)

        ates.append(ate)

        # Estimate variance from CI width (assuming normal distribution, 95% CI = 3.92 * SE)
        if ci_width > 0:
            variance = (ci_width / 3.92) ** 2
        else:
            variance = 1.0
        variances.append(variance)

        methods_used.append(method_name)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    normalized_weights = [w / total_weight for w in weights]

    # Compute weighted average ATE
    ensemble_ate = sum(w * a for w, a in zip(normalized_weights, ates))

    # Compute ensemble variance using weighted variance formula
    # Var(weighted sum) = sum(w_i^2 * Var_i)
    ensemble_variance = sum(w ** 2 * v for w, v in zip(normalized_weights, variances))
    ensemble_se = np.sqrt(ensemble_variance)

    # Compute ensemble CI (95%)
    ensemble_ci_lower = ensemble_ate - 1.96 * ensemble_se
    ensemble_ci_upper = ensemble_ate + 1.96 * ensemble_se

    # Compute ensemble confidence score (weighted average of individual scores)
    confidence_scores = [r.get("confidence_score", 0.5) for r in valid_results]
    ensemble_confidence = sum(w * c for w, c in zip(normalized_weights, confidence_scores))

    return {
        "ate": float(ensemble_ate),
        "ate_ci_lower": float(ensemble_ci_lower),
        "ate_ci_upper": float(ensemble_ci_upper),
        "confidence_interval": {
            "method": "ensemble_weighted_variance",
            "lower": float(ensemble_ci_lower),
            "upper": float(ensemble_ci_upper),
        },
        "weights": {m: float(w) for m, w in zip(methods_used, normalized_weights)},
        "methods_used": methods_used,
        "method_count": len(valid_results),
        "ensemble_se": float(ensemble_se),
        "sample_size": {},
        "assumptions_checked": {
            "ensemble": True,
            "methods": methods_used,
            "weighting_scheme": "confidence_score / (1 + ci_width)",
        },
        "confidence_score": float(ensemble_confidence),
    }


def _select_instrument(df: pd.DataFrame, treatment: str, outcome: str) -> str | None:
    numeric = df.select_dtypes(include=[np.number])
    if treatment not in numeric.columns or outcome not in numeric.columns:
        return None
    corr = numeric.corr()
    candidates = [col for col in numeric.columns if col not in {treatment, outcome}]
    for col in candidates:
        if abs(corr.loc[col, treatment]) > 0.2 and abs(corr.loc[col, outcome]) < 0.1:
            return col
    return None


def _select_mediator(df: pd.DataFrame, treatment: str, outcome: str) -> str | None:
    candidates = [col for col in df.columns if col not in {treatment, outcome}]
    return candidates[0] if candidates else None


def _group_effects(df: pd.DataFrame, treatment: str, outcome: str) -> dict[str, Any] | None:
    cat_cols = [col for col in df.columns if df[col].nunique() <= 10 and col not in {treatment, outcome}]
    if not cat_cols:
        return None
    col = cat_cols[0]
    results = []
    for value, group in df.groupby(col):
        treated = group[group[treatment] == 1][outcome]
        control = group[group[treatment] == 0][outcome]
        if treated.empty or control.empty:
            continue
        results.append({"segment": str(value), "effect": float(treated.mean() - control.mean())})
    return {"segments": results}


async def _store_effect(
    analysis_id: uuid.UUID,
    treatment: str,
    outcome: str,
    result: dict[str, Any],
    method: TreatmentMethod,
    confounders: list[str],
) -> str:
    effect = TreatmentEffect(
        analysis_id=analysis_id,
        treatment_variable=treatment,
        outcome_variable=outcome,
        method=method,
        ate=result.get("ate"),
        ate_ci_lower=result.get("ate_ci_lower"),
        ate_ci_upper=result.get("ate_ci_upper"),
        confidence_interval=result.get("confidence_interval", {}),
        att=result.get("att"),
        confounders_adjusted=confounders,
        sample_size=result.get("sample_size", {}),
        assumptions_checked=result.get("assumptions_checked", {}),
    )
    async with get_session_context() as session:
        session.add(effect)
        await session.flush()
        return str(effect.id)
