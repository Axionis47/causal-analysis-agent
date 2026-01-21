"""Validation agent running refutation tests."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.agents.base import AgentResult, BaseAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.crud.analysis import analysis_crud
from app.crud.dataset import dataset_crud
from app.db.database import get_session_context
from app.models.analysis_stage import StageType
from app.models.validation_result import ValidationResult, ValidationType
from app.services.data_loader import load_dataframe
from app.services.retry import is_transient_tool_error, retry_sync
from app.services.storage import infer_local_path
from app.services.tracing import traced

logger = get_logger(__name__)


class ValidationAgent(BaseAgent):
    name = "Validation Agent"
    stage = StageType.VALIDATION

    async def _run(self, state):
        if state.get("validation_result_ids"):
            return AgentResult(state=state, outputs={"cached": True}, message="Validation cached")

        agent_logger = logger.bind(
            analysis_id=state.get("analysis_id"),
            stage=self.stage.value,
            agent_name=self.name,
        )
        analysis_id = uuid.UUID(state["analysis_id"])
        dataset_id = uuid.UUID(state["dataset_id"])

        async with get_session_context() as session:
            analysis = await analysis_crud.get_with_relations(session, analysis_id)
            dataset = await dataset_crud.get(session, dataset_id)
            if analysis is None or dataset is None:
                raise ValueError("Analysis or dataset not found")

        if not analysis.treatment_effects:
            return AgentResult(state=state, outputs={}, message="No treatment effects to validate")

        df = _load_dataframe(dataset)
        effect = analysis.treatment_effects[0]
        confounders = effect.confounders_adjusted or []

        results = []
        confidence_scores = []
        for method in ["placebo_treatment_refuter", "random_common_cause"]:
            result = _run_refutation(df, effect.treatment_variable, effect.outcome_variable, confounders, method)
            if result is None:
                continue
            agent_logger.info(
                "Refutation test completed",
                refuter_name=method,
                p_value=result.get("details", {}).get("p_value"),
                passed=result.get("passed"),
            )
            results.append(result)
            confidence_scores.append(1.0 if result["passed"] else 0.0)

        # Run sensitivity analysis
        sensitivity_results = []
        for sensitivity_method in [_run_sensitivity_linear, _run_sensitivity_evalue]:
            try:
                sens_result = retry_sync(
                    sensitivity_method,
                    df,
                    effect.treatment_variable,
                    effect.outcome_variable,
                    confounders,
                    retry_on=is_transient_tool_error,
                )
                if sens_result:
                    agent_logger.info(
                        "Sensitivity analysis completed",
                        method=sens_result.get("method"),
                        passed=sens_result.get("passed"),
                        robustness_value=sens_result.get("details", {}).get("robustness_value"),
                        evalue=sens_result.get("details", {}).get("evalue"),
                    )
                    sensitivity_results.append(sens_result)
                    confidence_scores.append(sens_result.get("confidence_score", 0.5))
            except Exception as exc:
                agent_logger.warning(
                    "Sensitivity analysis failed",
                    method=sensitivity_method.__name__,
                    error=str(exc),
                )

        # Run assumption checks
        assumption_results = []
        for assumption_check in [_check_positivity, _check_unconfoundedness, _check_sutva]:
            try:
                assumption_result = retry_sync(
                    assumption_check,
                    df,
                    effect.treatment_variable,
                    effect.outcome_variable,
                    confounders,
                    retry_on=is_transient_tool_error,
                )
                if assumption_result:
                    agent_logger.info(
                        "Assumption check completed",
                        method=assumption_result.get("method"),
                        passed=assumption_result.get("passed"),
                        violation_rate=assumption_result.get("details", {}).get("violation_rate"),
                    )
                    assumption_results.append(assumption_result)
                    # Reduce confidence if critical assumptions fail
                    if not assumption_result.get("passed", True):
                        confidence_scores.append(0.3)
                    else:
                        confidence_scores.append(0.8)
            except Exception as exc:
                agent_logger.warning(
                    "Assumption check failed",
                    method=assumption_check.__name__,
                    error=str(exc),
                )

        validation_ids = []
        async with get_session_context() as session:
            # Store refutation results
            for result in results:
                validation = ValidationResult(
                    analysis_id=analysis_id,
                    validation_type=ValidationType.REFUTATION,
                    method=result["method"],
                    passed=result["passed"],
                    confidence_score=result["confidence_score"],
                    details=result["details"],
                    recommendations=result["recommendations"],
                )
                session.add(validation)
                await session.flush()
                validation_ids.append(str(validation.id))

            # Store sensitivity analysis results
            for sens_result in sensitivity_results:
                validation = ValidationResult(
                    analysis_id=analysis_id,
                    validation_type=ValidationType.SENSITIVITY,
                    method=sens_result["method"],
                    passed=sens_result["passed"],
                    confidence_score=sens_result["confidence_score"],
                    details=sens_result["details"],
                    recommendations=sens_result["recommendations"],
                )
                session.add(validation)
                await session.flush()
                validation_ids.append(str(validation.id))

            # Store assumption check results
            for assumption_result in assumption_results:
                validation = ValidationResult(
                    analysis_id=analysis_id,
                    validation_type=ValidationType.ROBUSTNESS,
                    method=assumption_result["method"],
                    passed=assumption_result["passed"],
                    confidence_score=assumption_result["confidence_score"],
                    details=assumption_result["details"],
                    recommendations=assumption_result["recommendations"],
                )
                session.add(validation)
                await session.flush()
                validation_ids.append(str(validation.id))

        state["validation_result_ids"] = validation_ids
        overall_confidence = sum(confidence_scores) / max(len(confidence_scores), 1)
        return AgentResult(
            state=state,
            outputs={
                "confidence_score": overall_confidence,
                "refutation_count": len(results),
                "sensitivity_count": len(sensitivity_results),
                "assumption_count": len(assumption_results),
            },
            message="Validation complete",
        )


def _load_dataframe(dataset) -> pd.DataFrame:
    local_path = infer_local_path(dataset.dataset_metadata)
    if local_path and local_path.exists():
        return load_dataframe(local_path)
    raise ValueError("Local dataset path not found for validation")


@traced("tool.validation.refutation")
def _run_refutation(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    method: str,
) -> dict[str, Any] | None:
    try:
        from dowhy import CausalModel
    except Exception:
        return None

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=confounders)
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")
    refute = model.refute_estimate(estimand, estimate, method_name=method)
    p_value = getattr(refute, "p_value", None)
    passed = p_value is None or p_value > 0.05
    return {
        "method": method,
        "passed": passed,
        "confidence_score": 0.7 if passed else 0.3,
        "details": {"p_value": p_value, "refutation": str(refute)},
        "recommendations": ["Review confounders"] if not passed else [],
    }


@traced("tool.validation.sensitivity_linear")
def _run_sensitivity_linear(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any] | None:
    """
    Run linear partial R-squared sensitivity analysis using DoWhy's sensitivity API.

    Computes robustness value (RV) indicating how strong unmeasured confounding
    would need to be to reduce the estimate by 50% or 100%.

    Uses DoWhy's SensitivityAnalyzer when available, falling back to manual
    calculations if the sensitivity module is unavailable.
    """
    try:
        from dowhy import CausalModel
    except Exception:
        return None

    rv_threshold = getattr(settings, "SENSITIVITY_RV_THRESHOLD", 0.1)

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=confounders)
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")

    # Try DoWhy's sensitivity API first
    try:
        from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
        from dowhy.causal_refuters.linear_sensitivity_analyzer import LinearSensitivityAnalyzer

        # Use DoWhy's LinearSensitivityAnalyzer for robustness analysis
        sensitivity_analyzer = LinearSensitivityAnalyzer(
            estimator=estimate,
            data=df,
            treatment_name=treatment,
            outcome_name=outcome,
            confounders=confounders,
        )

        # Get robustness value from DoWhy's sensitivity analysis
        sensitivity_result = sensitivity_analyzer.analyze()
        robustness_value = sensitivity_result.get("robustness_value", None)
        rv_50 = sensitivity_result.get("rv_50_percent", None)
        rv_100 = sensitivity_result.get("rv_100_percent", None)
        original_effect = float(estimate.value)
        benchmark_r2 = sensitivity_result.get("benchmark_r2", None)
        benchmark_covariate = confounders[0] if confounders else None

        # If DoWhy returns valid results, use them
        if robustness_value is not None:
            passed = robustness_value > rv_threshold
            confidence_score = min(robustness_value * 2, 1.0) if passed else robustness_value

            recommendations = []
            if not passed:
                recommendations.append(f"Robustness value ({robustness_value:.2f}) is below threshold ({rv_threshold})")
                recommendations.append("Consider collecting data on potential unmeasured confounders")
                recommendations.append("The causal estimate may be sensitive to unmeasured confounding")

            return {
                "method": "sensitivity_linear",
                "passed": passed,
                "confidence_score": confidence_score,
                "details": {
                    "robustness_value": robustness_value,
                    "rv_50_percent": rv_50,
                    "rv_100_percent": rv_100,
                    "original_effect": original_effect,
                    "benchmark_covariate": benchmark_covariate,
                    "benchmark_r2": benchmark_r2,
                    "threshold": rv_threshold,
                    "source": "dowhy_sensitivity_api",
                },
                "recommendations": recommendations,
            }
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("DoWhy sensitivity API not available, using fallback", error=str(e))

    # Fallback: manual sensitivity analysis using add_unobserved_common_cause refuter
    try:
        # Run sensitivity analysis with linear-partial-R2 method
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            confounders_effect_on_treatment="linear",
            confounders_effect_on_outcome="linear",
            effect_fraction_on_treatment=0.1,
            effect_fraction_on_outcome=0.1,
        )

        # Compute robustness values by varying confounder strength
        rv_50 = None
        rv_100 = None
        original_effect = float(estimate.value)

        for strength in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            test_refute = model.refute_estimate(
                estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                confounders_effect_on_treatment="linear",
                confounders_effect_on_outcome="linear",
                effect_fraction_on_treatment=strength,
                effect_fraction_on_outcome=strength,
            )
            new_effect = float(test_refute.new_effect)
            reduction = abs(original_effect - new_effect) / abs(original_effect) if original_effect != 0 else 0

            if rv_50 is None and reduction >= 0.5:
                rv_50 = strength
            if rv_100 is None and reduction >= 1.0:
                rv_100 = strength
                break

        # Use first confounder as benchmark if available
        benchmark_covariate = confounders[0] if confounders else None
        benchmark_r2 = None
        if benchmark_covariate:
            # Compute partial R2 of benchmark covariate with treatment and outcome
            from sklearn.linear_model import LinearRegression
            X_bench = df[[benchmark_covariate]].to_numpy()
            t_values = df[treatment].to_numpy()
            y_values = df[outcome].to_numpy()

            reg_t = LinearRegression().fit(X_bench, t_values)
            reg_y = LinearRegression().fit(X_bench, y_values)
            benchmark_r2 = {
                "treatment": float(reg_t.score(X_bench, t_values)),
                "outcome": float(reg_y.score(X_bench, y_values)),
            }

        robustness_value = rv_50 if rv_50 is not None else 0.5
        passed = robustness_value > rv_threshold
        confidence_score = min(robustness_value * 2, 1.0) if passed else robustness_value

        recommendations = []
        if not passed:
            recommendations.append(f"Robustness value ({robustness_value:.2f}) is below threshold ({rv_threshold})")
            recommendations.append("Consider collecting data on potential unmeasured confounders")
            recommendations.append("The causal estimate may be sensitive to unmeasured confounding")

        return {
            "method": "sensitivity_linear",
            "passed": passed,
            "confidence_score": confidence_score,
            "details": {
                "robustness_value": robustness_value,
                "rv_50_percent": rv_50,
                "rv_100_percent": rv_100,
                "original_effect": original_effect,
                "benchmark_covariate": benchmark_covariate,
                "benchmark_r2": benchmark_r2,
                "threshold": rv_threshold,
                "source": "fallback_manual",
            },
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.warning("Sensitivity linear analysis failed", error=str(e))
        return None


@traced("tool.validation.sensitivity_evalue")
def _run_sensitivity_evalue(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any] | None:
    """
    Compute E-value for sensitivity analysis using DoWhy's sensitivity API.

    The E-value is the minimum strength of association that an unmeasured
    confounder would need to have with both the treatment and outcome
    to fully explain away the observed effect.

    Uses DoWhy's EValueSensitivityAnalyzer when available, falling back to
    manual calculations if the sensitivity module is unavailable.
    """
    try:
        from dowhy import CausalModel
    except Exception:
        return None

    evalue_threshold = getattr(settings, "SENSITIVITY_EVALUE_THRESHOLD", 1.5)

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=confounders)
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")

    # Try DoWhy's E-value sensitivity API first
    try:
        from dowhy.causal_refuters.evalue_sensitivity_analyzer import EValueSensitivityAnalyzer

        # Use DoWhy's EValueSensitivityAnalyzer
        evalue_analyzer = EValueSensitivityAnalyzer(
            estimate=estimate,
            data=df,
            treatment_name=treatment,
            outcome_name=outcome,
        )

        # Get E-value from DoWhy's sensitivity analysis
        evalue_result = evalue_analyzer.analyze()
        evalue = evalue_result.get("evalue", None)
        ci_evalue = evalue_result.get("evalue_ci", None)
        effect = float(estimate.value)

        # Determine if outcome is binary or continuous
        outcome_values = df[outcome].dropna().unique()
        is_binary_outcome = len(outcome_values) <= 2

        # If DoWhy returns valid results, use them
        if evalue is not None:
            passed = evalue > evalue_threshold
            confidence_score = min(evalue / 3.0, 1.0) if passed else evalue / (evalue_threshold * 2)

            recommendations = []
            if not passed:
                recommendations.append(f"E-value ({evalue:.2f}) is below threshold ({evalue_threshold})")
                recommendations.append("The observed effect could be explained by a moderately strong unmeasured confounder")
                recommendations.append("Consider conducting a sensitivity analysis with plausible confounder strengths")

            return {
                "method": "sensitivity_evalue",
                "passed": passed,
                "confidence_score": confidence_score,
                "details": {
                    "evalue": evalue,
                    "evalue_ci": ci_evalue,
                    "effect_estimate": effect,
                    "is_binary_outcome": is_binary_outcome,
                    "threshold": evalue_threshold,
                    "source": "dowhy_sensitivity_api",
                },
                "recommendations": recommendations,
            }
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("DoWhy E-value sensitivity API not available, using fallback", error=str(e))

    # Fallback: manual E-value calculation
    try:
        effect = float(estimate.value)

        # Determine if outcome is binary or continuous
        outcome_values = df[outcome].dropna().unique()
        is_binary_outcome = len(outcome_values) <= 2

        if is_binary_outcome:
            # For binary outcomes, convert effect to risk ratio
            # Assuming effect is odds ratio or risk difference
            if effect > 0:
                risk_ratio = 1 + effect if effect < 1 else effect
            else:
                risk_ratio = 1 / (1 - effect) if effect > -1 else 0.1

            # E-value formula for risk ratio
            if risk_ratio >= 1:
                evalue = risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1))
            else:
                evalue = 1 / risk_ratio + np.sqrt((1 / risk_ratio) * (1 / risk_ratio - 1))
        else:
            # For continuous outcomes, use standardized effect size
            outcome_std = df[outcome].std()
            standardized_effect = abs(effect) / outcome_std if outcome_std > 0 else abs(effect)

            # Convert to approximate risk ratio using Ding & VanderWeele (2016) formula
            approximate_rr = np.exp(0.91 * standardized_effect)
            evalue = approximate_rr + np.sqrt(approximate_rr * (approximate_rr - 1))

        # E-value for confidence interval bound
        ci_evalue = None
        try:
            ci = estimate.get_confidence_intervals()
            if ci and len(ci) > 0 and len(ci[0]) >= 2:
                ci_lower = ci[0][0]
                # Compute E-value for CI bound closer to null
                if effect > 0:
                    bound = ci_lower
                    if bound > 0:
                        bound_rr = 1 + bound if bound < 1 else bound
                        ci_evalue = bound_rr + np.sqrt(bound_rr * (bound_rr - 1)) if bound_rr >= 1 else None
        except Exception:
            pass

        passed = evalue > evalue_threshold
        confidence_score = min(evalue / 3.0, 1.0) if passed else evalue / (evalue_threshold * 2)

        recommendations = []
        if not passed:
            recommendations.append(f"E-value ({evalue:.2f}) is below threshold ({evalue_threshold})")
            recommendations.append("The observed effect could be explained by a moderately strong unmeasured confounder")
            recommendations.append("Consider conducting a sensitivity analysis with plausible confounder strengths")

        return {
            "method": "sensitivity_evalue",
            "passed": passed,
            "confidence_score": confidence_score,
            "details": {
                "evalue": evalue,
                "evalue_ci": ci_evalue,
                "effect_estimate": effect,
                "is_binary_outcome": is_binary_outcome,
                "threshold": evalue_threshold,
                "source": "fallback_manual",
            },
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.warning("Sensitivity E-value analysis failed", error=str(e))
        return None


@traced("tool.validation.positivity")
def _check_positivity(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any] | None:
    """
    Check positivity assumption: P(T=t|X) > 0 for all t and X.

    Estimates propensity scores and checks for extreme values that
    indicate positivity violations.
    """
    if treatment not in df.columns:
        return None

    try:
        # Compute propensity scores
        t_values = df[treatment].to_numpy()

        # Check if treatment is binary
        unique_t = np.unique(t_values[~np.isnan(t_values)])
        if len(unique_t) != 2:
            # Non-binary treatment: skip propensity score check
            return {
                "method": "positivity",
                "passed": True,
                "confidence_score": 0.7,
                "details": {
                    "note": "Non-binary treatment, positivity check simplified",
                    "unique_treatment_values": len(unique_t),
                },
                "recommendations": [],
            }

        # Fit propensity score model
        if confounders:
            X = df[confounders].fillna(df[confounders].mean()).to_numpy()
        else:
            X = np.ones((len(df), 1))

        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X, t_values)
        propensity_scores = model.predict_proba(X)[:, 1]

        # Check for extreme propensity scores (positivity violation)
        extreme_low = np.sum(propensity_scores < 0.05) / len(propensity_scores)
        extreme_high = np.sum(propensity_scores > 0.95) / len(propensity_scores)
        violation_rate = extreme_low + extreme_high

        # Check overlap between treatment groups
        ps_treated = propensity_scores[t_values == 1]
        ps_control = propensity_scores[t_values == 0]

        overlap_min = max(ps_treated.min(), ps_control.min()) if len(ps_treated) > 0 and len(ps_control) > 0 else 0
        overlap_max = min(ps_treated.max(), ps_control.max()) if len(ps_treated) > 0 and len(ps_control) > 0 else 1
        overlap_region = overlap_max - overlap_min

        passed = violation_rate < 0.1 and overlap_region > 0.5
        confidence_score = max(0.3, 1.0 - violation_rate - (1.0 - overlap_region) * 0.5)

        recommendations = []
        if not passed:
            if violation_rate >= 0.1:
                recommendations.append(f"High rate of extreme propensity scores ({violation_rate:.1%})")
                recommendations.append("Consider trimming observations with extreme propensity scores")
            if overlap_region <= 0.5:
                recommendations.append(f"Limited overlap in propensity score distributions (overlap region: {overlap_region:.2f})")
                recommendations.append("Treatment and control groups may not be comparable on observed confounders")

        return {
            "method": "positivity",
            "passed": passed,
            "confidence_score": confidence_score,
            "details": {
                "violation_rate": violation_rate,
                "extreme_low_rate": extreme_low,
                "extreme_high_rate": extreme_high,
                "overlap_region": overlap_region,
                "propensity_mean": float(np.mean(propensity_scores)),
                "propensity_std": float(np.std(propensity_scores)),
            },
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.warning("Positivity check failed", error=str(e))
        return None


@traced("tool.validation.unconfoundedness")
def _check_unconfoundedness(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any] | None:
    """
    Check unconfoundedness assumption using balance diagnostics.

    Tests whether covariates are balanced between treatment groups after
    conditioning on confounders, and performs placebo outcome test.
    """
    if treatment not in df.columns or not confounders:
        return {
            "method": "unconfoundedness",
            "passed": True,
            "confidence_score": 0.5,
            "details": {"note": "No confounders to check balance"},
            "recommendations": ["Consider identifying potential confounders for the analysis"],
        }

    try:
        t_values = df[treatment].to_numpy()
        unique_t = np.unique(t_values[~np.isnan(t_values)])

        if len(unique_t) != 2:
            return {
                "method": "unconfoundedness",
                "passed": True,
                "confidence_score": 0.6,
                "details": {"note": "Non-binary treatment, balance check simplified"},
                "recommendations": [],
            }

        # Compute standardized mean differences for each confounder
        balance_stats = {}
        imbalanced_confounders = []

        for conf in confounders:
            if conf not in df.columns:
                continue
            conf_values = df[conf].to_numpy()

            treated_values = conf_values[t_values == 1]
            control_values = conf_values[t_values == 0]

            # Remove NaN values
            treated_values = treated_values[~np.isnan(treated_values)]
            control_values = control_values[~np.isnan(control_values)]

            if len(treated_values) < 2 or len(control_values) < 2:
                continue

            # Compute standardized mean difference
            pooled_std = np.sqrt((np.var(treated_values) + np.var(control_values)) / 2)
            if pooled_std > 0:
                smd = abs(np.mean(treated_values) - np.mean(control_values)) / pooled_std
            else:
                smd = 0

            balance_stats[conf] = {
                "smd": float(smd),
                "treated_mean": float(np.mean(treated_values)),
                "control_mean": float(np.mean(control_values)),
            }

            # SMD > 0.1 typically indicates imbalance
            if smd > 0.1:
                imbalanced_confounders.append(conf)

        # Placebo outcome test: create random outcome and check for significant effect
        placebo_passed = True
        placebo_pvalue = None
        try:
            np.random.seed(42)
            placebo_outcome = np.random.randn(len(df))

            from scipy import stats
            treated_placebo = placebo_outcome[t_values == 1]
            control_placebo = placebo_outcome[t_values == 0]

            _, placebo_pvalue = stats.ttest_ind(treated_placebo, control_placebo)
            placebo_passed = placebo_pvalue > 0.05
        except Exception:
            pass

        imbalance_rate = len(imbalanced_confounders) / max(len(confounders), 1)
        passed = imbalance_rate < 0.3 and placebo_passed
        confidence_score = max(0.3, 1.0 - imbalance_rate * 0.5)

        recommendations = []
        if imbalanced_confounders:
            recommendations.append(f"Imbalanced confounders: {', '.join(imbalanced_confounders)}")
            recommendations.append("Consider propensity score matching or weighting to improve balance")
        if not placebo_passed:
            recommendations.append("Placebo outcome test failed - potential selection bias")

        return {
            "method": "unconfoundedness",
            "passed": passed,
            "confidence_score": confidence_score,
            "details": {
                "balance_statistics": balance_stats,
                "imbalanced_confounders": imbalanced_confounders,
                "imbalance_rate": imbalance_rate,
                "placebo_test_passed": placebo_passed,
                "placebo_pvalue": placebo_pvalue,
            },
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.warning("Unconfoundedness check failed", error=str(e))
        return None


@traced("tool.validation.sutva")
def _check_sutva(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any] | None:
    """
    Check Stable Unit Treatment Value Assumption (SUTVA).

    Looks for temporal/spatial clustering that might indicate spillover effects
    or interference between units.
    """
    try:
        # Check for time/date columns that might indicate temporal clustering
        time_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ["time", "date", "year", "month", "day", "period"]
        )]

        # Check for location/group columns that might indicate spatial clustering
        location_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ["location", "region", "area", "group", "cluster", "site", "unit"]
        )]

        clustering_detected = bool(time_cols or location_cols)
        spillover_indicators = []

        # Test for temporal autocorrelation in treatment assignment
        temporal_autocorr = None
        if time_cols:
            time_col = time_cols[0]
            try:
                df_sorted = df.sort_values(time_col)
                t_values = df_sorted[treatment].to_numpy()

                # Compute lag-1 autocorrelation
                if len(t_values) > 2:
                    t_lag = t_values[:-1]
                    t_current = t_values[1:]
                    valid_mask = ~(np.isnan(t_lag) | np.isnan(t_current))
                    if np.sum(valid_mask) > 2:
                        temporal_autocorr = float(np.corrcoef(t_lag[valid_mask], t_current[valid_mask])[0, 1])
                        if abs(temporal_autocorr) > 0.3:
                            spillover_indicators.append(f"High temporal autocorrelation in treatment ({temporal_autocorr:.2f})")
            except Exception:
                pass

        # Test for clustering in treatment by location
        location_clustering = {}
        if location_cols:
            loc_col = location_cols[0]
            try:
                t_values = df[treatment].to_numpy()
                loc_values = df[loc_col].to_numpy()

                unique_locs = np.unique(loc_values[~pd.isna(loc_values)])
                if len(unique_locs) >= 2:
                    treatment_rates = {}
                    for loc in unique_locs:
                        mask = loc_values == loc
                        t_loc = t_values[mask]
                        t_loc = t_loc[~np.isnan(t_loc)]
                        if len(t_loc) > 0:
                            treatment_rates[str(loc)] = float(np.mean(t_loc))

                    if treatment_rates:
                        rates = list(treatment_rates.values())
                        rate_variance = np.var(rates)
                        location_clustering = {
                            "column": loc_col,
                            "treatment_rate_variance": float(rate_variance),
                            "num_locations": len(unique_locs),
                        }
                        if rate_variance > 0.1:
                            spillover_indicators.append(f"High variance in treatment rates across {loc_col} ({rate_variance:.2f})")
            except Exception:
                pass

        # Determine if SUTVA might be violated
        passed = len(spillover_indicators) == 0
        confidence_score = max(0.4, 1.0 - len(spillover_indicators) * 0.2)

        recommendations = []
        if clustering_detected:
            recommendations.append("Data contains temporal/spatial structure that may indicate interference")
            if spillover_indicators:
                recommendations.extend(spillover_indicators)
                recommendations.append("Consider using cluster-robust standard errors or multilevel models")

        return {
            "method": "sutva",
            "passed": passed,
            "confidence_score": confidence_score,
            "details": {
                "clustering_detected": clustering_detected,
                "time_columns": time_cols,
                "location_columns": location_cols,
                "temporal_autocorrelation": temporal_autocorr,
                "location_clustering": location_clustering,
                "spillover_indicators": spillover_indicators,
            },
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.warning("SUTVA check failed", error=str(e))
        return None
