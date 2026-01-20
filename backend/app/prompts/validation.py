"""Prompt templates for validation and sensitivity analysis interpretation."""

from __future__ import annotations

import json
from typing import Any

SENSITIVITY_TEMPLATE = """Analyze the sensitivity analysis results and provide interpretation.

## Sensitivity Analysis Results
Robustness Value (RV): {robustness_value}
E-value: {evalue}
Original Effect Estimate: {effect_estimate}

## Thresholds
RV Threshold: {rv_threshold} (effects with RV above this are considered robust)
E-value Threshold: {evalue_threshold} (effects with E-value above this are considered robust)

## Context
Treatment Variable: {treatment}
Outcome Variable: {outcome}
Confounders Adjusted: {confounders}

Provide a JSON response with:
- "interpretation": A 2-3 sentence plain-language summary of what these sensitivity values mean
- "robustness_assessment": "robust", "moderate", or "sensitive" based on the values
- "key_concerns": List of potential issues with the causal estimate
- "recommendations": List of actionable next steps

Focus on practical implications for the causal claim.
"""

ASSUMPTION_TEMPLATE = """Interpret the causal assumption check results.

## Positivity Check
Violation Rate: {positivity_violation_rate}
Overlap Region: {overlap_region}
Passed: {positivity_passed}

## Unconfoundedness Check
Imbalance Rate: {imbalance_rate}
Imbalanced Confounders: {imbalanced_confounders}
Placebo Test Passed: {placebo_passed}
Passed: {unconfoundedness_passed}

## SUTVA Check
Clustering Detected: {clustering_detected}
Spillover Indicators: {spillover_indicators}
Passed: {sutva_passed}

## Context
Treatment: {treatment}
Outcome: {outcome}
Sample Size: {sample_size}

Provide a JSON response with:
- "overall_validity": "high", "moderate", or "low"
- "critical_violations": List of assumption violations that seriously threaten validity
- "interpretation": Plain-language summary of assumption check implications
- "recommendations": Prioritized list of actions to address violations
"""

REFUTATION_TEMPLATE = """Interpret the refutation test results.

## Refutation Tests
{refutation_results}

## Effect Estimate
Treatment Effect (ATE): {ate}
Confidence Interval: [{ci_lower}, {ci_upper}]

Provide a JSON response with:
- "stability_assessment": "stable", "somewhat_stable", or "unstable"
- "interpretation": Plain-language explanation of what refutation results mean
- "concerns": List of concerns raised by failed tests
- "confidence_modifier": A value between 0.5 and 1.0 to apply to the confidence score
"""


def build_sensitivity_prompt(
    robustness_value: float | None,
    evalue: float | None,
    effect_estimate: float,
    treatment: str,
    outcome: str,
    confounders: list[str],
    rv_threshold: float = 0.1,
    evalue_threshold: float = 1.5,
) -> str:
    """Build prompt for LLM interpretation of sensitivity analysis."""
    return SENSITIVITY_TEMPLATE.format(
        robustness_value=robustness_value or "N/A",
        evalue=evalue or "N/A",
        effect_estimate=effect_estimate,
        treatment=treatment,
        outcome=outcome,
        confounders=", ".join(confounders) if confounders else "None",
        rv_threshold=rv_threshold,
        evalue_threshold=evalue_threshold,
    )


def build_assumption_prompt(
    positivity_result: dict[str, Any] | None,
    unconfoundedness_result: dict[str, Any] | None,
    sutva_result: dict[str, Any] | None,
    treatment: str,
    outcome: str,
    sample_size: int,
) -> str:
    """Build prompt for LLM interpretation of assumption checks."""
    pos_details = positivity_result.get("details", {}) if positivity_result else {}
    unconf_details = unconfoundedness_result.get("details", {}) if unconfoundedness_result else {}
    sutva_details = sutva_result.get("details", {}) if sutva_result else {}

    return ASSUMPTION_TEMPLATE.format(
        positivity_violation_rate=pos_details.get("violation_rate", "N/A"),
        overlap_region=pos_details.get("overlap_region", "N/A"),
        positivity_passed=positivity_result.get("passed", "N/A") if positivity_result else "N/A",
        imbalance_rate=unconf_details.get("imbalance_rate", "N/A"),
        imbalanced_confounders=", ".join(unconf_details.get("imbalanced_confounders", [])) or "None",
        placebo_passed=unconf_details.get("placebo_test_passed", "N/A"),
        unconfoundedness_passed=unconfoundedness_result.get("passed", "N/A") if unconfoundedness_result else "N/A",
        clustering_detected=sutva_details.get("clustering_detected", "N/A"),
        spillover_indicators=", ".join(sutva_details.get("spillover_indicators", [])) or "None",
        sutva_passed=sutva_result.get("passed", "N/A") if sutva_result else "N/A",
        treatment=treatment,
        outcome=outcome,
        sample_size=sample_size,
    )


def build_refutation_prompt(
    refutation_results: list[dict[str, Any]],
    ate: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
) -> str:
    """Build prompt for LLM interpretation of refutation tests."""
    results_text = "\n".join([
        f"- {r.get('method', 'Unknown')}: {'Passed' if r.get('passed') else 'Failed'} "
        f"(p-value: {r.get('details', {}).get('p_value', 'N/A')})"
        for r in refutation_results
    ])

    return REFUTATION_TEMPLATE.format(
        refutation_results=results_text or "No refutation tests completed",
        ate=ate or "N/A",
        ci_lower=ci_lower or "N/A",
        ci_upper=ci_upper or "N/A",
    )
