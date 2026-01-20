# Advanced Causal Methods

This document describes the advanced causal inference capabilities available in the system, including sensitivity analysis, bootstrap confidence intervals, cross-validation for causal discovery, and ensemble methods.

## Table of Contents

1. [Sensitivity Analysis](#sensitivity-analysis)
2. [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
3. [Cross-Validation for Causal Discovery](#cross-validation-for-causal-discovery)
4. [Ensemble Methods](#ensemble-methods)
5. [Robustness Checks](#robustness-checks)
6. [Configuration](#configuration)

## Sensitivity Analysis

Sensitivity analysis assesses how robust causal effect estimates are to unmeasured confounding. Two methods are implemented:

### Linear Partial R-squared (Robustness Value)

The robustness value (RV) quantifies how strong an unmeasured confounder would need to be to explain away the observed effect.

**Interpretation:**
- RV > 0.1: Effect is reasonably robust to unmeasured confounding
- RV between 0.05-0.1: Moderate sensitivity
- RV < 0.05: Effect may be sensitive to unmeasured confounding

**Output includes:**
- `robustness_value`: Overall RV
- `rv_50_percent`: Confounder strength to reduce effect by 50%
- `rv_100_percent`: Confounder strength to completely nullify effect
- `benchmark_r2`: Comparison with known confounders

### E-value

The E-value is the minimum strength of association that an unmeasured confounder would need to have with both the treatment and outcome to fully explain away the observed effect.

**Interpretation:**
- E-value > 2.0: Strong effect unlikely to be explained by confounding
- E-value 1.5-2.0: Moderately robust
- E-value < 1.5: Effect could potentially be explained by moderate confounding

**Output includes:**
- `evalue`: Main E-value estimate
- `evalue_ci`: E-value for confidence interval bound
- `is_binary_outcome`: Whether binary or continuous outcome formula was used

## Bootstrap Confidence Intervals

Bootstrap confidence intervals provide more reliable uncertainty estimates than asymptotic methods, especially for small samples.

### Implementation

For each treatment effect method (PSM, Doubly Robust, IV), bootstrap CIs are computed by:

1. Resampling the data with replacement (default: 500 iterations)
2. Re-estimating the treatment effect on each bootstrap sample
3. Computing percentile-based confidence intervals

### Configuration

```python
BOOTSTRAP_NUM_SIMULATIONS = 500  # Number of bootstrap iterations
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95  # Confidence level (95%)
```

### Output

Each treatment effect estimate includes:
- `ate_ci_lower`: Lower bound of confidence interval
- `ate_ci_upper`: Upper bound of confidence interval
- `bootstrap_std`: Standard deviation of bootstrap estimates
- `confidence_interval.samples`: First 100 bootstrap samples (for visualization)

## Cross-Validation for Causal Discovery

Cross-validation assesses the stability of discovered causal edges across data subsamples.

### Stability Selection Algorithm

1. Subsample the data (default: 80% of observations)
2. Run the discovery algorithm (PC, GES, or FCI)
3. Repeat for multiple folds (default: 5)
4. Count how often each edge appears
5. Keep only "stable" edges (appearing in >= 60% of folds)

### Output

- **Original Graph**: All edges discovered on full dataset
- **Consensus Graph**: Only stable edges (high confidence)
- `edge_frequencies`: How often each edge appeared across folds
- `mean_stability`: Average stability across all edges

### Configuration

```python
DISCOVERY_CV_FOLDS = 5  # Number of cross-validation folds
DISCOVERY_STABILITY_THRESHOLD = 0.6  # Minimum frequency for stable edge
DISCOVERY_SUBSAMPLE_FRACTION = 0.8  # Data fraction per fold
```

### Interpretation

- Edges in consensus graph: High confidence, appeared consistently
- Edges only in original: May be sensitive to sampling variation
- Low mean stability: Consider collecting more data

## Ensemble Methods

The ensemble method combines multiple treatment effect estimators for more robust estimates.

### Weighting Scheme

Each method receives a weight based on:
- Confidence score
- Confidence interval width (narrower = higher weight)

```
weight = confidence_score / (1 + ci_width)
```

### Ensemble Estimate

- Weighted average ATE across methods
- Ensemble variance using weighted variance formula
- Combined confidence interval

### Output

```json
{
  "ate": 0.25,
  "ate_ci_lower": 0.18,
  "ate_ci_upper": 0.32,
  "weights": {
    "propensity_matching": 0.35,
    "doubly_robust": 0.45,
    "iv": 0.20
  },
  "methods_used": ["propensity_matching", "doubly_robust", "iv"]
}
```

### When to Use

Ensemble estimates are most valuable when:
- Multiple methods agree (adds confidence)
- Individual methods have different assumptions
- You want a single "best" estimate

## Robustness Checks

Three causal assumption checks are performed:

### Positivity Check

Verifies P(Treatment|Confounders) > 0 for all confounder strata.

**Checks:**
- Extreme propensity scores (< 0.05 or > 0.95)
- Overlap in propensity score distributions

**Violations indicate:**
- Some subgroups never receive treatment
- Limited comparability between groups

### Unconfoundedness Check

Tests whether covariates are balanced between treatment groups.

**Checks:**
- Standardized mean differences for each confounder
- Placebo outcome test (random outcome should show no effect)

**Violations indicate:**
- Selection bias may be present
- Matching/weighting may be needed

### SUTVA Check

Stable Unit Treatment Value Assumption - no interference between units.

**Checks:**
- Temporal autocorrelation in treatment assignment
- Clustering in treatment by location/group

**Violations indicate:**
- Spillover effects may be present
- Cluster-robust methods may be needed

## Configuration

All advanced methods can be configured via environment variables:

```bash
# Bootstrap
BOOTSTRAP_NUM_SIMULATIONS=500
BOOTSTRAP_CONFIDENCE_LEVEL=0.95

# Causal Discovery CV
DISCOVERY_CV_FOLDS=5
DISCOVERY_STABILITY_THRESHOLD=0.6
DISCOVERY_SUBSAMPLE_FRACTION=0.8

# Sensitivity Thresholds
SENSITIVITY_RV_THRESHOLD=0.1
SENSITIVITY_EVALUE_THRESHOLD=1.5
```

## Computational Considerations

### Performance Impact

| Method | Approximate Time Impact |
|--------|------------------------|
| Bootstrap CI (500 sims) | 5-10x base method |
| Discovery CV (5 folds) | 5x discovery time |
| Sensitivity Analysis | 10-20x base estimate |
| Assumption Checks | ~1x base method |

### Recommendations

1. **Small datasets (<1000 rows)**: Use all methods, runtime is manageable
2. **Medium datasets (1000-10000)**: Consider reducing bootstrap iterations
3. **Large datasets (>10000)**: Consider subsampling for bootstrap

### Memory Usage

Bootstrap stores up to 100 samples per method. For large analyses with many treatment effects, consider:
- Reducing stored samples in config
- Using summary statistics only
