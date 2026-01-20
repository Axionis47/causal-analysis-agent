# Causal Analysis Types

This system supports six analysis types. You can select them in the UI or allow the EDA agent to auto-activate based on data characteristics.

## Data Quality Validation
Before running full EDA, a lightweight validator runs to surface warnings and errors. You can also call a preview endpoint to see these checks without creating an analysis.

Validation rules and thresholds:
- Row count: error if < 100 rows, warning if < 500 rows
- Missing values: error if any column > 80% missing, warning if > 50%
- Numeric columns: error if < 2 numeric columns, warning if < 3
- Variance: warning if a numeric column has zero variance
- Duplicates: warning if > 30% duplicate rows
- Class imbalance: warning if binary column imbalance > 95%

Preview endpoint:
- POST /api/v1/analyses/preview with kaggle_url and optional sample_rows (default 10000, max 50000)
- Response includes warnings, dataset info, and whether you can proceed with override

Override behavior:
- Set config.override_quality_warnings = true when creating an analysis to bypass non-critical warnings.

## Data Preprocessing

After EDA analysis completes, the system applies systematic preprocessing transformations to prepare data for discovery and treatment agents. Preprocessing is applied by default but can be configured or disabled.

### Preprocessing Steps

1. **Missing Value Imputation**
   - Numeric columns: mean imputation via scikit-learn's SimpleImputer
   - Categorical columns: mode imputation (most frequent value)
   - Columns with >80% missing are skipped (logged as warning)

2. **Outlier Handling**
   - Method: IQR (default) or z-score based detection
   - Action: Winsorization (capping at threshold boundaries)
   - IQR threshold: 1.5 (configurable, range 0.5-5.0)
   - Z-score: values with |z| > 3 are capped

3. **Categorical Encoding**
   - Low cardinality (≤10 unique): one-hot encoding with drop_first=True
   - Medium cardinality (11-50): ordinal encoding based on frequency
   - High cardinality (>50): frequency encoding (map to value counts)
   - Encoding mappings stored for inverse transformation

4. **Numeric Scaling**
   - StandardScaler applied (mean=0, std=1)
   - Binary columns (0/1 only) are skipped
   - Scaling parameters stored in step metadata

5. **Feature Engineering** (disabled by default)
   - Interaction terms: treatment × top 3 confounders
   - Polynomial features: squared terms for treatment and outcome
   - Binned features: quartile bins for highly skewed columns (|skewness| > 1)
   - Engineered features prefixed with `fe_`

### Configuration Options

```json
{
  "enable_preprocessing": true,
  "preprocessing_config": {
    "enable_imputation": true,
    "enable_outlier_handling": true,
    "enable_encoding": true,
    "enable_scaling": true,
    "enable_feature_engineering": false,
    "outlier_method": "iqr",
    "outlier_threshold": 1.5,
    "categorical_encoding_strategy": "auto"
  }
}
```

### Disabling Preprocessing

To run an analysis with raw data (no preprocessing):
```json
{
  "enable_preprocessing": false
}
```

### Preprocessing and Reproducibility

- Both raw and preprocessed datasets are stored in GCS
- Preprocessing steps and parameters are persisted in DataUnderstanding
- Full metadata enables reproducible transformations
- Preprocessing config is version-controlled with analysis

### Preview Endpoint

Preview preprocessing effects before running a full analysis:

```bash
curl -X POST /api/v1/analyses/{id}/preview-preprocessing \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "enable_imputation": true,
      "enable_outlier_handling": true,
      "outlier_threshold": 2.0
    }
  }'
```

Response includes:
- original_shape and preprocessed_shape
- steps_applied with parameters and statistics
- sample_data (first 10 rows)
- column_changes (added, removed, modified)

## Treatment Effects
- Methods: propensity score matching, doubly robust, IV (optional)
- Outputs: ATE/ATT with confidence intervals

## Causal Discovery
- Methods: PC (stable), GES/FCI (beta)
- Outputs: causal DAG, edge confidence

## Mediation
- Methods: DoWhy two-stage regression
- Outputs: direct and indirect effects via mediator

## Heterogeneous Effects
- Methods: Causal Forest or grouped effects
- Outputs: CATE by segment and dispersion

## Time-Varying Treatment
- Methods: time-binned difference in means
- Outputs: effects across temporal bins

## Instrumental Variables
- Methods: DoWhy IV estimator
- Outputs: instrumented ATE and instrument diagnostics
