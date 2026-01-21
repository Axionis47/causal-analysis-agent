# Dataset Registry for E2E Evaluation

This document describes the dataset registry used for end-to-end evaluation of the causal analysis system.

## Purpose

The dataset registry (`datasets.yaml`) provides a structured collection of 8 real Kaggle datasets with:

- **Ground truth annotations**: Expected causal structures, ATE ranges, and validation thresholds
- **Metadata**: Dataset size, column counts, expected filenames
- **Analysis configurations**: Preprocessing settings and analysis types
- **Stress test mappings**: Which system features each dataset tests

This enables automated E2E testing that validates real analysis results against domain-expected outcomes without requiring dataset-specific code changes.

## Dataset Selection Criteria

Datasets were selected to provide comprehensive coverage of:

1. **Scale variations**: Small (<1k), medium (1k-50k), and large (>50k) row counts
2. **Data quality issues**: Missing values, outliers, zero-inflated data
3. **Outcome types**: Binary classification, continuous regression, multi-class
4. **Causal complexity**: Simple direct effects to complex confounder chains
5. **Encoding challenges**: High cardinality categoricals, one-hot encoded features
6. **Algorithm stress tests**: Memory pressure, sampling triggers, high dimensionality

## Coverage Matrix

| Feature | Datasets |
|---------|----------|
| Small (<1k rows) | Titanic, Pima, CausalML |
| Medium (1k-50k) | Wine, Heart, Marketing |
| Large (>50k) | Adult, Forest Cover |
| High missing values | Titanic (22% Age) |
| Outliers | Heart (Cholesterol), Pima (zeros) |
| High cardinality | Adult (occupation, native-country) |
| Binary outcome | Titanic, Heart, Pima, CausalML, Marketing |
| Continuous outcome | Wine |
| Multi-class | Forest Cover |
| Simple graph | Titanic, Wine |
| Complex confounders | Heart, Adult, Marketing |
| Latent confounders | Forest Cover (FCI test) |
| Sampling trigger | Forest Cover (>500k rows) |
| High dimensionality | Forest Cover (55 cols) |

## Ground Truth Definition Methodology

### ATE Ranges

ATE (Average Treatment Effect) ranges are defined based on:

1. **Domain knowledge**: Expected direction and magnitude from literature
2. **Conservative bounds**: Wide enough to accommodate algorithm variance
3. **Statistical reasoning**: Based on effect sizes seen in similar studies

For example, the Titanic dataset has an expected ATE range of -0.25 to -0.15 for Pclass→Survived, reflecting that 1st class passengers had 15-25% higher survival rates than 3rd class.

### Graph Structure

Expected edges are defined based on:

1. **Domain expertise**: Known causal relationships in the field
2. **Variable definitions**: Logical dependencies between features
3. **Prior research**: Published causal analyses of similar data

### Tolerance Rationale

- **ATE tolerance (±0.5)**: Accounts for sampling variance, algorithm differences, and preprocessing choices
- **Graph F1 threshold (0.7)**: Balances precision/recall, allowing some edge discovery variance while ensuring core structure is captured
- **Refutation pass rate (0.8)**: Requires most validation tests to pass while allowing for some statistical noise

## How to Add New Datasets

1. **Verify dataset availability**: Ensure the Kaggle dataset is public and stable
2. **Define causal structure**: Identify treatment, outcome, and confounders
3. **Research ground truth**: Determine expected ATE range from literature or domain expertise
4. **Add YAML entry**: Follow the schema in `datasets.yaml`
5. **Test validation**: Run the ground truth validator with sample outputs

### YAML Schema

```yaml
dataset_id: unique_snake_case_id
name: "Human-Readable Dataset Name"
kaggle_url: "https://www.kaggle.com/..."
description: >
  Multi-line description of dataset and what it tests.
metadata:
  rows: 1000
  columns: 10
  size_mb: 1.5
  kaggle_type: "dataset"  # or "competition"
  expected_file: "data.csv"
causal_structure:
  treatment: "treatment_column"
  outcome: "outcome_column"
  confounders:
    - "confounder1"
    - "confounder2"
  expected_edges:
    - { source: "var1", target: "var2" }
  graph_description: >
    Natural language description of expected DAG.
ground_truth:
  ate_range:
    min: -0.5
    max: 0.5
  ate_tolerance: 0.5
  graph_f1_threshold: 0.7
  validation_pass_rate_threshold: 0.8
analysis_config:
  analysis_types:
    - "discovery"
    - "treatment"
    - "validation"
  enable_preprocessing: true
  preprocessing_config:
    enable_imputation: true
    enable_encoding: true
    enable_scaling: true
    enable_outlier_detection: false
  override_quality_warnings: false
expected_characteristics:
  missing_percent: 0.0
  quality_issues: []
  data_types:
    column_name: "numerical"  # or "categorical" or "binary"
stress_test_focus:
  - "feature_being_tested"
```

## Dataset Details

### 1. Titanic

- **URL**: https://www.kaggle.com/c/titanic
- **Treatment**: Passenger class (Pclass)
- **Outcome**: Survival (binary)
- **Key tests**: Missing value imputation (22% Age), categorical encoding, class imbalance
- **Expected ATE**: -0.25 to -0.15 (1st class increases survival)

### 2. Heart Disease

- **URL**: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- **Treatment**: Cholesterol level (binarized)
- **Outcome**: Heart disease diagnosis (binary)
- **Key tests**: Medical confounders, outlier handling (Cholesterol)
- **Expected ATE**: 0.05 to 0.15 (high cholesterol increases disease risk)

### 3. Pima Diabetes

- **URL**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Treatment**: BMI (binarized)
- **Outcome**: Diabetes diagnosis (binary)
- **Key tests**: Zero-inflated data, class imbalance (65% negative)
- **Expected ATE**: 0.2 to 0.4 (high BMI increases diabetes risk)

### 4. Wine Quality

- **URL**: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
- **Treatment**: Alcohol content (binarized)
- **Outcome**: Quality score (continuous)
- **Key tests**: Continuous outcome, algorithm comparison
- **Expected ATE**: 0.3 to 0.6 (high alcohol improves quality)

### 5. Adult Census Income

- **URL**: https://www.kaggle.com/datasets/uciml/adult-census-income
- **Treatment**: Education level (binarized)
- **Outcome**: Income >50K (binary)
- **Key tests**: High cardinality encoding, large dataset (48k rows)
- **Expected ATE**: 0.15 to 0.25 (college increases high income probability)

### 6. CausalML Example

- **URL**: https://www.kaggle.com/datasets/vikasmalhotra08/causalml-package-example-dataset
- **Treatment**: Marketing treatment (binary)
- **Outcome**: Conversion (binary)
- **Key tests**: IV method validation, CATE estimation, uplift modeling
- **Expected ATE**: 0.05 to 0.15 (treatment increases conversion)

### 7. Digital Marketing

- **URL**: https://www.kaggle.com/datasets/rahuljangir78/causal-digital-marketing-campaign-dataset
- **Treatment**: Campaign exposure (binary)
- **Outcome**: Conversion (binary)
- **Key tests**: Marketing confounders, doubly robust estimation
- **Expected ATE**: 0.02 to 0.08 (campaign increases conversion)

### 8. Forest Cover Type

- **URL**: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
- **Treatment**: Elevation (binarized)
- **Outcome**: Cover type (binarized from multi-class)
- **Key tests**: Sampling trigger (581k rows), high dimensionality (55 cols), FCI algorithm
- **Expected ATE**: 0.1 to 0.3 (elevation affects forest cover type)

## Usage

### Loading the Registry

```python
from evals import GroundTruth, load_datasets_yaml

# Load all datasets
datasets = load_datasets_yaml()

# Use the GroundTruth class for validation
gt = GroundTruth()

# Get dataset configuration
config = gt.get_dataset("titanic")

# Get analysis config for API submission
analysis_config = gt.get_analysis_config("titanic")
```

### Validating Results

```python
from evals import GroundTruth

gt = GroundTruth()

# Validate ATE
ate_result = gt.validate_ate("titanic", estimated_ate=-0.20)
print(f"ATE validation passed: {ate_result['passed']}")

# Validate graph structure
edges = [("pclass", "survived"), ("age", "survived")]
graph_result = gt.validate_graph("titanic", edges)
print(f"Graph F1: {graph_result['f1']:.3f}")

# Validate refutation tests
ref_result = gt.validate_refutation("titanic", pass_rate=0.85)
print(f"Refutation passed: {ref_result['passed']}")

# Generate full report
report = gt.generate_report(
    "titanic",
    estimated_ate=-0.20,
    predicted_edges=edges,
    refutation_pass_rate=0.85
)
print(report)
```

## Compatibility Notes

The dataset registry is designed to be compatible with:

- **AnalysisConfig schema** (`app/schemas/analysis.py`): Analysis config structure matches API expectations
- **CausalGraph model** (`app/models/causal_graph.py`): Edge format uses `{source, target}` dictionaries
- **TreatmentEffect model** (`app/models/treatment_effect.py`): ATE field naming conventions match
- **Metrics module** (`evals/metrics.py`): Reuses `precision_recall` and `ate_bias` functions
