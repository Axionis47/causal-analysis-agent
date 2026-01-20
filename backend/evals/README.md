# Agentic Evaluations

This folder contains synthetic data generators and metrics used to evaluate agent quality in CI and nightly runs.

## Contents

- `synthetic.py`: generates simple causal datasets with known structures
- `metrics.py`: helper metrics (precision/recall, ATE bias)

## Usage

```bash
cd backend
poetry run python -m evals.synthetic
```
