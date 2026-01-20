# GitHub Repository Setup

Recommended repository: `plotpointe-causal-analysis`.

## Branching

- `main`: production deployments
- `develop`: staging

## Required Secrets (GitHub Actions)

- `GCP_PROJECT_ID`: `plotpointe`
- `GCP_SA_KEY`: service account JSON key
- `GCP_REGION`: `us-central1`

## Repository Topics

- causal-inference
- multi-agent
- langgraph
- production-ml

## Contribution Guidelines

- Run `make lint` and `make test` before opening PRs.
- Use descriptive PR titles and include testing notes.
