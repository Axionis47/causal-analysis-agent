# GCP Setup

This project expects a GCP project named `plotpointe` with Cloud Run, Cloud SQL, Memorystore, Cloud Storage, and Secret Manager configured.

## Required APIs

- Cloud Run
- Cloud SQL Admin
- Memorystore for Redis
- Cloud Storage
- Secret Manager
- Cloud Build
- Container Registry
- Vertex AI

## Resource Names (Recommended)

- Cloud SQL: `plotpointe-postgres`
- Redis: `plotpointe-redis`
- Buckets:
  - `plotpointe-datasets`
  - `plotpointe-reports`
- Service account: `plotpointe-ci-cd@plotpointe.iam.gserviceaccount.com`

## Connection Strings

Store these in your secrets or `.env` values:

- `DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>:5432/causal_analysis`
- `REDIS_URL=redis://<host>:6379/0`

## CLI Reference

```bash
# Create project
 gcloud projects create plotpointe --name="PlotPointe Causal Analysis"

# Enable APIs
 gcloud services enable run.googleapis.com sqladmin.googleapis.com redis.googleapis.com \
  storage.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com \
  containerregistry.googleapis.com aiplatform.googleapis.com

# Cloud SQL
 gcloud sql instances create plotpointe-postgres \
  --database-version=POSTGRES_15 --tier=db-custom-2-7680 --region=us-central1

# Redis
 gcloud redis instances create plotpointe-redis \
  --size=5 --region=us-central1 --redis-version=redis_7_0 --tier=standard

# Buckets
 gsutil mb -l us-central1 gs://plotpointe-datasets
 gsutil mb -l us-central1 gs://plotpointe-reports
```

## Secrets

Create Secret Manager secrets for:

- `vertex-ai-api-key`
- `openai-api-key`
- `anthropic-api-key`
- `langsmith-api-key`
- `kaggle-api-credentials`
- `database-password`

## Local Auth

```bash
gcloud auth login
gcloud config set project plotpointe
gcloud config set compute/region us-central1
```

## Scripts

- `infrastructure/gcp/setup.sh` contains the bootstrap commands for the resources above.
