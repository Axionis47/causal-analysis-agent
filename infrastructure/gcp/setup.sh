#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-plotpointe}
REGION=${REGION:-us-central1}

# Create project (if needed)
# gcloud projects create "$PROJECT_ID" --name="PlotPointe Causal Analysis"

# Enable APIs
APIS=(
  run.googleapis.com
  sqladmin.googleapis.com
  redis.googleapis.com
  storage.googleapis.com
  secretmanager.googleapis.com
  cloudbuild.googleapis.com
  containerregistry.googleapis.com
  aiplatform.googleapis.com
)

gcloud services enable "${APIS[@]}" --project "$PROJECT_ID"

# Cloud SQL
# gcloud sql instances create plotpointe-postgres \
#   --project "$PROJECT_ID" \
#   --database-version=POSTGRES_15 \
#   --tier=db-custom-2-7680 \
#   --region "$REGION" \
#   --backup --backup-start-time=03:00

# Redis
# gcloud redis instances create plotpointe-redis \
#   --project "$PROJECT_ID" \
#   --size=5 \
#   --region "$REGION" \
#   --redis-version=redis_7_0 \
#   --tier=standard

# Buckets
# gsutil mb -l "$REGION" gs://plotpointe-datasets
# gsutil mb -l "$REGION" gs://plotpointe-reports

# Service account
# gcloud iam service-accounts create plotpointe-ci-cd \
#   --project "$PROJECT_ID" \
#   --display-name="PlotPointe CI/CD Service Account"

# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member="serviceAccount:plotpointe-ci-cd@$PROJECT_ID.iam.gserviceaccount.com" \
#   --role="roles/run.admin"
