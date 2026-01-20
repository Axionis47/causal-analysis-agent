# API Guide

This document provides a comprehensive guide to the Causal Analysis API endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

---

## Authentication

All endpoints (except public share links) require authentication via Bearer token:

```http
Authorization: Bearer <your-access-token>
```

### Registration

Register a new user account.

```http
POST /api/v1/auth/register
```

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "created_at": "2026-01-20T12:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securePassword123"}'
```

### Login

Authenticate and obtain an access token.

```http
POST /api/v1/auth/token
```

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securePassword123"}'
```

**Token Usage:**
```bash
# Store token in environment variable
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Use in subsequent requests
curl http://localhost:8000/api/v1/analyses \
  -H "Authorization: Bearer $TOKEN"
```

**Token Details:**
- Expires after 30 minutes (configurable)
- Algorithm: HS256
- Payload contains: `user_id`, `exp`

### Password Reset Flow

**Step 1: Request Reset**
```http
POST /api/v1/auth/password-reset
```

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response:** `202 Accepted`
```json
{
  "message": "If an account with that email exists, a password reset link has been sent."
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/password-reset \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com"}'
```

**Step 2: Confirm Reset**
```http
POST /api/v1/auth/password-reset/confirm
```

**Request:**
```json
{
  "token": "abc123def456...",
  "new_password": "newSecurePassword123"
}
```

**Response:** `200 OK`
```json
{
  "message": "Password has been reset successfully."
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/password-reset/confirm \
  -H "Content-Type: application/json" \
  -d '{"token": "abc123def456...", "new_password": "newSecurePassword123"}'
```

---

## User Profile Endpoints

### Get Current User
```http
GET /api/v1/users/me
```

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "created_at": "2026-01-20T12:00:00Z"
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer $TOKEN"
```

### Update Profile
```http
PATCH /api/v1/users/me
```

**Request (update email):**
```json
{
  "email": "newemail@example.com"
}
```

**Request (update password):**
```json
{
  "password": "newSecurePassword123"
}
```

**cURL Example:**
```bash
curl -X PATCH http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "newemail@example.com"}'
```

---

## Data Quality Preview

Preview dataset quality before creating an analysis.

```http
POST /api/v1/analyses/preview
```

**Request:**
```json
{
  "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
  "sample_rows": 10000
}
```

**Response:** `200 OK`
```json
{
  "passed": true,
  "warnings": [
    {
      "level": "warning",
      "code": "HIGH_MISSING_RATE",
      "message": "Column 'age' has 15% missing values",
      "column": "age"
    }
  ],
  "can_proceed_with_override": true,
  "dataset_info": {
    "row_count": 1000,
    "column_count": 10,
    "columns_preview": ["id", "age", "income", "education"]
  },
  "estimated_duration_seconds": 300
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/preview \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/datasets/uciml/iris"}'
```

**Warning Levels:**
- `error`: Analysis cannot proceed
- `warning`: Can proceed with caution
- `info`: Informational only

**Override Behavior:**
If warnings are present but `can_proceed_with_override` is true, create analysis with:
```json
{
  "kaggle_url": "...",
  "config": {
    "override_quality_warnings": true
  }
}
```

---

## Preprocessing Preview

Preview preprocessing transformations before running full analysis.

```http
POST /api/v1/analyses/{id}/preview-preprocessing
```

**Request:**
```json
{
  "config": {
    "imputation": {"strategy": "median"},
    "outlier_handling": {"method": "winsorize", "threshold": 3},
    "encoding": {"method": "one_hot"},
    "scaling": {"method": "standard"}
  }
}
```

**Response:** `200 OK`
```json
{
  "original_shape": [1000, 10],
  "preprocessed_shape": [950, 12],
  "steps_applied": [
    {
      "step_type": "imputation",
      "affected_columns": ["age", "income"],
      "parameters": {"strategy": "median"},
      "statistics": {"rows_affected": 50}
    },
    {
      "step_type": "encoding",
      "affected_columns": ["gender"],
      "parameters": {"method": "one_hot"},
      "statistics": {"new_columns": ["gender_M", "gender_F"]}
    }
  ],
  "sample_data": {
    "rows": [{"age": 25, "income": 50000, "gender_M": 1, "gender_F": 0}]
  },
  "column_changes": {
    "added": ["gender_M", "gender_F"],
    "removed": ["gender"],
    "modified": ["age", "income"]
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/{analysis_id}/preview-preprocessing \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"config": {"imputation": {"strategy": "median"}}}'
```

**Preprocessing Options:**
- `imputation`: Fill missing values (mean, median, mode, constant)
- `outlier_handling`: Handle outliers (clip, remove, winsorize)
- `encoding`: Encode categoricals (one_hot, label, target)
- `scaling`: Scale numerics (standard, minmax, robust)
- `feature_engineering`: Create new features (interactions, polynomial)

## Analysis Endpoints

### Create Analysis
```http
POST /api/v1/analyses
```

Creates a new causal analysis job.

**Request Body:**
```json
{
  "kaggle_url": "https://www.kaggle.com/datasets/username/dataset-name",
  "config": {
    "override_quality_warnings": false
  }
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "kaggle_url": "https://www.kaggle.com/datasets/username/dataset-name",
  "status": "pending",
  "config": {},
  "created_at": "2026-01-20T12:00:00Z"
}
```

### List Analyses
```http
GET /api/v1/analyses?skip=0&limit=100
```

Returns paginated list of user's analyses.

### Get Analysis
```http
GET /api/v1/analyses/{id}
```

Returns analysis details with all related entities.

### Delete Analysis
```http
DELETE /api/v1/analyses/{id}
```

Deletes an analysis and cancels any running tasks.

### Stream Progress
```http
GET /api/v1/analyses/{id}/stream
```

Server-Sent Events stream for real-time progress updates.

---

## Version Endpoints

### List Versions
```http
GET /api/v1/analyses/{analysis_id}/versions?skip=0&limit=100
```

Returns paginated list of configuration versions.

**Response:**
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
      "version_number": 3,
      "config": {"treatment": "x", "outcome": "y"},
      "config_hash": "abc123...",
      "changed_by": "550e8400-e29b-41d4-a716-446655440002",
      "change_summary": "Updated treatment variable",
      "is_manual_snapshot": false,
      "created_at": "2026-01-20T15:00:00Z"
    }
  ],
  "total": 3,
  "skip": 0,
  "limit": 100
}
```

### Get Version
```http
GET /api/v1/analyses/{analysis_id}/versions/{version_number}
```

Returns a specific version by its number.

### Create Manual Snapshot
```http
POST /api/v1/analyses/{analysis_id}/versions/snapshot
```

Creates a manual snapshot of the current configuration.

**Request Body:**
```json
{
  "change_summary": "Pre-deployment snapshot"
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "version_number": 4,
  "is_manual_snapshot": true,
  "change_summary": "Pre-deployment snapshot",
  ...
}
```

### Compare Versions
```http
GET /api/v1/analyses/{analysis_id}/versions/{v1}/compare/{v2}
```

Compares two versions and returns detailed diff.

**Response:**
```json
{
  "version1": {...},
  "version2": {...},
  "config_diff": {
    "added": {},
    "removed": {},
    "modified": {
      "treatment_variable": {
        "old": "x",
        "new": "y"
      }
    },
    "unchanged": {"outcome_variable": "z"}
  },
  "diff_summary": "1 field modified",
  "similarity_score": 0.95
}
```

### Revert to Version
```http
POST /api/v1/analyses/{analysis_id}/versions/{version_number}/revert
```

Reverts the analysis to a previous version's configuration.

**Request Body:**
```json
{
  "version_number": 2,
  "confirmation": true
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440004",
  "version_number": 5,
  "change_summary": "Reverted to version 2",
  "parent_version_id": "550e8400-e29b-41d4-a716-446655440002",
  ...
}
```

### Get Version Timeline
```http
GET /api/v1/analyses/{analysis_id}/versions/timeline
```

Returns version history formatted for timeline visualization.

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "timeline": [
    {
      "version_number": 3,
      "created_at": "2026-01-20T15:00:00Z",
      "changed_by": "550e8400-e29b-41d4-a716-446655440002",
      "change_summary": "Updated config",
      "is_manual_snapshot": false,
      "config_hash_short": "abc123...",
      "label": "v3"
    }
  ],
  "total_versions": 3,
  "latest_version": 3
}
```

### Get Version Statistics
```http
GET /api/v1/analyses/{analysis_id}/versions/stats
```

Returns statistics about version history.

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_versions": 5,
  "latest_version_number": 5,
  "auto_versions": 3,
  "manual_snapshots": 1,
  "reverts": 1,
  "first_version_date": "2026-01-20T12:00:00Z",
  "last_version_date": "2026-01-20T18:00:00Z"
}
```

---

## Results Endpoints

### Get Summary
```http
GET /api/v1/results/{analysis_id}/summary
```

Returns executive summary of analysis results.

### Get Technical Report
```http
GET /api/v1/results/{analysis_id}/technical
```

Returns detailed technical report.

### Get Visualizations
```http
GET /api/v1/results/{analysis_id}/visualizations
```

Returns visualization data including causal graphs.

### Get Validation Results
```http
GET /api/v1/results/{analysis_id}/validation
```

Returns validation and refutation test results.

### Export Results
```http
GET /api/v1/results/{analysis_id}/export
```

Exports all results as JSON.

### Download Report
```http
GET /api/v1/results/{analysis_id}/download?format=pdf
```

Downloads formatted report in specified format (pdf, markdown, html, pptx).

---

## Compare Analyses

```http
GET /api/v1/analyses/compare?ids=id1,id2,id3
```

Compares 2-3 analyses and returns comparison metrics.

**Response:**
```json
{
  "analyses": [...],
  "comparison_metrics": {
    "graph_similarity": 0.85,
    "ate_differences": [...],
    "confidence_deltas": {...}
  },
  "summary": {
    "num_analyses": 2,
    "graph_similarity_percent": 85.0,
    "agreement_level": "high"
  }
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 401 Unauthorized
```json
{
  "detail": "Not authenticated"
}
```

### 403 Forbidden
```json
{
  "detail": "Not authorized to access this resource"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded"
}
```

---

## cURL Examples

### Create Analysis
```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/datasets/test/sample"}'
```

### List Versions
```bash
curl http://localhost:8000/api/v1/analyses/{analysis_id}/versions \
  -H "Authorization: Bearer $TOKEN"
```

### Create Snapshot
```bash
curl -X POST http://localhost:8000/api/v1/analyses/{analysis_id}/versions/snapshot \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"change_summary": "Pre-deployment checkpoint"}'
```

### Compare Versions
```bash
curl http://localhost:8000/api/v1/analyses/{analysis_id}/versions/1/compare/3 \
  -H "Authorization: Bearer $TOKEN"
```

### Revert to Version
```bash
curl -X POST http://localhost:8000/api/v1/analyses/{analysis_id}/versions/2/revert \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"version_number": 2, "confirmation": true}'
```

---

## Admin Endpoints

All admin endpoints require admin authentication.

### Get User Quota
```http
GET /api/v1/admin/users/{user_id}/quota
```

**Response:** `200 OK`
```json
{
  "analyses_this_hour": 3,
  "analyses_this_day": 15,
  "running_analyses": 1,
  "total_tokens_used": 150000,
  "total_cost": 2.50,
  "limits": {
    "hourly": 10,
    "daily": 50,
    "concurrent": 3
  }
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/admin/users/{user_id}/quota \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Get User Usage Statistics
```http
GET /api/v1/admin/users/{user_id}/usage?days=30
```

**Response:** `200 OK`
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "period_days": 30,
  "total_analyses": 45,
  "total_tokens_used": 500000,
  "total_cost": 12.50,
  "running_analyses": 1,
  "status_breakdown": {
    "completed": 40,
    "failed": 3,
    "cancelled": 2
  },
  "rate_limits": {
    "hourly_remaining": 7,
    "daily_remaining": 35
  }
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/admin/users/{user_id}/usage?days=30" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### List User Analyses
```http
GET /api/v1/admin/users/{user_id}/analyses?skip=0&limit=100&status_filter=completed
```

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/admin/users/{user_id}/analyses?status_filter=completed" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Reset User Quota
```http
POST /api/v1/admin/users/{user_id}/quota/reset?reset_hourly=true&reset_daily=true
```

**Response:** `200 OK`
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "reset_types": ["hourly", "daily"],
  "message": "Successfully reset hourly, daily rate limits for user"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/admin/users/{user_id}/quota/reset?reset_hourly=true&reset_daily=true" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Get System Statistics
```http
GET /api/v1/admin/stats
```

**Response:** `200 OK`
```json
{
  "total_users": 150,
  "total_analyses": 2500,
  "total_tokens_used": 50000000,
  "total_cost": 125.50,
  "running_analyses": 5,
  "status_breakdown": {
    "completed": 2300,
    "failed": 150,
    "cancelled": 50
  },
  "llm_cache_hit_rate": 0.35,
  "llm_costs_by_model": {
    "gpt-4": 80.25,
    "gpt-3.5-turbo": 25.00,
    "claude-3-opus": 20.25
  },
  "llm_avg_latency_ms_by_provider": {
    "openai": 1500,
    "anthropic": 2000,
    "vertex": 1800
  }
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/admin/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Get LLM Cache Statistics
```http
GET /api/v1/admin/llm-cache/stats
```

**Response:** `200 OK`
```json
{
  "hits": 15000,
  "misses": 10000,
  "hit_rate": 0.6
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/admin/llm-cache/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Invalidate LLM Cache
```http
DELETE /api/v1/admin/llm-cache?pattern=eda:*
```

**Response:** `200 OK`
```json
{
  "deleted": 500,
  "pattern": "eda:*"
}
```

**cURL Example:**
```bash
# Invalidate all cache entries
curl -X DELETE http://localhost:8000/api/v1/admin/llm-cache \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Invalidate by pattern
curl -X DELETE "http://localhost:8000/api/v1/admin/llm-cache?pattern=eda:*" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### List LLM Logs
```http
GET /api/v1/admin/llm-logs?analysis_id=...&provider=openai&skip=0&limit=100
```

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
      "provider": "openai",
      "model": "gpt-4",
      "prompt": "Analyze the following...",
      "response": "Based on the data...",
      "prompt_tokens": 500,
      "completion_tokens": 200,
      "total_tokens": 700,
      "estimated_cost": 0.021,
      "cache_hit": false,
      "latency_ms": 1500,
      "error": null,
      "created_at": "2026-01-20T12:00:00Z"
    }
  ],
  "total": 5000,
  "skip": 0,
  "limit": 100,
  "total_cost": 125.50,
  "costs_by_model": {
    "gpt-4": 80.25,
    "gpt-3.5-turbo": 25.00
  }
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/admin/llm-logs?provider=openai&limit=50" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### List All Users
```http
GET /api/v1/admin/users?skip=0&limit=50
```

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "email": "user@example.com",
      "created_at": "2026-01-15T10:00:00Z",
      "analysis_count": 45,
      "total_tokens_used": 500000,
      "total_cost": 12.50
    }
  ],
  "total": 150,
  "skip": 0,
  "limit": 50
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/admin/users?limit=50" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## Sharing Endpoints

### Create Share Link
```http
POST /api/v1/analyses/{analysis_id}/share
```

**Request:**
```json
{
  "expires_in_days": 7,
  "is_public": false
}
```

**Response:** `200 OK`
```json
{
  "token": "abc123def456",
  "share_url": "http://localhost:3000/shared/abc123def456",
  "expires_at": "2026-01-27T12:00:00Z",
  "is_public": false,
  "view_count": 0,
  "created_at": "2026-01-20T12:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/{analysis_id}/share \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_days": 7, "is_public": true}'
```

**Share Types:**
- `is_public: true` - Anyone with link can view
- `is_public: false` - Requires authentication to view

### List Share Links
```http
GET /api/v1/analyses/{analysis_id}/shares
```

**Response:** `200 OK`
```json
{
  "shares": [
    {
      "token": "abc123def456",
      "share_url": "http://localhost:3000/shared/abc123def456",
      "expires_at": "2026-01-27T12:00:00Z",
      "is_public": false,
      "view_count": 15,
      "created_at": "2026-01-20T12:00:00Z"
    }
  ]
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/analyses/{analysis_id}/shares \
  -H "Authorization: Bearer $TOKEN"
```

### Revoke Share Link
```http
DELETE /api/v1/analyses/{analysis_id}/share/{token}
```

**Response:** `204 No Content`

**cURL Example:**
```bash
curl -X DELETE http://localhost:8000/api/v1/analyses/{analysis_id}/share/abc123def456 \
  -H "Authorization: Bearer $TOKEN"
```

### Access Shared Analysis
```http
GET /api/v1/shared/{token}
```

**Response:** `200 OK`
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "kaggle_url": "https://www.kaggle.com/datasets/uciml/iris",
  "status": "completed",
  "shared_by": {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "display_name": "John Doe",
    "email": null
  },
  "share_created_at": "2026-01-20T12:00:00Z",
  "is_public": true,
  "summary": {...},
  "technical": {...},
  "visualizations": {...},
  "validation": {...}
}
```

**cURL Example (public link):**
```bash
curl http://localhost:8000/api/v1/shared/abc123def456
```

**cURL Example (private link):**
```bash
curl http://localhost:8000/api/v1/shared/abc123def456 \
  -H "Authorization: Bearer $TOKEN"
```

---

## Comments Endpoints

### Create Comment
```http
POST /api/v1/analyses/{analysis_id}/comments
```

**Request (top-level comment):**
```json
{
  "content": "This finding aligns with our hypothesis."
}
```

**Request (reply):**
```json
{
  "content": "Agreed, the effect size is larger than expected.",
  "parent_comment_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
  "user_id": "550e8400-e29b-41d4-a716-446655440002",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440002",
    "email": "user@example.com",
    "display_name": "John Doe"
  },
  "parent_comment_id": null,
  "content": "This finding aligns with our hypothesis.",
  "is_resolved": false,
  "created_at": "2026-01-20T12:00:00Z",
  "updated_at": "2026-01-20T12:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/{analysis_id}/comments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Great analysis! The causal graph matches our domain knowledge."}'
```

### List Comments
```http
GET /api/v1/analyses/{analysis_id}/comments
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/analyses/{analysis_id}/comments \
  -H "Authorization: Bearer $TOKEN"
```

### Update Comment
```http
PATCH /api/v1/comments/{comment_id}
```

**Request:**
```json
{
  "content": "Updated comment text",
  "is_resolved": true
}
```

**cURL Example:**
```bash
curl -X PATCH http://localhost:8000/api/v1/comments/{comment_id} \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_resolved": true}'
```

### Delete Comment
```http
DELETE /api/v1/comments/{comment_id}
```

**cURL Example:**
```bash
curl -X DELETE http://localhost:8000/api/v1/comments/{comment_id} \
  -H "Authorization: Bearer $TOKEN"
```

---

## Common Workflows

### Complete Analysis Workflow

1. **Register and Login**
```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'

# Login and get token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}' | jq -r '.access_token')
```

2. **Preview Data Quality**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/preview \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/datasets/uciml/iris"}'
```

3. **Create Analysis**
```bash
ANALYSIS_ID=$(curl -s -X POST http://localhost:8000/api/v1/analyses \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/datasets/uciml/iris"}' | jq -r '.id')
```

4. **Stream Progress**
```bash
curl -N http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/stream \
  -H "Authorization: Bearer $TOKEN"
```

5. **View Results**
```bash
# Summary
curl http://localhost:8000/api/v1/results/$ANALYSIS_ID/summary \
  -H "Authorization: Bearer $TOKEN"

# Technical details
curl http://localhost:8000/api/v1/results/$ANALYSIS_ID/technical \
  -H "Authorization: Bearer $TOKEN"
```

6. **Download Report**
```bash
curl -O -J http://localhost:8000/api/v1/results/$ANALYSIS_ID/download?format=pdf \
  -H "Authorization: Bearer $TOKEN"
```

### Hypothesis Testing Workflow

1. **Create Base Analysis**
```bash
ANALYSIS_ID=$(curl -s -X POST http://localhost:8000/api/v1/analyses \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url": "https://www.kaggle.com/datasets/test/data"}' | jq -r '.id')
```

2. **Run Hypothesis Test**
```bash
NEW_ANALYSIS_ID=$(curl -s -X POST http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/hypothesis-test \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "treatment": "education",
    "outcome": "income",
    "confounders": ["age", "gender"],
    "analysis_types": ["treatment_effects"]
  }' | jq -r '.id')
```

3. **Compare Results**
```bash
curl "http://localhost:8000/api/v1/analyses/compare?ids=$ANALYSIS_ID,$NEW_ANALYSIS_ID" \
  -H "Authorization: Bearer $TOKEN"
```

### Collaboration Workflow

1. **Create Analysis**
2. **Add Comments**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/comments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Please review the causal graph for domain validity."}'
```

3. **Share Link**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/share \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_days": 30, "is_public": false}'
```

4. **Revoke Share When Done**
```bash
curl -X DELETE http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/share/abc123 \
  -H "Authorization: Bearer $TOKEN"
```

### Version Management Workflow

1. **Create Analysis and Make Changes**
2. **Create Snapshot Before Major Change**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/versions/snapshot \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"change_summary": "Checkpoint before updating treatment variable"}'
```

3. **Compare Versions**
```bash
curl http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/versions/1/compare/3 \
  -H "Authorization: Bearer $TOKEN"
```

4. **Revert If Needed**
```bash
curl -X POST http://localhost:8000/api/v1/analyses/$ANALYSIS_ID/versions/2/revert \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"version_number": 2, "confirmation": true}'
```

### Admin Monitoring Workflow

1. **View System Stats**
```bash
curl http://localhost:8000/api/v1/admin/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

2. **Check User Quota**
```bash
curl http://localhost:8000/api/v1/admin/users/$USER_ID/quota \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

3. **View LLM Logs**
```bash
curl "http://localhost:8000/api/v1/admin/llm-logs?limit=100" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

4. **Invalidate Cache If Needed**
```bash
curl -X DELETE "http://localhost:8000/api/v1/admin/llm-cache?pattern=eda:*" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## Rate Limiting

### Rate Limit Headers

Every response includes rate limit information:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit-Hourly` | Maximum analyses per hour |
| `X-RateLimit-Remaining-Hourly` | Remaining analyses this hour |
| `X-RateLimit-Limit-Daily` | Maximum analyses per day |
| `X-RateLimit-Remaining-Daily` | Remaining analyses today |
| `X-RateLimit-Reset-Hourly` | Unix timestamp when hourly limit resets |

### 429 Response Example

```json
{
  "detail": "Rate limit exceeded. Try again in 3600 seconds.",
  "retry_after": 3600
}
```

### Handling Rate Limits

```python
import requests
import time

def create_analysis_with_retry(url, token, data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=data
        )

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

---

## Error Handling

### Error Response Format

All errors return JSON with a `detail` field:

```json
{
  "detail": "Error description"
}
```

### Validation Errors (422)

```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "value is not a valid email address",
      "type": "value_error.email"
    }
  ]
}
```

### Common Error Scenarios

| Status | Scenario | Solution |
|--------|----------|----------|
| 400 | Invalid Kaggle URL | Use format `https://www.kaggle.com/datasets/user/name` |
| 401 | Token expired | Re-authenticate with `/api/v1/auth/token` |
| 403 | Wrong user | Ensure you own the resource or have share access |
| 404 | Not found | Check the resource ID is correct |
| 422 | Validation failed | Check request body matches schema |
| 429 | Rate limit | Wait for `Retry-After` seconds |

### Python Error Handler

```python
def handle_api_response(response):
    if response.status_code == 401:
        # Token expired, refresh it
        refresh_token()
        return retry_request()

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        time.sleep(retry_after)
        return retry_request()

    if response.status_code >= 400:
        error = response.json()
        raise APIError(error.get("detail", "Unknown error"))

    return response.json()
```
