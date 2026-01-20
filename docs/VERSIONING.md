# Analysis Versioning System

This document describes the versioning system for analysis configurations in the Causal Analysis application.

## Overview

The versioning system tracks all changes to analysis configurations, enabling:
- **Audit Trail**: Complete history of who changed what and when
- **Configuration Comparison**: Side-by-side diff of any two versions
- **Rollback Capability**: Revert to any previous configuration
- **Manual Snapshots**: Create named checkpoints before major changes

## Key Concepts

### Version Types

1. **Automatic Versions**: Created whenever the analysis config is updated
2. **Manual Snapshots**: Explicitly created by users with optional descriptions
3. **Revert Versions**: Created when reverting to a previous version (preserves lineage)

### Version Immutability

Once created, versions are **never modified or deleted** (except when the parent analysis is deleted). This ensures complete audit integrity.

## Database Schema

### AnalysisVersion Model

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `analysis_id` | UUID | Foreign key to Analysis |
| `version_number` | Integer | Auto-incrementing per analysis |
| `config` | JSONB | Snapshot of the config at this version |
| `config_hash` | String(64) | SHA-256 hash for quick comparison |
| `changed_by` | UUID | User who made the change |
| `change_summary` | Text | Optional description |
| `is_manual_snapshot` | Boolean | True if manually created |
| `parent_version_id` | UUID | Set on revert to track lineage |
| `created_at` | DateTime | Timestamp |

## API Endpoints

### List Versions
```http
GET /api/v1/analyses/{analysis_id}/versions
```

Returns paginated list of versions, newest first.

**Query Parameters:**
- `skip` (int): Offset for pagination (default: 0)
- `limit` (int): Number of results (default: 100, max: 1000)

**Response:**
```json
{
  "items": [...],
  "total": 10,
  "skip": 0,
  "limit": 100
}
```

### Get Specific Version
```http
GET /api/v1/analyses/{analysis_id}/versions/{version_number}
```

Returns a single version by its number.

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
    "unchanged": {}
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
  "version_number": 3,
  "confirmation": true
}
```

**Important**: Reverting creates a NEW version with the old config. The original history is preserved.

### Get Timeline
```http
GET /api/v1/analyses/{analysis_id}/versions/timeline
```

Returns version history formatted for timeline visualization.

### Get Statistics
```http
GET /api/v1/analyses/{analysis_id}/versions/stats
```

Returns statistics about version history.

## Automatic Versioning

When updating an analysis config via the API, versions are created automatically:

```python
# This automatically creates a new version if config changed
await analysis_crud.update_with_versioning(
    db,
    db_obj=analysis,
    obj_in={"config": new_config},
    changed_by=user_id,
    change_summary="Updated treatment variable"
)
```

**Note**: If the new config is identical to the current config (same hash), no version is created.

## Config Hash

Each version stores a SHA-256 hash of the JSON-serialized config (with sorted keys). This enables:

- **Quick duplicate detection**: Skip creating versions for identical configs
- **Integrity verification**: Detect tampering or corruption
- **Fast comparison**: Check if two versions are identical without deep comparison

## Frontend Integration

The history page (`/analyses/{id}/history`) provides:

1. **Version Timeline**: Visual list of all versions with metadata
2. **Version Comparison**: Select two versions to see side-by-side diff
3. **Manual Snapshots**: Create labeled checkpoints
4. **Revert Workflow**: Restore previous configurations with confirmation

## Best Practices

### When to Create Manual Snapshots

- Before major configuration changes
- Before sharing with collaborators
- At project milestones
- Before running expensive computations

### Naming Conventions

Use descriptive change summaries:
- "Initial configuration"
- "Updated treatment variable to education_level"
- "Pre-experiment snapshot"
- "Reverted to version 3"

### Version Retention

Currently, all versions are retained indefinitely. Future enhancements may include:
- Automatic cleanup of old versions
- Version archival
- Export/import of version history

## Error Handling

| Error | HTTP Status | Description |
|-------|-------------|-------------|
| Version not found | 404 | The specified version doesn't exist |
| Analysis not found | 404 | The analysis doesn't exist |
| Unauthorized | 403 | User doesn't have access to the analysis |
| Confirmation required | 400 | Revert requires confirmation=true |

## Security Considerations

- Only the analysis owner can access/modify versions
- Shared access users can view but not modify versions
- Version data includes audit metadata (who, when)
- Config hashes provide integrity verification
