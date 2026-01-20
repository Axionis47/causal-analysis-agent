# Rate Limiting and Quota Management

This document describes the rate limiting and quota management system implemented in the Causal Analysis API.

## Overview

The system implements multi-layered rate limiting to prevent abuse and ensure fair resource allocation:

1. **HTTP-level rate limiting** using SlowAPI (slowapi)
2. **Concurrent analysis limits** enforced at the database level
3. **Cost tracking** for LLM token usage

## Configuration

Rate limiting is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable/disable rate limiting |
| `RATE_LIMIT_ANALYSES_PER_HOUR` | `10` | Maximum analyses per hour per user |
| `RATE_LIMIT_ANALYSES_PER_DAY` | `50` | Maximum analyses per day per user |
| `RATE_LIMIT_CONCURRENT_ANALYSES` | `3` | Maximum running analyses per user |
| `RATE_LIMIT_STORAGE_URL` | `redis://localhost:6379/3` | Redis URL for rate limit state |
| `ADMIN_USER_IDS` | `[]` | Comma-separated list of admin user UUIDs |

## Rate Limit Headers

The API includes rate limit information in response headers:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit-Hourly` | Maximum analyses per hour |
| `X-RateLimit-Remaining-Hourly` | Remaining analyses this hour |
| `X-RateLimit-Limit-Daily` | Maximum analyses per day |
| `X-RateLimit-Remaining-Daily` | Remaining analyses today |
| `X-RateLimit-Reset` | Unix timestamp when hourly limit resets |
| `Retry-After` | Seconds to wait before retrying (on 429) |

## Rate Limiting Flow

```
┌─────────┐    POST /api/v1/analyses    ┌─────────┐
│ Client  │ ─────────────────────────────▶│   API   │
└─────────┘                               └────┬────┘
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                    ┌─────────────────┐ ┌────────────┐ ┌────────────────┐
                    │  Rate Limiter   │ │   Quota    │ │   Database     │
                    │    (Redis)      │ │  Manager   │ │   (Postgres)   │
                    └────────┬────────┘ └─────┬──────┘ └───────┬────────┘
                             │                │                │
                 Check hourly/daily    Check concurrent  Create analysis
                    limits             analysis count
                             │                │                │
                             ▼                ▼                ▼
                    ┌─────────────────────────────────────────────────┐
                    │              All checks pass?                   │
                    │                                                 │
                    │  YES → Create analysis, increment counters      │
                    │  NO  → Return 429 Too Many Requests             │
                    └─────────────────────────────────────────────────┘
```

## Sliding Window Algorithm

The rate limiter uses Redis sorted sets to implement a sliding window algorithm:

1. Each request timestamp is stored in a sorted set with the timestamp as both score and member
2. On each check, expired entries (outside the window) are removed
3. The count of remaining entries determines if the user is within limits
4. TTL is set on keys to ensure automatic cleanup

This approach provides accurate rate limiting without the "burst at window boundary" problem of fixed windows.

## Cost Tracking

LLM token usage is tracked for each analysis:

### Model Pricing (per 1K tokens)

| Model | Price (USD) |
|-------|-------------|
| GPT-4 | $0.030 |
| GPT-4 Turbo | $0.010 |
| GPT-3.5 Turbo | $0.002 |
| Claude 3 Opus | $0.015 |
| Claude 3 Sonnet | $0.003 |
| Claude 3 Haiku | $0.00025 |
| Gemini Pro | $0.00125 |

### Token Estimation

When actual token counts are unavailable, the system estimates using ~4 characters per token.

## Admin API

Administrators can view and manage user quotas:

### Endpoints

- `GET /api/v1/admin/users/{user_id}/quota` - Get user's current quota usage
- `GET /api/v1/admin/users/{user_id}/usage` - Get detailed usage statistics
- `GET /api/v1/admin/users/{user_id}/analyses` - List user's analyses with costs
- `POST /api/v1/admin/users/{user_id}/quota/reset` - Reset rate limits
- `GET /api/v1/admin/stats` - Aggregate statistics
- `GET /api/v1/admin/users` - List all users with usage summaries

### Example Response

```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "analyses_this_hour": 7,
  "analyses_this_day": 23,
  "running_analyses": 2,
  "total_tokens_used": 145230,
  "total_cost": 4.36,
  "limits": {
    "hourly": 10,
    "daily": 50,
    "concurrent": 3
  }
}
```

## Error Responses

### 429 Too Many Requests

Returned when any rate limit is exceeded:

```json
{
  "detail": "Hourly rate limit exceeded. Limit: 10/hour. Remaining: 0."
}
```

Headers:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705678800
Retry-After: 3600
```

## Troubleshooting

### Rate limits not being enforced

1. Check `RATE_LIMIT_ENABLED` is set to `true`
2. Verify Redis is running and accessible at `RATE_LIMIT_STORAGE_URL`
3. Check Redis logs for connection errors

### User stuck at limit but should have quota

1. Admin can reset user's quota: `POST /api/v1/admin/users/{id}/quota/reset`
2. Check if user has running analyses counting toward concurrent limit
3. Verify Redis sorted set data with `redis-cli ZRANGE rate_limit:{user_id}:hourly 0 -1`

### Cost tracking shows 0

1. Verify agents are calling `_track_llm_usage()` after LLM calls
2. Check database for `llm_tokens_used` and `estimated_cost` columns
3. Review agent logs for tracking errors

## Database Schema

The `analyses` table includes cost tracking columns:

```sql
ALTER TABLE analyses
ADD COLUMN llm_tokens_used INTEGER NOT NULL DEFAULT 0,
ADD COLUMN estimated_cost NUMERIC(10, 4) NOT NULL DEFAULT 0.0;

CREATE INDEX ix_analyses_user_id_created_at ON analyses(user_id, created_at);
```

## Redis Key Structure

Rate limit data is stored in Redis with the following key pattern:

```
rate_limit:{user_id}:{limit_type}
```

Example:
```
rate_limit:123e4567-e89b-12d3-a456-426614174000:hourly
rate_limit:123e4567-e89b-12d3-a456-426614174000:daily
```

Each key is a sorted set where:
- Member: `{timestamp}:{user_id}`
- Score: timestamp

TTL is set to window duration + 60 seconds for automatic cleanup.
