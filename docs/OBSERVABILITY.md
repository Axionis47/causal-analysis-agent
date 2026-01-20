# Observability

## Structured Logging

Logs are emitted in JSON for machine parsing and include consistent context fields. Context variables
(request_id, user_id, analysis_id) are bound per request/task and injected into every log record.
Sensitive keys are automatically redacted before output (for example: api_key, token, password).

## Sentry Integration

Errors are reported to Sentry when `SENTRY_DSN` is configured. The SDK is initialized with FastAPI
and Celery integrations, tags the environment/app version, and redacts sensitive data before
sending. Configure alerts and dashboards in Sentry using the project that matches your deployment.

## Request Tracing

Each request receives an `X-Request-ID` header. If a client supplies one, it is propagated; otherwise
a UUID is generated. The request ID is bound to logs and included in responses to enable tracing
across services and async tasks.

## Log Aggregation

Ship JSON logs to your aggregator of choice:
- ELK/Elastic Stack for search and dashboards
- Datadog for APM + log correlation
- CloudWatch Logs for AWS-native ingestion

## Debugging Guide

Use context keys to follow a single request or analysis:
- `analysis_id`: tracks an analysis across agents and tasks
- `user_id`: tracks a user session
- `request_id`: tracks a single API request

## Performance Monitoring

Agent and task logs include duration metrics to help identify slow LLM calls and processing steps.
Use these fields to build latency dashboards and set alert thresholds.

## Example Log Queries

```
# Find all logs for a specific analysis
analysis_id:"abc-123"

# Find all errors for a user
user_id:"user-456" AND level:"error"

# Find slow LLM calls
duration_ms:>5000 AND module:"agents.eda"
```
