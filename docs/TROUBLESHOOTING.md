# Troubleshooting Guide

This guide covers common errors and their solutions for the Causal Analysis Agent.

---

## Common Errors

### Invalid Kaggle URL

**Error:**
```json
{
  "detail": "Invalid Kaggle URL"
}
```

**Cause:** URL format not recognized by the parser.

**Solution:** Use one of these formats:
- Dataset: `https://www.kaggle.com/datasets/username/dataset-name`
- Competition: `https://www.kaggle.com/competitions/competition-name`

**Examples:**
```
# Valid dataset URLs
https://www.kaggle.com/datasets/uciml/iris
https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

# Valid competition URLs
https://www.kaggle.com/competitions/titanic
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
```

---

### Kaggle Credentials Not Configured

**Error:**
```json
{
  "detail": "Kaggle credentials not configured"
}
```

**Cause:** Missing `KAGGLE_USERNAME` or `KAGGLE_KEY` in environment.

**Solution:**

1. Get API credentials from Kaggle:
   - Go to https://www.kaggle.com/settings
   - Click "Create New Token"
   - Download `kaggle.json`

2. Add to `.env` file:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

3. Restart the backend service:
   ```bash
   docker-compose restart backend celery-worker
   ```

**Alternative:** Set credentials via admin panel if using encrypted storage.

---

### Rate Limit Exceeded (429)

**Error:**
```json
{
  "detail": "Rate limit exceeded. Try again in 3600 seconds.",
  "retry_after": 3600
}
```

**Cause:** Exceeded hourly, daily, or concurrent analysis quota.

**Solution:**

1. Check remaining quota:
   ```bash
   curl http://localhost:8000/api/v1/analyses \
     -H "Authorization: Bearer $TOKEN" \
     -v 2>&1 | grep "X-RateLimit"
   ```

2. Wait for limit reset (check `X-RateLimit-Reset-Hourly` header)

3. For admin: Reset user quota:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/admin/users/{user_id}/quota/reset" \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

---

### Maximum Concurrent Analyses Reached

**Error:**
```json
{
  "detail": "Maximum concurrent analyses reached"
}
```

**Cause:** User has 3+ running analyses (default limit).

**Solution:**

1. Check running analyses:
   ```bash
   curl "http://localhost:8000/api/v1/analyses?status=running" \
     -H "Authorization: Bearer $TOKEN"
   ```

2. Wait for existing analyses to complete

3. Or cancel running analyses:
   ```bash
   curl -X DELETE http://localhost:8000/api/v1/analyses/{id} \
     -H "Authorization: Bearer $TOKEN"
   ```

---

### Data Quality Validation Failed

**Error:**
```json
{
  "detail": "Data quality validation failed",
  "warnings": [
    {"level": "error", "code": "INSUFFICIENT_ROWS", "message": "Dataset has only 50 rows, minimum 100 required"}
  ]
}
```

**Cause:** Dataset doesn't meet minimum requirements:
- Less than 100 rows
- Less than 2 numeric columns
- More than 80% missing values in critical columns

**Solution:**

1. Preview data quality first:
   ```bash
   curl -X POST http://localhost:8000/api/v1/analyses/preview \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"kaggle_url": "your_url"}'
   ```

2. If warnings are acceptable, override:
   ```bash
   curl -X POST http://localhost:8000/api/v1/analyses \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "kaggle_url": "your_url",
       "config": {"override_quality_warnings": true}
     }'
   ```

---

### Unsupported Dataset File Type

**Error:**
```json
{
  "detail": "Unsupported dataset file type"
}
```

**Cause:** Kaggle dataset contains only unsupported file formats.

**Supported formats:**
- `.csv` (CSV files)
- `.parquet` (Parquet files)
- `.xlsx`, `.xls` (Excel files)

**Solution:**
- Choose a different dataset with supported formats
- Upload your data to Kaggle in CSV format first

---

### Analysis Stuck in Running Status

**Symptoms:**
- Analysis shows "running" for extended period (>30 minutes)
- No progress updates
- Eventually times out

**Cause:** Celery worker crashed, task timeout, or infrastructure issue.

**Solution:**

1. Check Celery worker logs:
   ```bash
   docker logs causal-analysis-celery-worker --tail 100
   ```

2. Check if worker is alive:
   ```bash
   docker ps | grep celery
   ```

3. Restart worker if needed:
   ```bash
   docker-compose restart celery-worker
   ```

4. Cancel and retry the analysis:
   ```bash
   # Cancel stuck analysis
   curl -X DELETE http://localhost:8000/api/v1/analyses/{id} \
     -H "Authorization: Bearer $TOKEN"

   # Create new analysis
   curl -X POST http://localhost:8000/api/v1/analyses \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"kaggle_url": "your_url"}'
   ```

---

### LLM API Rate Limit Exceeded

**Error in logs:**
```
LLM provider rate limit exceeded: openai
```

**Cause:** Too many LLM calls to the provider in a short time.

**Solution:**

1. Check LLM cache hit rate (higher is better):
   ```bash
   curl http://localhost:8000/api/v1/admin/llm-cache/stats \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

2. Wait for provider rate limit to reset (usually 1 minute)

3. If persistent, consider:
   - Increasing LLM API quota with provider
   - Enabling fallback providers
   - Improving cache hit rate

---

### Database Connection Failed

**Error:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Cause:** PostgreSQL not running or wrong credentials.

**Solution:**

1. Check PostgreSQL container:
   ```bash
   docker ps | grep postgres
   docker logs causal-analysis-postgres
   ```

2. Verify connection string in `.env`:
   ```
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/causal_analysis
   ```

3. Restart database:
   ```bash
   docker-compose restart postgres
   ```

4. Wait for database to be ready (check logs for "ready to accept connections")

---

### Redis Connection Failed

**Error:**
```
redis.exceptions.ConnectionError: Connection refused
```

**Cause:** Redis not running or wrong URL.

**Solution:**

1. Check Redis container:
   ```bash
   docker ps | grep redis
   docker logs causal-analysis-redis
   ```

2. Verify Redis URL in `.env`:
   ```
   REDIS_URL=redis://localhost:6379/0
   ```

3. Restart Redis:
   ```bash
   docker-compose restart redis
   ```

---

### Report Generation Failed

**Error:**
```json
{
  "detail": "Report generation failed"
}
```

**Cause:** Missing dependencies (weasyprint for PDF) or storage issues.

**Solution:**

1. Check backend logs for specific error:
   ```bash
   docker logs causal-analysis-backend --tail 100 | grep -i report
   ```

2. For PDF issues, verify weasyprint dependencies:
   ```bash
   docker exec causal-analysis-backend weasyprint --version
   ```

3. For storage issues, verify GCS credentials or local path

4. **Fallback:** Download JSON format instead:
   ```bash
   curl "http://localhost:8000/api/v1/results/{id}/download?format=json" \
     -H "Authorization: Bearer $TOKEN"
   ```

---

### Share Link Expired (410)

**Error:**
```json
{
  "detail": "Share link has expired"
}
```

**Cause:** Share link exceeded its expiration date.

**Solution:**

Create a new share link with longer expiration:
```bash
curl -X POST http://localhost:8000/api/v1/analyses/{id}/share \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_days": 30, "is_public": true}'
```

---

### Authentication Required (401)

**Error:**
```json
{
  "detail": "Not authenticated"
}
```

**Cause:** Missing or invalid Bearer token.

**Solution:**

1. Verify token is included in request:
   ```bash
   curl http://localhost:8000/api/v1/analyses \
     -H "Authorization: Bearer $TOKEN"
   ```

2. If token expired, get a new one:
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/token \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password"}'
   ```

3. Use the new token in subsequent requests

---

### Forbidden (403)

**Error:**
```json
{
  "detail": "Not authorized to access this resource"
}
```

**Cause:** Trying to access another user's analysis or admin endpoint.

**Solution:**
- Verify you own the analysis
- For shared analyses, use the share token URL
- For admin endpoints, ensure your account has admin privileges

---

## Performance Issues

### Slow Analysis Execution

**Symptoms:**
- Analysis takes longer than expected
- Progress stuck at certain stages

**Potential Causes and Solutions:**

1. **Large Dataset (>1M rows)**
   - System automatically samples large datasets
   - Consider using smaller sample in config:
     ```json
     {"config": {"max_sample_size": 100000}}
     ```

2. **LLM Provider Latency**
   - Check admin stats for provider latency:
     ```bash
     curl http://localhost:8000/api/v1/admin/stats \
       -H "Authorization: Bearer $ADMIN_TOKEN"
     ```
   - Consider switching primary provider if one is slow

3. **Database Performance**
   - Check for long-running queries
   - Add database indexes if needed

---

### High LLM Costs

**Symptoms:**
- Unexpected cost increases
- Low cache hit rate

**Solutions:**

1. Check cache hit rate (target: >30%):
   ```bash
   curl http://localhost:8000/api/v1/admin/llm-cache/stats \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

2. Review LLM logs for repeated prompts:
   ```bash
   curl "http://localhost:8000/api/v1/admin/llm-logs?limit=100" \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

3. Consider using cheaper models for simple tasks:
   - Configure cost-based routing in settings
   - Use GPT-3.5-turbo for extraction tasks

---

### Memory Issues

**Symptoms:**
- Worker crashes with OOM (Out of Memory)
- Container restarts

**Solutions:**

1. Reduce dataset sample size in config
2. Increase container memory limits in `docker-compose.yml`:
   ```yaml
   celery-worker:
     deploy:
       resources:
         limits:
           memory: 4G
   ```

3. Restart workers periodically if memory leaks suspected

---

## Debugging Tips

### Enable Debug Logging

Set in `.env`:
```
LOG_LEVEL=DEBUG
```

Restart services to apply.

### Check Sentry for Error Traces

1. Access Sentry dashboard
2. Filter by environment and time range
3. Review stack traces for root cause

### Use LangSmith for Agent Tracing

1. Enable LangSmith in `.env`:
   ```
   LANGSMITH_API_KEY=your_key
   LANGSMITH_PROJECT=causal-analysis
   ```

2. View traces at https://smith.langchain.com

### Query Database Directly

Connect to PostgreSQL:
```bash
docker exec -it causal-analysis-postgres psql -U postgres -d causal_analysis
```

Check analysis state:
```sql
SELECT id, status, current_stage, error, created_at
FROM analyses
WHERE id = 'your-analysis-id';
```

### Check Redis for Cached Data

Connect to Redis:
```bash
docker exec -it causal-analysis-redis redis-cli
```

Check keys:
```
KEYS llm:*
GET llm:your-key
```

---

## Getting Help

### Documentation

- API Guide: `docs/API_GUIDE.md`
- Architecture: `docs/ARCHITECTURE.md`
- This troubleshooting guide: `docs/TROUBLESHOOTING.md`

### API Documentation

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

### Report Issues

GitHub Issues: https://github.com/anthropics/claude-code/issues

When reporting issues, include:
1. Error message (full JSON response)
2. Analysis ID (if applicable)
3. Steps to reproduce
4. Environment (local/production)
5. Relevant logs
