# Production Readiness Checklist

Use this checklist to validate the full system in GCP.

## Deployment

- [ ] Backend, frontend, worker deployed to Cloud Run
- [ ] Cloud SQL migrations applied
- [ ] Redis available and reachable
- [ ] Storage buckets accessible
- [ ] Secrets configured in Secret Manager

## Functional Workflow

- [ ] Submit Kaggle dataset URL from frontend
- [ ] SSE progress stream updates in real-time
- [ ] Results tab renders summary, technical, visualizations, validation
- [ ] Export JSON downloads successfully

## Resilience

- [ ] Worker restart resumes from checkpoint
- [ ] Timeouts enforce per-agent limits
- [ ] LLM fallback kicks in when provider fails
- [ ] Partial results stored on failure

## Observability

- [ ] LangSmith traces show agent and tool runs
- [ ] Cloud Logging captures errors
- [ ] CI/CD logs show smoke tests

## Security

- [ ] JWT auth enabled
- [ ] Kaggle credentials encrypted
- [ ] Secrets not logged
- [ ] CSRF protection enabled
- [ ] Security headers configured (CSP, X-Frame-Options, HSTS)
- [ ] CORS origins validated (no wildcards in production)
- [ ] HTTPS enforced (HSTS enabled)
- [ ] Credential encryption keys rotated and secured
- [ ] Input sanitization tested (XSS prevention)
- [ ] SQL injection audit passes
- [ ] Password strength validation enabled
- [ ] Security headers tested via security scan
- [ ] Penetration testing completed
