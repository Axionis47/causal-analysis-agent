# Security Documentation

This document describes the security measures implemented in the Causal Analysis application.

## Table of Contents

1. [CSRF Protection](#csrf-protection)
2. [Security Headers](#security-headers)
3. [Input Sanitization](#input-sanitization)
4. [Credential Encryption](#credential-encryption)
5. [CORS Configuration](#cors-configuration)
6. [SQL Injection Prevention](#sql-injection-prevention)
7. [Password Security](#password-security)
8. [Security Testing](#security-testing)
9. [Key Rotation](#key-rotation)
10. [Incident Response](#incident-response)

---

## CSRF Protection

### Implementation

Cross-Site Request Forgery (CSRF) protection is implemented using a double-submit cookie pattern:

1. **Token Generation**: A cryptographically secure token is generated using `secrets.token_urlsafe(32)`.
2. **Token Storage**: Tokens are stored in Redis with a 1-hour TTL and set in a cookie.
3. **Token Validation**: All state-changing requests (POST, PUT, PATCH, DELETE) must include the `X-CSRF-Token` header.

### Configuration

```bash
CSRF_ENABLED=true
CSRF_COOKIE_NAME=csrf_token
CSRF_HEADER_NAME=X-CSRF-Token
```

### Exempt Paths

The following paths are exempt from CSRF validation:
- `/health`
- `/api/docs`
- `/api/redoc`
- `/api/openapi.json`
- `/api/v1/auth/login`
- `/api/v1/auth/register`
- `/api/v1/auth/token`
- `/api/v1/csrf-token`

### Frontend Integration

The frontend automatically:
1. Retrieves CSRF tokens from cookies
2. Includes the `X-CSRF-Token` header in all state-changing requests
3. Refreshes tokens on 403 CSRF errors

---

## Security Headers

### Implemented Headers

| Header | Value | Purpose |
|--------|-------|---------|
| Content-Security-Policy | See below | Prevents XSS and data injection |
| X-Frame-Options | DENY | Prevents clickjacking |
| X-Content-Type-Options | nosniff | Prevents MIME type sniffing |
| X-XSS-Protection | 1; mode=block | Legacy XSS protection |
| Strict-Transport-Security | max-age=31536000; includeSubDomains | Enforces HTTPS |
| Referrer-Policy | strict-origin-when-cross-origin | Controls referrer information |
| Permissions-Policy | geolocation=(), microphone=(), camera=() | Restricts browser features |

### Content-Security-Policy

```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
font-src 'self' data:;
connect-src 'self' https://api.openai.com https://api.anthropic.com;
frame-ancestors 'none'
```

### Configuration

```bash
SECURITY_HEADERS_ENABLED=true
CSP_POLICY="default-src 'self'; ..."
ENABLE_HSTS=true
HSTS_MAX_AGE=31536000
```

---

## Input Sanitization

### Backend Sanitization

All user-generated content is sanitized using the `bleach` library:

**Allowed Tags for Comments:**
- `b`, `i`, `a`, `code`, `pre`, `strong`, `em`

**Allowed Attributes:**
- `a`: `href`, `title`

### Sanitization Functions

- `sanitize_html(value)`: Strips all HTML tags
- `sanitize_user_content(value, max_length)`: Allows safe HTML subset
- `validate_url_safe(url)`: Validates URLs against XSS patterns
- `sanitize_display_name(name)`: Sanitizes display names

### Frontend Sanitization

The frontend uses `DOMPurify` for client-side sanitization as a defense-in-depth measure:

```typescript
import { sanitizeHtml, sanitizeUrl } from '@/lib/sanitize';

// Use for rendering user content
<div dangerouslySetInnerHTML={{ __html: sanitizeHtml(content) }} />

// Use for URL validation
<a href={sanitizeUrl(url)}>Link</a>
```

### XSS Patterns Blocked

- `javascript:` URLs
- `data:` URLs
- `vbscript:` URLs
- Event handlers (`onclick`, `onerror`, etc.)
- `<script>` tags
- Embedded credentials in URLs

---

## Credential Encryption

### Algorithm

Credentials are encrypted using Fernet (AES-128-CBC with HMAC-SHA256):

1. The encryption key is derived from `CREDENTIALS_ENCRYPTION_KEY` using SHA-256
2. The key is base64-encoded to create a valid Fernet key
3. Each credential stores its `encryption_key_version` for key rotation

### Key Rotation

Key rotation is supported through versioned keys:

```bash
CREDENTIALS_ENCRYPTION_KEY_VERSION=v1
CREDENTIALS_ENCRYPTION_KEY_V1=<32-byte-key>
CREDENTIALS_ENCRYPTION_KEY_V2=<32-byte-key>  # For rotation
```

To rotate keys:
1. Set `CREDENTIALS_ENCRYPTION_KEY_V2` to a new key
2. Run the rotation function to re-encrypt all credentials
3. Update `CREDENTIALS_ENCRYPTION_KEY_VERSION` to `v2`
4. Remove the old key after verifying rotation

---

## CORS Configuration

### Validation Rules

- Origins must start with `http://` or `https://`
- Wildcards (`*`) are not allowed in production
- The `null` origin is rejected
- Each origin is validated as a proper URL

### Configuration

```bash
CORS_ORIGINS=["https://your-frontend-domain.com"]
CORS_MAX_AGE=3600
CORS_ALLOW_CREDENTIALS=true
```

### Allowed Methods

- GET
- POST
- PUT
- PATCH
- DELETE
- OPTIONS

---

## SQL Injection Prevention

### ORM Usage

All database queries use SQLAlchemy ORM with parameterized queries:

```python
# Safe - uses parameterized query
result = await session.execute(
    select(User).where(User.email == email)
)
```

### Audit Script

The SQL injection audit script runs in CI/CD to detect potential vulnerabilities:

```bash
cd backend && poetry run python -m scripts.audit_sql_injection
```

It scans for:
- Raw SQL with string concatenation
- F-strings in SQL queries
- `text()` usage outside migrations
- Unsafe `execute()` calls

---

## Password Security

### Requirements

Passwords must meet the following criteria:
- Minimum 12 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character
- Not in the common passwords list

### Storage

Passwords are hashed using bcrypt with appropriate work factor.

---

## Security Testing

### Backend Tests

```bash
cd backend && poetry run pytest tests/test_security.py -v
```

Tests cover:
- Input sanitization (XSS payloads)
- Password validation
- URL validation
- Credential encryption
- Display name sanitization

### Frontend Tests

```bash
cd frontend && npm run test -- --testPathPattern=security
```

Tests cover:
- XSS payload sanitization
- URL validation
- HTML escaping
- Display name sanitization

---

## Key Rotation

### Process

1. **Generate New Key**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Configure New Key**
   ```bash
   CREDENTIALS_ENCRYPTION_KEY_V2=<new-key>
   ```

3. **Run Rotation Script**
   ```python
   from app.services.credentials import rotate_credential_encryption
   await rotate_credential_encryption(session, "v1", "v2")
   ```

4. **Update Version**
   ```bash
   CREDENTIALS_ENCRYPTION_KEY_VERSION=v2
   ```

5. **Verify and Cleanup**
   - Verify all credentials decrypt successfully
   - Remove old key from environment

---

## Incident Response

### Security Incident Procedure

1. **Detection**: Monitor for unusual activity in logs
2. **Containment**: Disable affected accounts/tokens
3. **Investigation**: Review audit logs and traces
4. **Remediation**: Patch vulnerabilities, rotate keys
5. **Recovery**: Restore normal operations
6. **Post-Incident**: Document lessons learned

### Key Contacts

- Security Team: security@your-company.com
- On-Call Engineer: See PagerDuty

### Reporting Vulnerabilities

Please report security vulnerabilities responsibly by emailing security@your-company.com.

---

## Security Checklist

For production deployment, verify:

- [ ] CSRF protection enabled (`CSRF_ENABLED=true`)
- [ ] Security headers enabled (`SECURITY_HEADERS_ENABLED=true`)
- [ ] HSTS enabled (`ENABLE_HSTS=true`)
- [ ] CORS origins validated (no wildcards)
- [ ] Encryption keys are strong (32+ bytes)
- [ ] All secrets stored in Secret Manager
- [ ] SQL injection audit passes
- [ ] Security tests pass
- [ ] Penetration testing completed
- [ ] HTTPS enforced everywhere
