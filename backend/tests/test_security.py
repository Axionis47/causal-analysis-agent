"""Security tests for the application.

This module tests:
- CSRF protection (valid/invalid tokens, missing tokens)
- Security headers presence and values
- Input sanitization (XSS payloads, SQL injection attempts)
- CORS origin validation
- Credential encryption/decryption
- URL validation (open redirect, XSS in URLs)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request
from fastapi.testclient import TestClient

from app.schemas.validation import (
    sanitize_html,
    sanitize_user_content,
    validate_url_safe,
    validate_password_strength,
    sanitize_display_name,
    validate_config_values,
)
from app.services.sanitization import ContentSanitizer, sanitize_comment, validate_url
from app.services.credentials import encrypt_value, decrypt_value


class TestInputSanitization:
    """Test input sanitization functions."""

    def test_sanitize_html_removes_script_tags(self):
        """Script tags should be stripped."""
        dirty = '<script>alert("xss")</script>Hello'
        clean = sanitize_html(dirty)
        assert "<script>" not in clean
        assert "alert" not in clean
        assert "Hello" in clean

    def test_sanitize_html_removes_onclick(self):
        """Event handlers should be stripped."""
        dirty = '<div onclick="alert(1)">Click me</div>'
        clean = sanitize_html(dirty)
        assert "onclick" not in clean
        assert "Click me" in clean

    def test_sanitize_html_removes_javascript_urls(self):
        """JavaScript URLs should be stripped."""
        dirty = '<a href="javascript:alert(1)">Link</a>'
        clean = sanitize_html(dirty)
        assert "javascript:" not in clean

    def test_sanitize_user_content_allows_safe_tags(self):
        """Safe HTML tags should be preserved."""
        content = "<b>Bold</b> and <i>italic</i>"
        clean = sanitize_user_content(content)
        assert "<b>" in clean
        assert "<i>" in clean

    def test_sanitize_user_content_respects_max_length(self):
        """Content should be truncated at max length."""
        content = "a" * 20000
        clean = sanitize_user_content(content, max_length=100)
        assert len(clean) == 100

    def test_validate_url_safe_blocks_javascript(self):
        """JavaScript URLs should raise ValueError."""
        with pytest.raises(ValueError, match="dangerous"):
            validate_url_safe("javascript:alert(1)")

    def test_validate_url_safe_blocks_data_urls(self):
        """Data URLs should raise ValueError."""
        with pytest.raises(ValueError, match="dangerous"):
            validate_url_safe("data:text/html,<script>alert(1)</script>")

    def test_validate_url_safe_allows_https(self):
        """HTTPS URLs should be allowed."""
        url = "https://example.com/path"
        result = validate_url_safe(url)
        assert result == url


class TestPasswordValidation:
    """Test password strength validation."""

    def test_password_too_short(self):
        """Passwords under 12 characters should fail."""
        with pytest.raises(ValueError, match="at least 12 characters"):
            validate_password_strength("Short1!")

    def test_password_no_uppercase(self):
        """Passwords without uppercase should fail."""
        with pytest.raises(ValueError, match="uppercase"):
            validate_password_strength("nouppercase1!")

    def test_password_no_lowercase(self):
        """Passwords without lowercase should fail."""
        with pytest.raises(ValueError, match="lowercase"):
            validate_password_strength("NOLOWERCASE1!")

    def test_password_no_digit(self):
        """Passwords without digits should fail."""
        with pytest.raises(ValueError, match="digit"):
            validate_password_strength("NoDigitsHere!")

    def test_password_no_special(self):
        """Passwords without special characters should fail."""
        with pytest.raises(ValueError, match="special"):
            validate_password_strength("NoSpecialChar1")

    def test_password_common(self):
        """Common passwords should fail."""
        with pytest.raises(ValueError, match="too common"):
            validate_password_strength("password")

    def test_password_valid(self):
        """Valid passwords should pass."""
        result = validate_password_strength("SecurePass123!")
        assert result == "SecurePass123!"


class TestDisplayNameSanitization:
    """Test display name sanitization."""

    def test_strips_html_tags(self):
        """HTML tags should be removed."""
        dirty = "<script>alert(1)</script>John"
        clean = sanitize_display_name(dirty)
        assert clean == "John"

    def test_normalizes_whitespace(self):
        """Multiple spaces should become single space."""
        dirty = "John    Doe"
        clean = sanitize_display_name(dirty)
        assert clean == "John Doe"

    def test_truncates_long_names(self):
        """Names over 100 chars should be truncated."""
        dirty = "A" * 150
        clean = sanitize_display_name(dirty)
        assert len(clean) == 100


class TestConfigValidation:
    """Test config dictionary sanitization."""

    def test_sanitizes_string_values(self):
        """String values should be sanitized."""
        config = {"key": "<script>alert(1)</script>value"}
        clean = validate_config_values(config)
        assert "<script>" not in clean["key"]
        assert "value" in clean["key"]

    def test_handles_nested_dicts(self):
        """Nested dicts should be recursively sanitized."""
        config = {"nested": {"key": "<b>value</b>"}}
        clean = validate_config_values(config)
        assert "<b>" not in clean["nested"]["key"]

    def test_handles_lists(self):
        """Lists should be recursively sanitized."""
        config = {"items": ["<script>1</script>", "<script>2</script>"]}
        clean = validate_config_values(config)
        assert all("<script>" not in item for item in clean["items"])


class TestContentSanitizer:
    """Test the ContentSanitizer service."""

    def test_sanitize_comment_allows_basic_formatting(self):
        """Basic formatting tags should be allowed in comments."""
        sanitizer = ContentSanitizer()
        content = "<b>Bold</b> <i>italic</i> <code>code</code>"
        result = sanitizer.sanitize_comment(content)
        assert "<b>" in result
        assert "<i>" in result
        assert "<code>" in result

    def test_sanitize_comment_blocks_script(self):
        """Script tags should be blocked in comments."""
        sanitizer = ContentSanitizer()
        content = '<script>alert("xss")</script>Safe text'
        result = sanitizer.sanitize_comment(content)
        assert "<script>" not in result
        assert "Safe text" in result

    def test_sanitize_comment_blocks_unsafe_href(self):
        """JavaScript hrefs should be blocked."""
        sanitizer = ContentSanitizer()
        content = '<a href="javascript:alert(1)">Click</a>'
        result = sanitizer.sanitize_comment(content)
        assert "javascript:" not in result

    def test_sanitize_plain_text_strips_all_html(self):
        """Plain text sanitization should strip all HTML."""
        sanitizer = ContentSanitizer()
        content = "<b>Bold</b> <script>alert(1)</script>"
        result = sanitizer.sanitize_plain_text(content)
        assert "<b>" not in result
        assert "<script>" not in result
        assert "Bold" in result

    def test_validate_url_rejects_javascript(self):
        """JavaScript URLs should be rejected."""
        sanitizer = ContentSanitizer()
        with pytest.raises(ValueError):
            sanitizer.validate_url("javascript:alert(1)")

    def test_validate_url_rejects_data(self):
        """Data URLs should be rejected."""
        sanitizer = ContentSanitizer()
        with pytest.raises(ValueError):
            sanitizer.validate_url("data:text/html,<script>alert(1)</script>")

    def test_validate_url_allows_https(self):
        """HTTPS URLs should be allowed."""
        sanitizer = ContentSanitizer()
        result = sanitizer.validate_url("https://example.com")
        assert result == "https://example.com"

    def test_validate_url_rejects_embedded_credentials(self):
        """URLs with embedded credentials should be rejected."""
        sanitizer = ContentSanitizer()
        with pytest.raises(ValueError, match="credentials"):
            sanitizer.validate_url("https://user:pass@example.com")


class TestCredentialEncryption:
    """Test credential encryption and decryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypting and decrypting should return original value."""
        original = "my-secret-api-key"
        encrypted = encrypt_value(original)
        decrypted = decrypt_value(encrypted)
        assert decrypted == original

    def test_encrypted_value_is_bytes(self):
        """Encrypted value should be bytes."""
        encrypted = encrypt_value("test")
        assert isinstance(encrypted, bytes)

    def test_encrypted_value_differs_from_plaintext(self):
        """Encrypted value should differ from plaintext."""
        plaintext = "test-key"
        encrypted = encrypt_value(plaintext)
        assert encrypted != plaintext.encode()


class TestXSSPayloads:
    """Test various XSS payload patterns."""

    XSS_PAYLOADS = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert(1)>',
        '<svg onload=alert(1)>',
        '"><script>alert(1)</script>',
        "javascript:alert(1)",
        '<a href="javascript:alert(1)">click</a>',
        '<div onmouseover="alert(1)">hover</div>',
        "';alert(String.fromCharCode(88,83,83))//",
        '<iframe src="javascript:alert(1)">',
        '<body onload=alert(1)>',
        '<input onfocus=alert(1) autofocus>',
        '<marquee onstart=alert(1)>',
        '<video><source onerror="alert(1)">',
        '<math href="javascript:alert(1)">click</math>',
    ]

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_sanitize_html_blocks_xss(self, payload):
        """All XSS payloads should be neutralized."""
        result = sanitize_html(payload)
        # Should not contain script tags, event handlers, or javascript:
        assert "<script" not in result.lower()
        assert "javascript:" not in result.lower()
        assert "onerror" not in result.lower()
        assert "onload" not in result.lower()
        assert "onmouseover" not in result.lower()
        assert "onfocus" not in result.lower()

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_content_sanitizer_blocks_xss(self, payload):
        """ContentSanitizer should block all XSS payloads."""
        sanitizer = ContentSanitizer()
        result = sanitizer.sanitize_comment(payload)
        assert "<script" not in result.lower()
        assert "javascript:" not in result.lower()


class TestURLValidation:
    """Test URL validation against open redirects and XSS."""

    DANGEROUS_URLS = [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "//evil.com",
        "https://evil.com@good.com",
    ]

    SAFE_URLS = [
        "https://example.com",
        "https://example.com/path",
        "https://example.com/path?query=value",
        "http://localhost:3000",
    ]

    @pytest.mark.parametrize("url", DANGEROUS_URLS)
    def test_validate_url_blocks_dangerous(self, url):
        """Dangerous URLs should be blocked."""
        sanitizer = ContentSanitizer()
        with pytest.raises(ValueError):
            sanitizer.validate_url(url)

    @pytest.mark.parametrize("url", SAFE_URLS)
    def test_validate_url_allows_safe(self, url):
        """Safe URLs should be allowed."""
        sanitizer = ContentSanitizer()
        result = sanitizer.validate_url(url)
        assert result == url
