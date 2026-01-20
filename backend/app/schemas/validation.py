"""Reusable validators for input sanitization and validation."""

import re
from typing import Any
from urllib.parse import urlparse

import bleach

# Allowed HTML tags for user content (comments, descriptions)
ALLOWED_TAGS = ["b", "i", "a", "code", "pre", "strong", "em"]
ALLOWED_ATTRIBUTES = {"a": ["href", "title"]}

# Patterns for detecting XSS in URLs
XSS_URL_PATTERNS = [
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"data:", re.IGNORECASE),
    re.compile(r"vbscript:", re.IGNORECASE),
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
]

# Common password patterns to reject
COMMON_PASSWORDS = {
    "password",
    "123456",
    "12345678",
    "qwerty",
    "abc123",
    "monkey",
    "1234567",
    "letmein",
    "trustno1",
    "dragon",
    "baseball",
    "iloveyou",
    "master",
    "sunshine",
    "ashley",
    "bailey",
    "shadow",
    "123123",
    "654321",
    "superman",
    "qazwsx",
    "michael",
    "football",
    "password1",
    "password123",
}


def sanitize_html(value: str) -> str:
    """Strip all HTML tags from input, leaving only plain text.

    Args:
        value: Input string that may contain HTML

    Returns:
        Plain text with all HTML tags removed
    """
    return bleach.clean(value, tags=[], strip=True)


def sanitize_user_content(value: str, max_length: int = 10000) -> str:
    """Sanitize user-generated content allowing safe HTML subset.

    Allows bold, italic, links, and code formatting only.

    Args:
        value: Input string with potentially unsafe HTML
        max_length: Maximum allowed content length

    Returns:
        Sanitized HTML string with only allowed tags
    """
    # Truncate to max length first
    if len(value) > max_length:
        value = value[:max_length]

    return bleach.clean(
        value,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
    )


def validate_url_safe(value: str) -> str:
    """Validate URL is safe and doesn't contain XSS patterns.

    Args:
        value: URL string to validate

    Raises:
        ValueError: If URL contains dangerous patterns

    Returns:
        The validated URL string
    """
    for pattern in XSS_URL_PATTERNS:
        if pattern.search(value):
            raise ValueError("URL contains potentially dangerous content")

    return value


def validate_url_scheme(value: str, allowed_schemes: list[str] | None = None) -> str:
    """Validate URL has an allowed scheme.

    Args:
        value: URL string to validate
        allowed_schemes: List of allowed schemes (default: https only)

    Raises:
        ValueError: If URL scheme is not allowed

    Returns:
        The validated URL string
    """
    if allowed_schemes is None:
        allowed_schemes = ["https"]

    parsed = urlparse(value)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")

    return value


def validate_url_domain(value: str, allowed_domains: list[str]) -> str:
    """Validate URL domain is in the allowed list.

    Args:
        value: URL string to validate
        allowed_domains: List of allowed domain names

    Raises:
        ValueError: If URL domain is not allowed

    Returns:
        The validated URL string
    """
    parsed = urlparse(value)
    if parsed.netloc not in allowed_domains:
        raise ValueError(f"URL domain must be one of: {', '.join(allowed_domains)}")

    return value


def validate_url_no_credentials(value: str) -> str:
    """Validate URL doesn't contain embedded credentials.

    Args:
        value: URL string to validate

    Raises:
        ValueError: If URL contains user:pass@ pattern

    Returns:
        The validated URL string
    """
    parsed = urlparse(value)
    if parsed.username or parsed.password:
        raise ValueError("URL must not contain embedded credentials")

    return value


def validate_url_no_fragment(value: str) -> str:
    """Validate URL doesn't contain a fragment.

    Args:
        value: URL string to validate

    Raises:
        ValueError: If URL contains a fragment

    Returns:
        The validated URL string
    """
    parsed = urlparse(value)
    if parsed.fragment:
        raise ValueError("URL must not contain a fragment")

    return value


def validate_password_strength(password: str) -> str:
    """Validate password meets strength requirements.

    Requirements:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - Not in common passwords list

    Args:
        password: Password string to validate

    Raises:
        ValueError: If password doesn't meet requirements

    Returns:
        The validated password string
    """
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters long")

    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter")

    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter")

    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one digit")

    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        raise ValueError("Password must contain at least one special character")

    if password.lower() in COMMON_PASSWORDS:
        raise ValueError("Password is too common, please choose a stronger password")

    return password


def sanitize_display_name(value: str, max_length: int = 100) -> str:
    """Sanitize display name by stripping HTML and limiting length.

    Args:
        value: Display name to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized display name
    """
    # Strip all HTML
    clean = sanitize_html(value)
    # Normalize whitespace
    clean = " ".join(clean.split())
    # Truncate
    if len(clean) > max_length:
        clean = clean[:max_length]
    return clean


def validate_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize nested config dictionary values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Sanitized configuration dictionary
    """

    def sanitize_value(val: Any) -> Any:
        if isinstance(val, str):
            return sanitize_html(val)
        elif isinstance(val, dict):
            return {k: sanitize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [sanitize_value(item) for item in val]
        return val

    return {k: sanitize_value(v) for k, v in config.items()}
