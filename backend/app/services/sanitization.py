"""Content sanitization service for user-generated content."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import bleach

from app.core.config import settings


class ContentSanitizer:
    """Service for sanitizing user-generated content to prevent XSS and injection attacks.

    This class provides methods for sanitizing various types of user input:
    - Comments with allowed HTML subset
    - Plain text with all HTML stripped
    - URLs with validation against malicious patterns
    """

    # Default allowed tags for comment content
    DEFAULT_ALLOWED_TAGS = ["b", "i", "a", "code", "pre", "strong", "em"]

    # Default allowed attributes
    DEFAULT_ALLOWED_ATTRIBUTES = {"a": ["href", "title"]}

    # Maximum content length
    DEFAULT_MAX_LENGTH = 10000

    # XSS patterns to detect in URLs
    XSS_URL_PATTERNS = [
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"data:", re.IGNORECASE),
        re.compile(r"vbscript:", re.IGNORECASE),
        re.compile(r"<script", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
    ]

    def __init__(
        self,
        allowed_tags: list[str] | None = None,
        allowed_attributes: dict[str, list[str]] | None = None,
        max_content_length: int | None = None,
    ):
        """Initialize the sanitizer with configuration.

        Args:
            allowed_tags: List of allowed HTML tags. Defaults to DEFAULT_ALLOWED_TAGS.
            allowed_attributes: Dict of allowed attributes per tag.
                               Defaults to DEFAULT_ALLOWED_ATTRIBUTES.
            max_content_length: Maximum content length. Defaults from settings or 10000.
        """
        self.allowed_tags = allowed_tags or self._get_config_tags()
        self.allowed_attributes = allowed_attributes or self._get_config_attributes()
        self.max_content_length = max_content_length or self._get_config_max_length()

    def _get_config_tags(self) -> list[str]:
        """Get allowed tags from settings or use defaults."""
        if hasattr(settings, "SANITIZATION_ALLOWED_TAGS"):
            return settings.SANITIZATION_ALLOWED_TAGS
        return self.DEFAULT_ALLOWED_TAGS

    def _get_config_attributes(self) -> dict[str, list[str]]:
        """Get allowed attributes from settings or use defaults."""
        if hasattr(settings, "SANITIZATION_ALLOWED_ATTRIBUTES"):
            return settings.SANITIZATION_ALLOWED_ATTRIBUTES
        return self.DEFAULT_ALLOWED_ATTRIBUTES

    def _get_config_max_length(self) -> int:
        """Get max content length from settings or use default."""
        if hasattr(settings, "SANITIZATION_MAX_CONTENT_LENGTH"):
            return settings.SANITIZATION_MAX_CONTENT_LENGTH
        return self.DEFAULT_MAX_LENGTH

    def sanitize_comment(self, content: str) -> str:
        """Sanitize comment content allowing a safe HTML subset.

        Allows: bold, italic, links, code, and preformatted text.
        Removes: all other HTML tags and dangerous attributes.

        Args:
            content: Raw comment content from user

        Returns:
            Sanitized HTML string safe for rendering
        """
        if not content:
            return ""

        # Truncate to max length
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length]

        # Sanitize with bleach
        sanitized = bleach.clean(
            content,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True,
        )

        # Additionally validate any href attributes
        sanitized = self._sanitize_href_attributes(sanitized)

        return sanitized

    def sanitize_plain_text(self, content: str, max_length: int | None = None) -> str:
        """Strip all HTML from content, returning plain text.

        Args:
            content: Raw content that may contain HTML
            max_length: Optional max length override

        Returns:
            Plain text with all HTML removed
        """
        if not content:
            return ""

        length_limit = max_length or self.max_content_length

        # Truncate first
        if len(content) > length_limit:
            content = content[:length_limit]

        # Strip all HTML
        clean = bleach.clean(content, tags=[], strip=True)

        # Normalize whitespace
        clean = " ".join(clean.split())

        return clean

    def validate_url(self, url: str) -> str:
        """Validate URL for safety against XSS patterns.

        Args:
            url: URL string to validate

        Returns:
            The validated URL

        Raises:
            ValueError: If URL contains dangerous patterns or is malformed
        """
        if not url:
            raise ValueError("URL cannot be empty")

        # Check for XSS patterns
        for pattern in self.XSS_URL_PATTERNS:
            if pattern.search(url):
                raise ValueError("URL contains potentially dangerous content")

        # Parse and validate structure
        try:
            parsed = urlparse(url)
        except Exception:
            raise ValueError("Invalid URL format")

        # Must have a scheme
        if not parsed.scheme:
            raise ValueError("URL must have a scheme (http or https)")

        # Only allow http and https schemes
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("URL scheme must be http or https")

        # Must have a netloc (domain)
        if not parsed.netloc:
            raise ValueError("URL must have a domain")

        # Reject embedded credentials
        if parsed.username or parsed.password:
            raise ValueError("URL must not contain embedded credentials")

        return url

    def _sanitize_href_attributes(self, html: str) -> str:
        """Additional sanitization for href attributes in anchor tags.

        This catches any href attributes that might contain javascript: or data: URIs
        that bleach might have allowed.

        Args:
            html: HTML string that has been through bleach

        Returns:
            HTML with dangerous href values removed
        """
        # Pattern to find href attributes
        href_pattern = re.compile(r'href\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)

        def validate_href(match: re.Match[str]) -> str:
            href_value = match.group(1)
            for pattern in self.XSS_URL_PATTERNS:
                if pattern.search(href_value):
                    return 'href="#"'
            return match.group(0)

        return href_pattern.sub(validate_href, html)

    def sanitize_dict_values(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively sanitize all string values in a dictionary.

        Args:
            data: Dictionary with potentially unsafe values

        Returns:
            Dictionary with all string values sanitized
        """

        def sanitize_value(val: Any) -> Any:
            if isinstance(val, str):
                return self.sanitize_plain_text(val)
            elif isinstance(val, dict):
                return {k: sanitize_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [sanitize_value(item) for item in val]
            return val

        return {k: sanitize_value(v) for k, v in data.items()}


# Singleton instance for convenience
_default_sanitizer: ContentSanitizer | None = None


def get_sanitizer() -> ContentSanitizer:
    """Get the default ContentSanitizer instance.

    Returns:
        ContentSanitizer singleton instance
    """
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = ContentSanitizer()
    return _default_sanitizer


def sanitize_comment(content: str) -> str:
    """Convenience function to sanitize comment content.

    Args:
        content: Raw comment content

    Returns:
        Sanitized comment HTML
    """
    return get_sanitizer().sanitize_comment(content)


def sanitize_plain_text(content: str, max_length: int | None = None) -> str:
    """Convenience function to sanitize plain text.

    Args:
        content: Raw content
        max_length: Optional max length

    Returns:
        Sanitized plain text
    """
    return get_sanitizer().sanitize_plain_text(content, max_length)


def validate_url(url: str) -> str:
    """Convenience function to validate URL.

    Args:
        url: URL to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL is invalid or dangerous
    """
    return get_sanitizer().validate_url(url)
