"""Rate limiting configuration using SlowAPI."""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings


def get_user_id_from_request(request: Request) -> str:
    """
    Get user ID from request state for per-user rate limiting.

    Falls back to IP address if user_id is not set (unauthenticated requests).
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return str(user_id)
    return get_remote_address(request)


# Initialize SlowAPI rate limiter with per-user key function
limiter = Limiter(
    key_func=get_user_id_from_request,
    storage_uri=settings.RATE_LIMIT_STORAGE_URL,
    enabled=settings.RATE_LIMIT_ENABLED,
)

