"""Admin authentication utilities."""

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.core.auth import get_current_user
from app.core.config import settings


async def get_current_admin_user(
    user_id: Annotated[uuid.UUID, Depends(get_current_user)],
) -> uuid.UUID:
    """
    Dependency to verify the current user is an admin.

    Checks if the user's UUID is in the ADMIN_USER_IDS configuration.

    Args:
        user_id: Current user's UUID from authentication

    Returns:
        The user_id if user is an admin

    Raises:
        HTTPException: 403 Forbidden if user is not an admin
    """
    if str(user_id) not in settings.ADMIN_USER_IDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user_id
