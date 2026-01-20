"""User profile API endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, get_password_hash
from app.db.database import get_async_session
from app.models.user import User
from app.schemas.auth import UserResponse, UserUpdate

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    responses={
        200: {
            "description": "Current user's profile information",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "email": "user@example.com",
                        "created_at": "2026-01-20T12:00:00Z",
                    }
                }
            },
        },
        401: {"description": "Authentication required - missing or invalid Bearer token"},
        404: {"description": "User not found (account may have been deleted)"},
    },
)
async def get_current_user_profile(
    current_user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
) -> User:
    """
    Get the current authenticated user's profile.

    Returns the profile information for the user associated with the
    provided Bearer token. Use this endpoint to:

    - Verify authentication status
    - Display user information in the UI
    - Check account creation date

    **Response Fields:**
    - `id`: Unique user identifier (UUID)
    - `email`: User's email address
    - `created_at`: Account creation timestamp
    """
    result = await db.execute(select(User).where(User.id == current_user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
    responses={
        200: {
            "description": "Profile updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "email": "newemail@example.com",
                        "created_at": "2026-01-20T12:00:00Z",
                    }
                }
            },
        },
        400: {
            "description": "Email already in use by another account",
            "content": {
                "application/json": {
                    "example": {"detail": "Email already in use"}
                }
            },
        },
        401: {"description": "Authentication required"},
        404: {"description": "User not found"},
        422: {
            "description": "Validation error - invalid email or weak password",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "email"],
                                "msg": "value is not a valid email address",
                                "type": "value_error.email",
                            }
                        ]
                    }
                }
            },
        },
    },
)
async def update_current_user_profile(
    update_data: UserUpdate,
    current_user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
) -> User:
    """
    Update the current authenticated user's profile.

    Allows updating email address and/or password. All fields are optional -
    only provided fields will be updated.

    **Request Examples:**

    Update email only:
    ```json
    {
        "email": "newemail@example.com"
    }
    ```

    Update password only:
    ```json
    {
        "password": "newSecurePassword123"
    }
    ```

    Update both:
    ```json
    {
        "email": "newemail@example.com",
        "password": "newSecurePassword123"
    }
    ```

    **Email Update:**
    - New email must be unique (not used by another account)
    - No verification email is sent (immediate update)

    **Password Update:**
    - Minimum 8 characters
    - Password is hashed using bcrypt
    - Existing sessions remain valid (tokens not invalidated)
    """
    result = await db.execute(select(User).where(User.id == current_user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update email if provided
    if update_data.email is not None:
        # Check if email is already taken by another user
        email_check = await db.execute(
            select(User).where(User.email == update_data.email, User.id != current_user_id)
        )
        if email_check.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use",
            )
        user.email = update_data.email

    # Update password if provided
    if update_data.password is not None:
        user.hashed_password = get_password_hash(update_data.password)

    await db.flush()
    await db.refresh(user)
    return user
