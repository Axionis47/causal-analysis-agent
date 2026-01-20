"""Authentication API endpoints."""

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    create_access_token,
    generate_password_reset_token,
    get_password_hash,
    verify_password,
)
from app.core.config import settings
from app.db.database import get_async_session
from app.models.user import User
from app.schemas.auth import (
    PasswordResetConfirm,
    PasswordResetRequest,
    Token,
    UserLogin,
    UserRegister,
    UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# In-memory store for password reset tokens (in production, use Redis or database)
# Format: {token: {"user_id": uuid, "expires_at": datetime}}
_password_reset_tokens: dict[str, dict] = {}


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    responses={
        201: {
            "description": "User registered successfully",
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
        400: {
            "description": "Email already registered",
            "content": {
                "application/json": {
                    "example": {"detail": "Email already registered"}
                }
            },
        },
        422: {
            "description": "Validation error - invalid email format or weak password",
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
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_async_session),
) -> User:
    """
    Register a new user account.

    **Email Requirements:**
    - Must be a valid email format
    - Must be unique (not already registered)

    **Password Requirements:**
    - Minimum 8 characters
    - Passwords are hashed using bcrypt before storage

    **Request Example:**
    ```json
    {
        "email": "user@example.com",
        "password": "securePassword123"
    }
    ```

    After registration, use the `/api/v1/auth/token` endpoint to obtain
    an access token for authenticated requests.
    """
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


@router.post(
    "/token",
    response_model=Token,
    summary="Authenticate and get access token",
    responses={
        200: {
            "description": "Authentication successful - returns JWT access token",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "bearer",
                    }
                }
            },
        },
        401: {
            "description": "Invalid credentials",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid email or password"}
                }
            },
            "headers": {
                "WWW-Authenticate": {
                    "description": "Bearer authentication required",
                    "schema": {"type": "string"},
                }
            },
        },
    },
)
async def login(
    user_data: UserLogin,
    db: AsyncSession = Depends(get_async_session),
) -> Token:
    """
    Authenticate user credentials and return a JWT access token.

    **Request Example:**
    ```json
    {
        "email": "user@example.com",
        "password": "securePassword123"
    }
    ```

    **Token Details:**
    - Type: JWT (JSON Web Token)
    - Algorithm: HS256
    - Expiration: Configurable via ACCESS_TOKEN_EXPIRE_MINUTES (default: 60 minutes)
    - Payload contains: `user_id`, `exp` (expiration timestamp)

    **Using the Token:**
    Include the token in the `Authorization` header for all authenticated requests:
    ```
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    ```

    **Token Refresh Strategy:**
    Tokens are not automatically refreshed. When a token expires:
    1. Client receives 401 Unauthorized response
    2. Client should re-authenticate using this endpoint
    3. Store the new token and retry the failed request

    **Security Notes:**
    - Tokens should be stored securely (e.g., httpOnly cookies or secure storage)
    - Never expose tokens in URLs or logs
    - Implement token refresh before expiration for seamless UX
    """
    # Find user by email
    result = await db.execute(select(User).where(User.email == user_data.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token({"user_id": user.id})
    return Token(access_token=access_token)


@router.post(
    "/password-reset",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request password reset",
    responses={
        202: {
            "description": "Request accepted - email sent if account exists",
            "content": {
                "application/json": {
                    "example": {
                        "message": "If an account with that email exists, a password reset link has been sent."
                    }
                }
            },
        },
    },
)
async def request_password_reset(
    request_data: PasswordResetRequest,
    db: AsyncSession = Depends(get_async_session),
) -> dict[str, str]:
    """
    Request a password reset token to be sent via email.

    **Security Note:**
    This endpoint always returns a success message regardless of whether
    the email exists in the system. This prevents email enumeration attacks
    where an attacker could determine which emails are registered.

    **Request Example:**
    ```json
    {
        "email": "user@example.com"
    }
    ```

    **Password Reset Flow:**
    1. User submits email to this endpoint
    2. If email exists, a reset token is generated (expires in 1 hour)
    3. Reset link is sent to the email (in production, via email service)
    4. User clicks link and is directed to reset form
    5. User submits new password with token to `/password-reset/confirm`

    **Token Details:**
    - Cryptographically secure random token
    - Expires after 1 hour
    - Single-use (deleted after successful password reset)
    """
    # Find user by email
    result = await db.execute(select(User).where(User.email == request_data.email))
    user = result.scalar_one_or_none()

    if user:
        # Generate reset token
        token = generate_password_reset_token()
        expires_at = datetime.utcnow() + timedelta(hours=1)

        # Store token (in production, store in Redis/database and send via email)
        _password_reset_tokens[token] = {
            "user_id": user.id,
            "expires_at": expires_at,
        }

        # TODO: Send email with reset link
        # In production, integrate with email service (SendGrid, SES, etc.)
        # Example: await send_password_reset_email(user.email, token)

    # Always return success to prevent email enumeration
    return {"message": "If an account with that email exists, a password reset link has been sent."}


@router.post(
    "/password-reset/confirm",
    status_code=status.HTTP_200_OK,
    summary="Confirm password reset",
    responses={
        200: {
            "description": "Password reset successful",
            "content": {
                "application/json": {
                    "example": {"message": "Password has been reset successfully."}
                }
            },
        },
        400: {
            "description": "Invalid or expired reset token",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or expired reset token"}
                }
            },
        },
        422: {
            "description": "Validation error - password too weak",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "new_password"],
                                "msg": "Password must be at least 8 characters",
                                "type": "value_error",
                            }
                        ]
                    }
                }
            },
        },
    },
)
async def confirm_password_reset(
    confirm_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_async_session),
) -> dict[str, str]:
    """
    Confirm password reset using the token and set a new password.

    **Request Example:**
    ```json
    {
        "token": "abc123def456...",
        "new_password": "newSecurePassword123"
    }
    ```

    **Token Validation:**
    - Token must exist and match a pending reset request
    - Token must not be expired (1 hour lifetime)
    - Token is deleted after successful use (single-use)

    **Password Requirements:**
    - Minimum 8 characters
    - Password is hashed using bcrypt before storage

    **After Success:**
    - User can immediately login with the new password
    - Any existing sessions remain valid (tokens are not invalidated)
    - A confirmation email may be sent (in production)
    """
    # Validate token
    token_data = _password_reset_tokens.get(confirm_data.token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Check if token is expired
    if datetime.utcnow() > token_data["expires_at"]:
        # Clean up expired token
        del _password_reset_tokens[confirm_data.token]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Find user
    result = await db.execute(select(User).where(User.id == token_data["user_id"]))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Update password
    user.hashed_password = get_password_hash(confirm_data.new_password)
    await db.flush()

    # Remove used token
    del _password_reset_tokens[confirm_data.token]

    return {"message": "Password has been reset successfully."}
