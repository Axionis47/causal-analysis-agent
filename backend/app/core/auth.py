"""JWT authentication utilities for the API."""

import secrets
import uuid
from datetime import datetime, timedelta

import sentry_sdk
from fastapi import Depends, HTTPException, Request, status
from fastapi import Query
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.logging import bind_contextvars
from app.schemas.auth import TokenData

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def generate_password_reset_token() -> str:
    """Generate a secure random token for password reset."""
    return secrets.token_urlsafe(32)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


def create_access_token(data: dict) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    user_id = to_encode.get("user_id")
    if user_id is not None:
        to_encode["user_id"] = str(user_id)
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT access token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id_raw = payload.get("user_id")
        if user_id_raw is None:
            raise credentials_exception
        user_id = uuid.UUID(user_id_raw)
    except (JWTError, ValueError, TypeError):
        raise credentials_exception
    return TokenData(user_id=user_id)


def get_current_user(request: Request, token: str = Depends(oauth2_scheme)) -> uuid.UUID:
    """Extract the current user's ID from the JWT token."""
    token_data = decode_access_token(token)
    if token_data.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.user_id = token_data.user_id
    bind_contextvars(user_id=str(token_data.user_id))
    sentry_sdk.set_user({"id": str(token_data.user_id)})
    return token_data.user_id


def get_current_user_optional(
    request: Request,
    token: str | None = Query(default=None),
    header_token: str | None = Depends(oauth2_scheme_optional),
) -> uuid.UUID:
    """Allow JWT token via query or Authorization header (for SSE)."""
    if token:
        user_id = decode_access_token(token).user_id
        request.state.user_id = user_id
        bind_contextvars(user_id=str(user_id))
        sentry_sdk.set_user({"id": str(user_id)})
        return user_id
    if header_token:
        user_id = decode_access_token(header_token).user_id
        request.state.user_id = user_id
        bind_contextvars(user_id=str(user_id))
        sentry_sdk.set_user({"id": str(user_id)})
        return user_id
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
