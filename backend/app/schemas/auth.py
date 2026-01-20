"""Pydantic schemas for authentication."""

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.schemas.validation import (
    sanitize_display_name,
    validate_password_strength,
)


class Token(BaseModel):
    """Response schema for access tokens."""

    access_token: str
    token_type: str = Field(default="bearer")


class TokenData(BaseModel):
    """Schema for decoded token payload."""

    user_id: uuid.UUID | None = None


class UserRegister(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(min_length=12)
    display_name: str | None = Field(default=None, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        """Additional email validation beyond EmailStr."""
        # Ensure email is not too long
        if len(value) > 254:
            raise ValueError("Email address is too long")
        # Basic format check (EmailStr handles most validation)
        return value.lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """Validate password strength."""
        return validate_password_strength(value)

    @field_validator("display_name")
    @classmethod
    def sanitize_name(cls, value: str | None) -> str | None:
        """Sanitize display name to prevent XSS."""
        if value is None:
            return None
        return sanitize_display_name(value)


class UserLogin(BaseModel):
    """Schema for user login."""

    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """Normalize email for consistent lookup."""
        return value.lower().strip()


class UserResponse(BaseModel):
    """Schema for user response."""

    id: uuid.UUID
    email: str
    display_name: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=12)
    display_name: str | None = Field(default=None, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str | None) -> str | None:
        """Additional email validation beyond EmailStr."""
        if value is None:
            return None
        if len(value) > 254:
            raise ValueError("Email address is too long")
        return value.lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str | None) -> str | None:
        """Validate password strength."""
        if value is None:
            return None
        return validate_password_strength(value)

    @field_validator("display_name")
    @classmethod
    def sanitize_name(cls, value: str | None) -> str | None:
        """Sanitize display name to prevent XSS."""
        if value is None:
            return None
        return sanitize_display_name(value)


class PasswordResetRequest(BaseModel):
    """Schema for requesting a password reset."""

    email: EmailStr

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        """Normalize email for consistent lookup."""
        return value.lower().strip()


class PasswordResetConfirm(BaseModel):
    """Schema for confirming a password reset."""

    token: str = Field(min_length=1, max_length=500)
    new_password: str = Field(min_length=12)

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        """Validate password strength."""
        return validate_password_strength(value)
