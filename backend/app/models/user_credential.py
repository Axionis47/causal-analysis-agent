"""UserCredential model for storing encrypted API credentials."""

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, Index, LargeBinary, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class CredentialProvider(str, enum.Enum):
    """Credential provider type."""

    KAGGLE = "kaggle"


class UserCredential(BaseModel):
    """
    UserCredential model for storing encrypted API credentials.

    Stores encrypted credentials for external services like Kaggle.
    Uses AES-256 encryption for API keys.
    """

    __tablename__ = "user_credentials"

    # User reference (nullable for now, until user system is implemented)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # Provider information
    provider: Mapped[CredentialProvider] = mapped_column(
        Enum(CredentialProvider),
        nullable=False,
    )
    username: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )

    # Encrypted credentials
    encrypted_api_key: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False,
        comment="AES-256 encrypted API key",
    )
    encryption_key_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="v1",
        comment="Version of encryption key used",
    )

    # Validation and usage tracking
    validated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When credentials were last validated",
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When credentials were last used",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_user_credentials_user_provider", "user_id", "provider"),
        Index("ix_user_credentials_provider_username", "provider", "username"),
        Index("ix_user_credentials_key_version", "encryption_key_version"),
    )

