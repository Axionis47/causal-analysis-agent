"""Credential storage and encryption helpers with key rotation support."""

from __future__ import annotations

import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.user_credential import CredentialProvider, UserCredential

logger = get_logger(__name__)


def _get_encryption_key(version: str | None = None) -> str:
    """Get the encryption key for a specific version.

    Args:
        version: Key version (e.g., "v1", "v2"). If None, uses current version.

    Returns:
        The encryption key string for that version.

    Raises:
        ValueError: If the key version is not configured.
    """
    if version is None:
        version = settings.CREDENTIALS_ENCRYPTION_KEY_VERSION

    # Try versioned keys first
    versioned_key_attr = f"CREDENTIALS_ENCRYPTION_KEY_{version.upper()}"
    versioned_key = getattr(settings, versioned_key_attr, "")

    if versioned_key:
        return versioned_key

    # Fall back to default key for v1
    if version == "v1":
        return settings.CREDENTIALS_ENCRYPTION_KEY

    raise ValueError(f"Encryption key version '{version}' is not configured")


def _fernet(key_version: str | None = None) -> Fernet:
    """Create a Fernet instance for the specified key version.

    Args:
        key_version: Key version to use. If None, uses current version.

    Returns:
        Fernet instance configured with the appropriate key.
    """
    key_string = _get_encryption_key(key_version)
    digest = hashlib.sha256(key_string.encode()).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_value(value: str, key_version: str | None = None) -> bytes:
    """Encrypt a string value using the specified key version.

    Args:
        value: The plaintext string to encrypt.
        key_version: Key version to use. If None, uses current version.

    Returns:
        Encrypted bytes.
    """
    return _fernet(key_version).encrypt(value.encode())


def decrypt_value(value: bytes, key_version: str | None = None) -> str:
    """Decrypt an encrypted value using the specified key version.

    Args:
        value: The encrypted bytes to decrypt.
        key_version: Key version to use. If None, uses current version.

    Returns:
        Decrypted plaintext string.

    Raises:
        InvalidToken: If decryption fails (wrong key or corrupted data).
    """
    return _fernet(key_version).decrypt(value).decode()


def get_current_key_version() -> str:
    """Get the current encryption key version.

    Returns:
        Current key version string (e.g., "v1").
    """
    return settings.CREDENTIALS_ENCRYPTION_KEY_VERSION


async def get_kaggle_credentials(
    session: AsyncSession | None, user_id: Optional[str]
) -> tuple[str, str] | None:
    """Retrieve Kaggle credentials, supporting key rotation.

    Args:
        session: Database session.
        user_id: User ID to filter credentials by.

    Returns:
        Tuple of (username, api_key) or None if not found.
    """
    if session is not None:
        result = await session.execute(
            select(UserCredential).where(
                UserCredential.provider == CredentialProvider.KAGGLE
            )
        )
        credential = result.scalar_one_or_none()
        if credential is not None:
            # Use the key version stored with the credential
            key_version = credential.encryption_key_version
            try:
                decrypted_key = decrypt_value(
                    credential.encrypted_api_key, key_version
                )
                return credential.username, decrypted_key
            except InvalidToken:
                logger.error(
                    "Failed to decrypt credential",
                    credential_id=str(credential.id),
                    key_version=key_version,
                )
                raise

    if settings.KAGGLE_USERNAME and settings.KAGGLE_KEY:
        return settings.KAGGLE_USERNAME, settings.KAGGLE_KEY
    return None


async def store_kaggle_credentials(
    session: AsyncSession,
    username: str,
    api_key: str,
    user_id: Optional[str] = None,
) -> UserCredential:
    """Store Kaggle credentials with the current encryption key version.

    Args:
        session: Database session.
        username: Kaggle username.
        api_key: Kaggle API key.
        user_id: User ID to associate credentials with.

    Returns:
        The created UserCredential instance.
    """
    current_version = get_current_key_version()
    credential = UserCredential(
        user_id=user_id,
        provider=CredentialProvider.KAGGLE,
        username=username,
        encrypted_api_key=encrypt_value(api_key, current_version),
        encryption_key_version=current_version,
    )
    session.add(credential)
    await session.flush()
    return credential


async def rotate_credential_encryption(
    session: AsyncSession,
    old_version: str,
    new_version: str,
    batch_size: int = 100,
) -> int:
    """Rotate encryption keys for all credentials.

    This function re-encrypts all credentials from the old key version
    to the new key version. Should be run as part of a key rotation process.

    Args:
        session: Database session.
        old_version: The current key version to migrate from.
        new_version: The new key version to migrate to.
        batch_size: Number of credentials to process per batch.

    Returns:
        Number of credentials successfully rotated.

    Note:
        This should be run within a transaction to ensure atomicity.
        If any credential fails to rotate, the entire batch is rolled back.
    """
    rotated_count = 0

    # Get all credentials with the old key version
    result = await session.execute(
        select(UserCredential).where(
            UserCredential.encryption_key_version == old_version
        )
    )
    credentials = result.scalars().all()

    logger.info(
        "Starting credential key rotation",
        old_version=old_version,
        new_version=new_version,
        credential_count=len(credentials),
    )

    for credential in credentials:
        try:
            # Decrypt with old key
            decrypted_value = decrypt_value(
                credential.encrypted_api_key, old_version
            )

            # Re-encrypt with new key
            new_encrypted = encrypt_value(decrypted_value, new_version)

            # Update credential
            credential.encrypted_api_key = new_encrypted
            credential.encryption_key_version = new_version

            rotated_count += 1

            if rotated_count % batch_size == 0:
                await session.flush()
                logger.info(
                    "Credential rotation progress",
                    rotated=rotated_count,
                    total=len(credentials),
                )

        except Exception as e:
            logger.error(
                "Failed to rotate credential",
                credential_id=str(credential.id),
                error=str(e),
            )
            raise

    # Final flush
    await session.flush()

    logger.info(
        "Credential key rotation completed",
        rotated_count=rotated_count,
    )

    return rotated_count


async def verify_credential_decryption(
    session: AsyncSession,
    credential_id: str,
) -> bool:
    """Verify that a credential can be decrypted successfully.

    Args:
        session: Database session.
        credential_id: ID of the credential to verify.

    Returns:
        True if decryption succeeds, False otherwise.
    """
    result = await session.execute(
        select(UserCredential).where(UserCredential.id == credential_id)
    )
    credential = result.scalar_one_or_none()

    if credential is None:
        return False

    try:
        decrypt_value(credential.encrypted_api_key, credential.encryption_key_version)
        return True
    except InvalidToken:
        return False
