"""CRUD operations and helpers for AuditLog model."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.audit_log import AuditAction, AuditLog


class CRUDAuditLog(CRUDBase[AuditLog]):
    """CRUD operations for AuditLog model."""

    async def log_action(
        self,
        db: AsyncSession,
        *,
        entity_type: str,
        entity_id: uuid.UUID,
        action: AuditAction,
        user_id: uuid.UUID | None = None,
        changes: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        description: str | None = None,
    ) -> AuditLog:
        """
        Create an audit log entry.

        Args:
            db: Database session
            entity_type: Type of entity being audited (e.g., 'analysis', 'version')
            entity_id: ID of the entity
            action: Type of action performed
            user_id: User who performed the action
            changes: Before/after values for the change
            ip_address: IP address of the request
            user_agent: User agent string from the request
            request_id: Request ID for correlation
            description: Human-readable description

        Returns:
            Created AuditLog entry
        """
        audit_log = AuditLog(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            user_id=user_id,
            changes=changes or {},
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            description=description,
        )
        db.add(audit_log)
        await db.flush()
        return audit_log

    async def get_by_entity(
        self,
        db: AsyncSession,
        entity_type: str,
        entity_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs for a specific entity."""
        result = await db.execute(
            select(AuditLog)
            .where(
                AuditLog.entity_type == entity_type,
                AuditLog.entity_id == entity_id,
            )
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs for a specific user."""
        result = await db.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_request_id(
        self,
        db: AsyncSession,
        request_id: str,
    ) -> list[AuditLog]:
        """Get all audit logs for a specific request."""
        result = await db.execute(
            select(AuditLog)
            .where(AuditLog.request_id == request_id)
            .order_by(AuditLog.created_at.asc())
        )
        return list(result.scalars().all())


# Singleton instance
audit_log_crud = CRUDAuditLog(AuditLog)


async def log_analysis_action(
    db: AsyncSession,
    *,
    analysis_id: uuid.UUID,
    action: AuditAction,
    user_id: uuid.UUID | None = None,
    changes: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    description: str | None = None,
) -> AuditLog:
    """
    Convenience helper to log an analysis-related action.

    Args:
        db: Database session
        analysis_id: ID of the analysis
        action: Type of action performed
        user_id: User who performed the action
        changes: Before/after values
        ip_address: IP address of the request
        user_agent: User agent string
        request_id: Request ID for correlation
        description: Human-readable description

    Returns:
        Created AuditLog entry
    """
    return await audit_log_crud.log_action(
        db,
        entity_type="analysis",
        entity_id=analysis_id,
        action=action,
        user_id=user_id,
        changes=changes,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        description=description,
    )


async def log_version_action(
    db: AsyncSession,
    *,
    version_id: uuid.UUID,
    analysis_id: uuid.UUID,
    action: AuditAction,
    user_id: uuid.UUID | None = None,
    changes: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    description: str | None = None,
) -> AuditLog:
    """
    Convenience helper to log a version-related action.

    Args:
        db: Database session
        version_id: ID of the version
        analysis_id: ID of the parent analysis (included in changes)
        action: Type of action performed
        user_id: User who performed the action
        changes: Before/after values
        ip_address: IP address of the request
        user_agent: User agent string
        request_id: Request ID for correlation
        description: Human-readable description

    Returns:
        Created AuditLog entry
    """
    changes_with_analysis = changes.copy() if changes else {}
    changes_with_analysis["analysis_id"] = str(analysis_id)

    return await audit_log_crud.log_action(
        db,
        entity_type="version",
        entity_id=version_id,
        action=action,
        user_id=user_id,
        changes=changes_with_analysis,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        description=description,
    )
