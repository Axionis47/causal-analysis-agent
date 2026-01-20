"""CRUD operations for Analysis model."""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.crud.base import CRUDBase
from app.models.analysis import Analysis, AnalysisStatus
from app.models.audit_log import AuditAction


class CRUDAnalysis(CRUDBase[Analysis]):
    """CRUD operations for Analysis model."""

    async def get_with_relations(
        self, db: AsyncSession, id: uuid.UUID
    ) -> Analysis | None:
        """Get analysis with all related entities loaded."""
        result = await db.execute(
            select(Analysis)
            .where(Analysis.id == id)
            .options(
                selectinload(Analysis.datasets),
                selectinload(Analysis.stages),
                selectinload(Analysis.causal_graphs),
                selectinload(Analysis.treatment_effects),
                selectinload(Analysis.validation_results),
                selectinload(Analysis.agent_interactions),
                selectinload(Analysis.reports),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Analysis]:
        """Get analyses for a specific user."""
        result = await db.execute(
            select(Analysis)
            .where(Analysis.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(Analysis.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_status(
        self,
        db: AsyncSession,
        status: AnalysisStatus,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Analysis]:
        """Get analyses by status."""
        result = await db.execute(
            select(Analysis)
            .where(Analysis.status == status)
            .offset(skip)
            .limit(limit)
            .order_by(Analysis.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_kaggle_url(
        self, db: AsyncSession, kaggle_url: str
    ) -> list[Analysis]:
        """Get analyses for a specific Kaggle URL."""
        result = await db.execute(
            select(Analysis)
            .where(Analysis.kaggle_url == kaggle_url)
            .order_by(Analysis.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        db: AsyncSession,
        *,
        db_obj: Analysis,
        status: AnalysisStatus,
        error_message: str | None = None,
    ) -> Analysis:
        """Update analysis status."""
        update_data: dict[str, Any] = {"status": status}
        if error_message is not None:
            update_data["error_message"] = error_message
        return await self.update(db, db_obj=db_obj, obj_in=update_data)

    async def get_pending(self, db: AsyncSession, limit: int = 10) -> list[Analysis]:
        """Get pending analyses for processing."""
        return await self.get_by_status(db, AnalysisStatus.PENDING, limit=limit)

    async def get_running(self, db: AsyncSession) -> list[Analysis]:
        """Get currently running analyses."""
        return await self.get_by_status(db, AnalysisStatus.RUNNING)

    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: Analysis,
        obj_in: dict[str, Any],
        changed_by: uuid.UUID | None = None,
        change_summary: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
    ) -> Analysis:
        """
        Update an analysis. If config is being modified, automatically creates a version.

        All operations (version creation, analysis update, audit log) are performed
        in a single transaction to ensure atomicity.

        Args:
            db: Database session
            db_obj: Analysis object to update
            obj_in: Update data
            changed_by: User ID who made the change (used for versioning)
            change_summary: Optional description of the change (used for versioning)
            ip_address: IP address of the request (for audit logging)
            user_agent: User agent string (for audit logging)
            request_id: Request ID for correlation (for audit logging)

        Returns:
            Updated Analysis object
        """
        from app.crud.audit_log import log_analysis_action

        old_config = db_obj.config.copy() if db_obj.config else {}
        config_changed = False

        # Check if config is being updated and if it actually changed
        if "config" in obj_in:
            new_config = obj_in["config"]

            if self._is_config_changed(old_config, new_config):
                config_changed = True
                from app.crud.analysis_version import analysis_version_crud

                # Create a new version with the new config (flush, don't commit)
                await analysis_version_crud.create_version(
                    db,
                    analysis_id=db_obj.id,
                    config=new_config,
                    changed_by=changed_by or db_obj.user_id,
                    change_summary=change_summary,
                    is_manual_snapshot=False,
                    auto_commit=False,
                )

        # Apply the updates to the model
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        db.add(db_obj)

        # Log the update action
        changes = {"fields_updated": list(obj_in.keys())}
        if config_changed:
            changes["config_before"] = old_config
            changes["config_after"] = obj_in.get("config", {})

        await log_analysis_action(
            db,
            analysis_id=db_obj.id,
            action=AuditAction.UPDATE,
            user_id=changed_by or db_obj.user_id,
            changes=changes,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            description=change_summary or f"Updated analysis fields: {', '.join(obj_in.keys())}",
        )

        # Commit version creation, analysis update, and audit log together
        await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def update_with_versioning(
        self,
        db: AsyncSession,
        *,
        db_obj: Analysis,
        obj_in: dict[str, Any],
        changed_by: uuid.UUID | None = None,
        change_summary: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
    ) -> Analysis:
        """
        Update an analysis and automatically create a version if config changes.

        This method is now a thin wrapper around update() which handles versioning
        automatically. Kept for backward compatibility with explicit versioning calls.

        Args:
            db: Database session
            db_obj: Analysis object to update
            obj_in: Update data
            changed_by: User ID who made the change
            change_summary: Optional description of the change
            ip_address: IP address of the request (for audit logging)
            user_agent: User agent string (for audit logging)
            request_id: Request ID for correlation (for audit logging)

        Returns:
            Updated Analysis object
        """
        return await self.update(
            db,
            db_obj=db_obj,
            obj_in=obj_in,
            changed_by=changed_by,
            change_summary=change_summary,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
        )

    async def create_manual_snapshot(
        self,
        db: AsyncSession,
        *,
        analysis_id: uuid.UUID,
        changed_by: uuid.UUID | None = None,
        change_summary: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
    ) -> "AnalysisVersion":
        """
        Create a manual version snapshot without modifying the analysis.

        Args:
            db: Database session
            analysis_id: Analysis ID to snapshot
            changed_by: User ID creating the snapshot
            change_summary: Description of the snapshot
            ip_address: IP address of the request (for audit logging)
            user_agent: User agent string (for audit logging)
            request_id: Request ID for correlation (for audit logging)

        Returns:
            The created AnalysisVersion
        """
        from app.crud.analysis_version import analysis_version_crud
        from app.crud.audit_log import log_analysis_action
        from app.models.analysis_version import AnalysisVersion

        analysis = await self.get(db, analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found")

        version = await analysis_version_crud.create_version(
            db,
            analysis_id=analysis_id,
            config=analysis.config or {},
            changed_by=changed_by,
            change_summary=change_summary or "Manual snapshot",
            is_manual_snapshot=True,
            auto_commit=False,  # Don't commit yet - wait for audit log
        )

        # Log the snapshot action
        await log_analysis_action(
            db,
            analysis_id=analysis_id,
            action=AuditAction.SNAPSHOT,
            user_id=changed_by,
            changes={
                "version_number": version.version_number,
                "config_hash": version.config_hash,
            },
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            description=change_summary or "Manual snapshot created",
        )

        # Commit both version and audit log together
        await db.commit()
        await db.refresh(version)

        return version

    async def revert_to_version(
        self,
        db: AsyncSession,
        *,
        analysis_id: uuid.UUID,
        version_number: int,
        changed_by: uuid.UUID | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
    ) -> Analysis:
        """
        Revert an analysis to a specific version.

        This loads the config from the specified version and updates the analysis,
        which triggers the creation of a new version (maintaining history).

        Args:
            db: Database session
            analysis_id: Analysis ID to revert
            version_number: Version number to revert to
            changed_by: User ID performing the revert
            ip_address: IP address of the request (for audit logging)
            user_agent: User agent string (for audit logging)
            request_id: Request ID for correlation (for audit logging)

        Returns:
            Updated Analysis object
        """
        from app.crud.analysis_version import analysis_version_crud
        from app.crud.audit_log import log_analysis_action

        analysis = await self.get(db, analysis_id)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_id} not found")

        old_config = analysis.config.copy() if analysis.config else {}

        target_version = await analysis_version_crud.get_version(
            db, analysis_id, version_number
        )
        if target_version is None:
            raise ValueError(f"Version {version_number} not found for analysis {analysis_id}")

        # Create a new version with the reverted config (don't commit yet)
        new_version = await analysis_version_crud.create_version(
            db,
            analysis_id=analysis_id,
            config=target_version.config,
            changed_by=changed_by,
            change_summary=f"Reverted to version {version_number}",
            is_manual_snapshot=False,
            parent_version_id=target_version.id,
            auto_commit=False,
        )

        # Update the analysis config directly (without using update() to avoid committing separately)
        analysis.config = target_version.config
        db.add(analysis)

        # Log the revert action
        await log_analysis_action(
            db,
            analysis_id=analysis_id,
            action=AuditAction.REVERT,
            user_id=changed_by,
            changes={
                "reverted_to_version": version_number,
                "new_version_number": new_version.version_number,
                "config_before": old_config,
                "config_after": target_version.config,
            },
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            description=f"Reverted to version {version_number}",
        )

        # Commit version creation, analysis update, and audit log together
        await db.commit()
        await db.refresh(analysis)

        return analysis

    def _is_config_changed(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> bool:
        """Check if config has actually changed using hash comparison."""
        old_hash = self._compute_config_hash(old_config)
        new_hash = self._compute_config_hash(new_config)
        return old_hash != new_hash

    def _compute_config_hash(self, config: dict[str, Any]) -> str:
        """Compute SHA-256 hash of config for comparison."""
        config_json = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()


# Singleton instance
analysis_crud = CRUDAnalysis(Analysis)

