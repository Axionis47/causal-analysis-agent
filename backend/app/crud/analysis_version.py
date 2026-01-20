"""CRUD operations for AnalysisVersion model."""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.base import CRUDBase
from app.models.analysis_version import AnalysisVersion


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute SHA-256 hash of config for quick comparison."""
    # Sort keys to ensure consistent hashing
    config_json = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()


class CRUDAnalysisVersion(CRUDBase[AnalysisVersion]):
    """CRUD operations for AnalysisVersion model."""

    async def get_by_analysis(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AnalysisVersion]:
        """Get all versions for an analysis, ordered by version_number DESC."""
        result = await db.execute(
            select(AnalysisVersion)
            .where(AnalysisVersion.analysis_id == analysis_id)
            .order_by(AnalysisVersion.version_number.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_version(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
        version_number: int,
    ) -> AnalysisVersion | None:
        """Get a specific version by analysis ID and version number."""
        result = await db.execute(
            select(AnalysisVersion)
            .where(
                AnalysisVersion.analysis_id == analysis_id,
                AnalysisVersion.version_number == version_number,
            )
        )
        return result.scalar_one_or_none()

    async def get_latest_version(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
    ) -> AnalysisVersion | None:
        """Get the highest version number for an analysis."""
        result = await db.execute(
            select(AnalysisVersion)
            .where(AnalysisVersion.analysis_id == analysis_id)
            .order_by(AnalysisVersion.version_number.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_latest_version_number(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
    ) -> int:
        """Get the latest version number for an analysis, or 0 if none exists."""
        result = await db.execute(
            select(func.max(AnalysisVersion.version_number))
            .where(AnalysisVersion.analysis_id == analysis_id)
        )
        max_version = result.scalar_one_or_none()
        return max_version or 0

    async def create_version(
        self,
        db: AsyncSession,
        *,
        analysis_id: uuid.UUID,
        config: dict[str, Any],
        changed_by: uuid.UUID | None = None,
        change_summary: str | None = None,
        is_manual_snapshot: bool = False,
        parent_version_id: uuid.UUID | None = None,
        auto_commit: bool = False,
    ) -> AnalysisVersion:
        """
        Create a new version with auto-incremented version_number.

        Args:
            db: Database session
            analysis_id: ID of the analysis
            config: Config snapshot for this version
            changed_by: User ID who made the change
            change_summary: Description of what changed
            is_manual_snapshot: Whether this is a manual snapshot
            parent_version_id: Parent version ID (for reverts)
            auto_commit: If True, commits the transaction. If False (default),
                         only flushes to allow the caller to manage the transaction.
                         This enables atomic operations when version creation is part
                         of a larger transaction (e.g., update_with_versioning).

        Returns:
            The created AnalysisVersion
        """
        # Get the next version number
        latest_version_number = await self.get_latest_version_number(db, analysis_id)
        next_version_number = latest_version_number + 1

        # Compute config hash
        config_hash = compute_config_hash(config)

        # Create the version
        version = AnalysisVersion(
            analysis_id=analysis_id,
            version_number=next_version_number,
            config=config,
            config_hash=config_hash,
            changed_by=changed_by,
            change_summary=change_summary,
            is_manual_snapshot=is_manual_snapshot,
            parent_version_id=parent_version_id,
        )
        db.add(version)

        if auto_commit:
            # Standalone version creation - commit immediately
            await db.commit()
            await db.refresh(version)
        else:
            # Part of a larger transaction - flush to get the ID but don't commit
            # The caller is responsible for committing
            await db.flush()

        return version

    async def get_version_count(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
    ) -> int:
        """Count total versions for an analysis."""
        result = await db.execute(
            select(func.count())
            .select_from(AnalysisVersion)
            .where(AnalysisVersion.analysis_id == analysis_id)
        )
        return result.scalar_one()

    async def compare_versions(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
        version1: int,
        version2: int,
    ) -> dict[str, Any]:
        """
        Compare two versions of an analysis.

        Returns a dict with both versions' configs and computed diff.
        """
        v1 = await self.get_version(db, analysis_id, version1)
        v2 = await self.get_version(db, analysis_id, version2)

        if v1 is None or v2 is None:
            return {
                "error": "One or both versions not found",
                "version1": v1.to_dict() if v1 else None,
                "version2": v2.to_dict() if v2 else None,
            }

        return {
            "version1": v1.to_dict(),
            "version2": v2.to_dict(),
            "config1": v1.config,
            "config2": v2.config,
            "hash_match": v1.config_hash == v2.config_hash,
        }

    async def get_versions_by_hash(
        self,
        db: AsyncSession,
        analysis_id: uuid.UUID,
        config_hash: str,
    ) -> list[AnalysisVersion]:
        """Find versions with a specific config hash (for duplicate detection)."""
        result = await db.execute(
            select(AnalysisVersion)
            .where(
                AnalysisVersion.analysis_id == analysis_id,
                AnalysisVersion.config_hash == config_hash,
            )
            .order_by(AnalysisVersion.version_number.desc())
        )
        return list(result.scalars().all())


# Singleton instance
analysis_version_crud = CRUDAnalysisVersion(AnalysisVersion)
