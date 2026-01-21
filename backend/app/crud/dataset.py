"""CRUD operations for Dataset model."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.crud.base import CRUDBase
from app.models.dataset import Dataset


class CRUDDataset(CRUDBase[Dataset]):
    """CRUD operations for Dataset model."""

    async def get_with_understanding(
        self, db: AsyncSession, id: uuid.UUID
    ) -> Dataset | None:
        """Get dataset with data understanding loaded."""
        result = await db.execute(
            select(Dataset)
            .where(Dataset.id == id)
            .options(selectinload(Dataset.data_understanding))
        )
        return result.scalar_one_or_none()

    async def get_by_analysis(
        self, db: AsyncSession, analysis_id: uuid.UUID
    ) -> list[Dataset]:
        """Get all datasets for an analysis."""
        result = await db.execute(
            select(Dataset)
            .where(Dataset.analysis_id == analysis_id)
            .order_by(Dataset.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_cache_key(
        self, db: AsyncSession, cache_key: str
    ) -> Dataset | None:
        """Get dataset by cache key (for cache lookup).

        Returns the most recent dataset with a valid gcs_path if available,
        otherwise returns the most recent dataset with this cache_key.
        Handles multiple datasets with the same cache_key gracefully.
        """
        # First try to find a dataset with a valid gcs_path
        result = await db.execute(
            select(Dataset)
            .where(Dataset.cache_key == cache_key)
            .where(Dataset.gcs_path.isnot(None))
            .where(Dataset.gcs_path != "")
            .order_by(Dataset.created_at.desc())
        )
        dataset = result.scalars().first()
        if dataset:
            return dataset

        # Fall back to any dataset with this cache_key
        result = await db.execute(
            select(Dataset)
            .where(Dataset.cache_key == cache_key)
            .order_by(Dataset.created_at.desc())
        )
        return result.scalars().first()

    async def get_cached(
        self, db: AsyncSession, cache_key: str
    ) -> Dataset | None:
        """Get cached dataset if not expired and has valid gcs_path."""
        dataset = await self.get_by_cache_key(db, cache_key)
        if dataset and dataset.cache_expires_at and dataset.gcs_path:
            if dataset.cache_expires_at > datetime.now(timezone.utc):
                return dataset
        return None

    async def get_by_kaggle_url(
        self, db: AsyncSession, kaggle_url: str
    ) -> list[Dataset]:
        """Get datasets by Kaggle URL."""
        result = await db.execute(
            select(Dataset)
            .where(Dataset.kaggle_url == kaggle_url)
            .order_by(Dataset.created_at.desc())
        )
        return list(result.scalars().all())


# Singleton instance
dataset_crud = CRUDDataset(Dataset)

