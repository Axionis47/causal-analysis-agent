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
        """Get dataset by cache key (for cache lookup)."""
        result = await db.execute(
            select(Dataset).where(Dataset.cache_key == cache_key)
        )
        return result.scalar_one_or_none()

    async def get_cached(
        self, db: AsyncSession, cache_key: str
    ) -> Dataset | None:
        """Get cached dataset if not expired."""
        dataset = await self.get_by_cache_key(db, cache_key)
        if dataset and dataset.cache_expires_at:
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

