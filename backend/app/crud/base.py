"""Base CRUD class with common database operations."""

import uuid
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)


class CRUDBase(Generic[ModelType]):
    """
    Base CRUD class with common database operations.

    Provides generic create, read, update, delete operations
    for SQLAlchemy models.
    """

    def __init__(self, model: type[ModelType]) -> None:
        """Initialize CRUD with model class."""
        self.model = model

    async def get(self, db: AsyncSession, id: uuid.UUID) -> ModelType | None:
        """Get a single record by ID."""
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelType]:
        """Get multiple records with pagination."""
        result = await db.execute(
            select(self.model).offset(skip).limit(limit).order_by(self.model.created_at.desc())
        )
        return list(result.scalars().all())

    async def create(self, db: AsyncSession, *, obj_in: dict[str, Any]) -> ModelType:
        """Create a new record."""
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: dict[str, Any],
    ) -> ModelType:
        """Update an existing record."""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def delete(self, db: AsyncSession, *, id: uuid.UUID) -> ModelType | None:
        """Delete a record by ID."""
        obj = await self.get(db, id)
        if obj:
            await db.delete(obj)
            await db.commit()
        return obj

    async def exists(self, db: AsyncSession, id: uuid.UUID) -> bool:
        """Check if a record exists by ID."""
        result = await db.execute(
            select(self.model.id).where(self.model.id == id)
        )
        return result.scalar_one_or_none() is not None

    async def count(self, db: AsyncSession) -> int:
        """Count total records."""
        from sqlalchemy import func

        result = await db.execute(select(func.count()).select_from(self.model))
        return result.scalar_one()

