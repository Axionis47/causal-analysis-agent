"""Database configuration with async SQLAlchemy engine and session management."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Create async engine with connection pooling
# Pool size: min=5, max=20 as per tech spec
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,  # Default: 5
    max_overflow=settings.DATABASE_MAX_OVERFLOW,  # Default: 10 (total max = 15)
    pool_pre_ping=True,  # Verify connections are alive before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=settings.DEBUG,  # Log SQL statements in debug mode
)

# Create async session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Track the event loop the main engine was created in
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_main_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the event loop the main engine was created in."""
    global _main_loop
    return _main_loop


def _set_main_loop() -> None:
    """Set the main event loop (called during app startup)."""
    global _main_loop
    try:
        _main_loop = asyncio.get_running_loop()
    except RuntimeError:
        _main_loop = None


def _is_same_loop() -> bool:
    """Check if we're running in the same event loop as the main engine."""
    try:
        current_loop = asyncio.get_running_loop()
        return _main_loop is not None and current_loop is _main_loop
    except RuntimeError:
        return False


def _create_task_engine() -> AsyncEngine:
    """Create a fresh engine for Celery tasks."""
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=2,  # Smaller pool for tasks
        max_overflow=3,
        pool_pre_ping=True,
        pool_recycle=300,  # Shorter recycle for tasks
        echo=settings.DEBUG,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.

    Yields an async session and ensures it's closed after use.
    Use with FastAPI's Depends:

        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_session)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for getting async database sessions.

    Automatically detects if running in a Celery task (different event loop)
    and creates a fresh engine if needed.

    Use for manual session management:

        async with get_session_context() as session:
            result = await session.execute(query)
    """
    # Check if we're in the same event loop as the main engine
    if _is_same_loop():
        # Use the shared session factory
        async with async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    else:
        # Create a fresh engine for this task's event loop (Celery worker)
        task_engine = _create_task_engine()
        task_session_factory = async_sessionmaker(
            bind=task_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        async with task_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
                await task_engine.dispose()


@asynccontextmanager
async def get_task_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for getting async database sessions in Celery tasks.

    Creates a fresh engine for each task to avoid event loop issues
    when running async code in Celery workers.

        async with get_task_session_context() as session:
            result = await session.execute(query)
    """
    # Create a fresh engine for this task's event loop
    task_engine = _create_task_engine()
    task_session_factory = async_sessionmaker(
        bind=task_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    async with task_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            await task_engine.dispose()


async def init_db() -> None:
    """
    Initialize database by creating all tables.

    Note: In production, use Alembic migrations instead.
    This is primarily for development and testing.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connection pool."""
    await engine.dispose()

