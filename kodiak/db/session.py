"""
kodiak/db/session.py

Async SQLAlchemy engine + session factory.
Exposes `get_db` as a FastAPI dependency and `engine` for Alembic.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import event, pool, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kodiak.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine_kwargs: dict = {
    "echo": settings.db_echo,
    "pool_pre_ping": True,
    "pool_size": settings.db_pool_size,
    "max_overflow": settings.db_max_overflow,
    "pool_timeout": settings.db_pool_timeout,
    "pool_recycle": settings.db_pool_recycle,
}

# NullPool is used during Alembic migrations (no async runtime)
if settings.db_use_null_pool:
    _engine_kwargs["poolclass"] = pool.NullPool
    _engine_kwargs.pop("pool_size", None)
    _engine_kwargs.pop("max_overflow", None)
    _engine_kwargs.pop("pool_timeout", None)
    _engine_kwargs.pop("pool_recycle", None)

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    **_engine_kwargs,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a transactional async session.

    Usage::

        @router.get("/items")
        async def list_items(db: DbSession) -> list[ItemSchema]:
            ...

    The session is committed on success and rolled back on any exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Annotated shorthand for cleaner router signatures
DbSession = Annotated[AsyncSession, Depends(get_db)]

# ---------------------------------------------------------------------------
# Context manager variant (for workers / background tasks)
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside FastAPI's DI system.

    Usage::

        async with db_session() as session:
            result = await session.execute(select(User))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Low-level connection helper (used by Alembic env.py)
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def get_connection() -> AsyncGenerator[AsyncConnection, None]:
    """Yield a raw async connection (no ORM layer)."""
    async with engine.begin() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Health-check helper
# ---------------------------------------------------------------------------

async def check_db_health() -> bool:
    """Return True if the database is reachable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB health check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

async def dispose_engine() -> None:
    """Call on application shutdown to release all pooled connections."""
    await engine.dispose()