from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kodiak.config.settings import get_settings

settings = get_settings()
engine = create_async_engine(settings.database_url, echo=settings.is_test)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


def async_session() -> AsyncIterator[AsyncSession]:
    return SessionLocal()
