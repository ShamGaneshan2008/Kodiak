from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from kodiak.config.settings import settings


# ================= DB ENGINE =================
engine = create_async_engine(
    settings.database_url_async,
    echo=settings.db_echo,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ================= REDIS (FIXED LAZY LOADING) =================
_redis_client = None


def get_redis_client():
    global _redis_client

    if _redis_client is None:
        import redis.asyncio as redis  # lazy import prevents startup crash
        _redis_client = redis.from_url(str(settings.REDIS_URL))

    return _redis_client


async def get_redis():
    return get_redis_client()