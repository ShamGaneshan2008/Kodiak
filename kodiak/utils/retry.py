import asyncio
import functools
import time
from typing import Any, Callable, TypeVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=1.0, ge=0.0)
    max_delay: float = Field(default=30.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)


def retry_async(config: RetryConfig | None = None) -> Callable[[F], F]:
    """Decorator to retry an async function with exponential backoff."""
    cfg = config or RetryConfig()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = cfg.base_delay
            last_exception: BaseException | None = None

            for attempt in range(cfg.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == cfg.max_attempts - 1:
                        logger.error(
                            "async_retry_exhausted",
                            func=func.__name__,
                            attempts=cfg.max_attempts,
                        )
                        raise

                    logger.warning(
                        "async_retry_attempt",
                        func=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * cfg.backoff_factor, cfg.max_delay)

            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_sync(config: RetryConfig | None = None) -> Callable[[F], F]:
    """Decorator to retry a synchronous function with exponential backoff."""
    cfg = config or RetryConfig()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = cfg.base_delay
            last_exception: BaseException | None = None

            for attempt in range(cfg.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == cfg.max_attempts - 1:
                        logger.error(
                            "sync_retry_exhausted",
                            func=func.__name__,
                            attempts=cfg.max_attempts,
                        )
                        raise

                    logger.warning(
                        "sync_retry_attempt",
                        func=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    time.sleep(delay)
                    delay = min(delay * cfg.backoff_factor, cfg.max_delay)

            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator