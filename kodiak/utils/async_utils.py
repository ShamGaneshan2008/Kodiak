import asyncio
from typing import Any, Coroutine, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


async def run_concurrently(*coros: Coroutine[Any, Any, T]) -> list[T]:
    """Execute multiple coroutines concurrently and return their results."""
    if not coros:
        return []
    return await asyncio.gather(*coros)


async def gather_with_limit(
    coros: Sequence[Coroutine[Any, Any, T]], limit: int = 10
) -> list[T | BaseException]:
    """Execute coroutines concurrently with a concurrency limit."""
    if limit <= 0:
        raise ValueError("Limit must be greater than 0")
    semaphore = asyncio.Semaphore(limit)
    results: list[T | BaseException] = [None] * len(coros)  # type: ignore[list-item]

    async def _bounded_task(index: int, coro: Coroutine[Any, Any, T]) -> None:
        async with semaphore:
            results[index] = await coro

    tasks = [
        _bounded_task(i, c) for i, c in enumerate(coros)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def wait_with_timeout(
    coro: Coroutine[Any, Any, T], timeout: float
) -> T:
    """Wait for a coroutine to complete, raising TimeoutError if it exceeds the limit."""
    return await asyncio.wait_for(coro, timeout=timeout)


async def cancel_tasks(tasks: Sequence[asyncio.Task[Any]]) -> None:
    """Safely cancel a sequence of asyncio tasks."""
    for task in tasks:
        if not task.done():
            task.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, Exception):
            logger.warning("error_during_task_cancellation", error=str(result))