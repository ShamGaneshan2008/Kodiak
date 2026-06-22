from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_async(fn: Callable[[], Awaitable[T]], attempts: int = 3) -> T:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error
