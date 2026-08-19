"""Compatibility module for ``kodiak.orchestration.execution_result``.

The canonical ``ExecutionResult`` model -- along with its outcome enum
and ``RetryPolicy`` -- lives in ``kodiak.orchestration.execution``, the
new execution package. This module exists so that older imports of
``kodiak.orchestration.execution_result`` keep working; it re-exports the
canonical types rather than defining a competing model.

Beyond re-exporting, this module adds small, side-effect-free helpers
built *on top of* the canonical model:

* Outcome-specific factory functions (:func:`success`, :func:`failure`,
  :func:`cancelled`, :func:`timeout`, :func:`retry_exhausted`) so callers
  don't need to know ``ExecutionResult``'s full constructor signature.
* :func:`from_awaitable`, an async-friendly helper that awaits a
  coroutine and translates its outcome into an ``ExecutionResult``.
* JSON- and structlog-safe serialization (:func:`to_dict` / :func:`to_json`
  / :func:`log_result`).
* :func:`to_event`, which converts a terminal ``ExecutionResult`` into
  the matching :mod:`kodiak.orchestration.execution_events` event, so a
  result can be published through the same event stream used for the
  execution's lifecycle.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from structlog.typing import FilteringBoundLogger

# Re-exported for backward compatibility: code that previously imported
# ExecutionResult, ExecutionOutcome, or RetryPolicy from this module
# continues to work unchanged, now backed by the canonical execution
# package instead of a competing definition.
from kodiak.db.models.task import TaskStatus
from kodiak.orchestration.execution import (
    ExecutionOutcome,
    ExecutionResult,
    RetryPolicy,
)
from kodiak.orchestration.execution_events import (
    AnyExecutionEvent,
    ExecutionCancelledEvent,
    ExecutionFailedEvent,
    ExecutionSucceededEvent,
)

__all__ = [
    "ExecutionOutcome",
    "ExecutionResult",
    "RetryPolicy",
    "success",
    "failure",
    "cancelled",
    "timeout",
    "retry_exhausted",
    "from_awaitable",
    "to_dict",
    "to_json",
    "to_event",
    "log_result",
]


def _now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def _serialize(value: Any) -> Any:
    """Recursively convert a value into a JSON-safe representation.

    Args:
        value: Any value that may appear on an ``ExecutionResult``,
            including enums, timestamps, paths, and nested dataclasses
            such as ``RetryPolicy``.

    Returns:
        A value composed only of ``dict``, ``list``, ``str``, ``int``,
        ``float``, ``bool``, and ``None``.
    """
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def success(
    *,
    task_id: str,
    output: Any = None,
    attempts: int = 1,
    duration_seconds: float = 0.0,
) -> ExecutionResult:
    """Build a successful :class:`ExecutionResult`."""
    return ExecutionResult(
        task_id=task_id,
        outcome=ExecutionOutcome.SUCCESS,
        attempts=attempts,
        duration_seconds=duration_seconds,
        result=dict(output) if isinstance(output, Mapping) else {},
        final_status=TaskStatus.COMPLETED,
    )


def failure(
    *,
    task_id: str,
    error: BaseException | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    output: Any = None,
    attempts: int = 1,
    duration_seconds: float = 0.0,
) -> ExecutionResult:
    """Build a failed :class:`ExecutionResult`."""
    resolved_type = error_type or (type(error).__name__ if error else "UnknownError")
    resolved_message = error_message or (str(error) if error else "Execution failed")
    error_dict: dict[str, Any] = {"type": resolved_type, "message": resolved_message}
    return ExecutionResult(
        task_id=task_id,
        outcome=ExecutionOutcome.FAILURE,
        attempts=attempts,
        duration_seconds=duration_seconds,
        result=dict(output) if isinstance(output, Mapping) else {},
        error=error_dict,
        final_status=TaskStatus.FAILED,
    )


def cancelled(
    *,
    task_id: str,
    reason: str,
    attempts: int = 1,
    duration_seconds: float = 0.0,
) -> ExecutionResult:
    """Build a cancelled :class:`ExecutionResult`."""
    return ExecutionResult(
        task_id=task_id,
        outcome=ExecutionOutcome.CANCELLED,
        attempts=attempts,
        duration_seconds=duration_seconds,
        error={"type": "ExecutionCancelledError", "message": reason},
        final_status=TaskStatus.CANCELLED,
    )


def timeout(
    *,
    task_id: str,
    timeout_seconds: float,
    attempts: int = 1,
    duration_seconds: float = 0.0,
) -> ExecutionResult:
    """Build a timed-out :class:`ExecutionResult`."""
    return ExecutionResult(
        task_id=task_id,
        outcome=ExecutionOutcome.TIMEOUT,
        attempts=attempts,
        duration_seconds=duration_seconds,
        error={
            "type": "TimeoutError",
            "message": f"Execution exceeded timeout of {timeout_seconds}s",
        },
        final_status=TaskStatus.FAILED,
    )


def retry_exhausted(
    *,
    task_id: str,
    attempts: int,
    retry_policy: RetryPolicy,
    last_error: BaseException | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    duration_seconds: float = 0.0,
) -> ExecutionResult:
    """Build an :class:`ExecutionResult` for retries that never succeeded."""
    resolved_type = error_type or (
        type(last_error).__name__ if last_error else "RetriesExhaustedError"
    )
    resolved_message = error_message or (
        str(last_error) if last_error else f"Execution failed after {attempts} attempts"
    )
    return ExecutionResult(
        task_id=task_id,
        outcome=ExecutionOutcome.RETRY_EXHAUSTED,
        attempts=attempts,
        duration_seconds=duration_seconds,
        error={"type": resolved_type, "message": resolved_message},
        final_status=TaskStatus.FAILED,
    )


async def from_awaitable(
    awaitable: Awaitable[Any],
    *,
    task_id: str,
    timeout_seconds: float | None = None,
) -> ExecutionResult:
    """Await a coroutine and translate its outcome into an ExecutionResult."""
    started = _now()
    try:
        if timeout_seconds is not None:
            output = await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        else:
            output = await awaitable
    except TimeoutError:
        elapsed = (_now() - started).total_seconds()
        return timeout(
            task_id=task_id,
            timeout_seconds=timeout_seconds or 0.0,
            duration_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (_now() - started).total_seconds()
        return failure(task_id=task_id, error=exc, duration_seconds=elapsed)
    elapsed = (_now() - started).total_seconds()
    return success(task_id=task_id, output=output, duration_seconds=elapsed)


def to_dict(result: ExecutionResult) -> dict[str, Any]:
    """Convert an :class:`ExecutionResult` into a JSON-safe dictionary.

    Args:
        result: The execution result to serialize.

    Returns:
        A dictionary containing every field of ``result``, with enums,
        timestamps, paths, and nested dataclasses converted into
        JSON-safe primitives.
    """
    if is_dataclass(result) and not isinstance(result, type):
        return {f.name: _serialize(getattr(result, f.name)) for f in fields(result)}
    # Defensive fallback in case the canonical model is not a dataclass.
    return {key: _serialize(value) for key, value in vars(result).items()}


def to_json(result: ExecutionResult) -> str:
    """Serialize an :class:`ExecutionResult` to a JSON string.

    Args:
        result: The execution result to serialize.

    Returns:
        A JSON-encoded representation of ``result``.
    """
    return json.dumps(to_dict(result), default=str)


def to_event(result: ExecutionResult, *, execution_id: str | None = None) -> AnyExecutionEvent:
    """Convert a terminal :class:`ExecutionResult` into its matching event.

    Args:
        result: The execution result to convert.
        execution_id: Overrides the execution id used on the emitted
            event. Defaults to ``result.execution_id``.

    Returns:
        The ``ExecutionEvent`` subclass matching ``result.outcome``.

    Raises:
        ValueError: If ``result.outcome`` is not a recognized value.
    """
    resolved_execution_id = execution_id or f"{result.task_id}:{result.outcome.value}"

    error_payload = result.error or {}
    error_type = error_payload.get("type", result.outcome.value)
    error_message = error_payload.get("message", "Execution failed")

    if result.outcome is ExecutionOutcome.SUCCESS:
        return ExecutionSucceededEvent(
            task_id=result.task_id,
            execution_id=resolved_execution_id,
            result=result,
            duration_seconds=result.duration_seconds,
        )
    if result.outcome is ExecutionOutcome.CANCELLED:
        return ExecutionCancelledEvent(
            task_id=result.task_id,
            execution_id=resolved_execution_id,
            reason=error_message or "Execution cancelled",
        )
    if result.outcome in (
        ExecutionOutcome.FAILURE,
        ExecutionOutcome.TIMEOUT,
        ExecutionOutcome.RETRY_EXHAUSTED,
    ):
        return ExecutionFailedEvent(
            task_id=result.task_id,
            execution_id=resolved_execution_id,
            error_type=str(error_type),
            error_message=str(error_message),
            duration_seconds=result.duration_seconds,
            attempt=result.attempts,
            final=True,
        )
    raise ValueError(f"Unrecognized execution outcome: {result.outcome!r}")


def log_result(
    result: ExecutionResult,
    logger: FilteringBoundLogger,
    *,
    level: str | None = None,
) -> None:
    """Emit an :class:`ExecutionResult` through a structlog-compatible logger.

    Args:
        result: The execution result to log.
        logger: A bound structlog logger to emit the result through.
        level: The log level to emit at. Defaults to ``"info"`` for a
            successful outcome and ``"error"`` otherwise. Must name a
            method on ``logger``.
    """
    resolved_level = level or ("info" if result.outcome is ExecutionOutcome.SUCCESS else "error")
    log_method = getattr(logger, resolved_level)
    log_method(f"execution_result.{result.outcome.value}", **to_dict(result))
