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
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from structlog.typing import FilteringBoundLogger

# Re-exported for backward compatibility: code that previously imported
# ExecutionResult, ExecutionOutcome, or RetryPolicy from this module
# continues to work unchanged, now backed by the canonical execution
# package instead of a competing definition.
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
    return datetime.now(timezone.utc)


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


def _duration_seconds(result: ExecutionResult) -> float:
    """Resolve the duration of an execution, in seconds.

    Prefers a ``duration_seconds`` attribute already exposed by the
    canonical model; falls back to deriving it from ``started_at`` and
    ``finished_at`` otherwise.

    Args:
        result: The execution result to measure.

    Returns:
        The elapsed wall-clock time, in seconds.
    """
    existing = getattr(result, "duration_seconds", None)
    if isinstance(existing, (int, float)):
        return float(existing)
    return (result.finished_at - result.started_at).total_seconds()


def success(
    *,
    execution_id: str,
    task_id: str,
    started_at: datetime,
    finished_at: datetime | None = None,
    output: Any = None,
    metadata: Mapping[str, Any] | None = None,
    attempts: int = 1,
) -> ExecutionResult:
    """Build a successful :class:`ExecutionResult`.

    Args:
        execution_id: Identifier correlating this result to its
            execution attempt.
        task_id: Identifier of the task that was executed.
        started_at: When the execution attempt began.
        finished_at: When the execution attempt completed. Defaults to
            the current UTC time.
        output: The payload produced by the execution (e.g. a diff
            summary, PR URL, or generated artifact references).
        metadata: Free-form structured metadata for observability.
        attempts: Number of attempts made before succeeding.

    Returns:
        An ``ExecutionResult`` with ``outcome=ExecutionOutcome.SUCCESS``.
    """
    return ExecutionResult(
        execution_id=execution_id,
        task_id=task_id,
        outcome=ExecutionOutcome.SUCCESS,
        started_at=started_at,
        finished_at=finished_at or _now(),
        output=output,
        metadata=dict(metadata or {}),
        error_type=None,
        error_message=None,
        attempts=attempts,
        max_attempts=None,
        retry_policy=None,
    )


def failure(
    *,
    execution_id: str,
    task_id: str,
    started_at: datetime,
    error: BaseException | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    finished_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    attempts: int = 1,
) -> ExecutionResult:
    """Build a failed :class:`ExecutionResult`.

    Args:
        execution_id: Identifier correlating this result to its
            execution attempt.
        task_id: Identifier of the task that was executed.
        started_at: When the execution attempt began.
        error: The exception that caused the failure, if available.
            Used to derive ``error_type``/``error_message`` when they
            are not supplied explicitly.
        error_type: The class name of the causing exception. Derived
            from ``error`` when omitted.
        error_message: A human-readable description of the failure.
            Derived from ``error`` when omitted.
        finished_at: When the execution attempt failed. Defaults to the
            current UTC time.
        metadata: Free-form structured metadata for observability.
        attempts: Number of attempts made before failing.

    Returns:
        An ``ExecutionResult`` with ``outcome=ExecutionOutcome.FAILURE``.
    """
    resolved_type = error_type or (type(error).__name__ if error else "UnknownError")
    resolved_message = error_message or (str(error) if error else "Execution failed")
    return ExecutionResult(
        execution_id=execution_id,
        task_id=task_id,
        outcome=ExecutionOutcome.FAILURE,
        started_at=started_at,
        finished_at=finished_at or _now(),
        output=None,
        metadata=dict(metadata or {}),
        error_type=resolved_type,
        error_message=resolved_message,
        attempts=attempts,
        max_attempts=None,
        retry_policy=None,
    )


def cancelled(
    *,
    execution_id: str,
    task_id: str,
    started_at: datetime,
    reason: str,
    finished_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    attempts: int = 1,
) -> ExecutionResult:
    """Build a cancelled :class:`ExecutionResult`.

    Args:
        execution_id: Identifier correlating this result to its
            execution attempt.
        task_id: Identifier of the task that was executed.
        started_at: When the execution attempt began.
        reason: A human-readable description of why execution was
            cancelled.
        finished_at: When execution stopped. Defaults to the current
            UTC time.
        metadata: Free-form structured metadata for observability.
        attempts: Number of attempts made before cancellation.

    Returns:
        An ``ExecutionResult`` with ``outcome=ExecutionOutcome.CANCELLED``.
    """
    return ExecutionResult(
        execution_id=execution_id,
        task_id=task_id,
        outcome=ExecutionOutcome.CANCELLED,
        started_at=started_at,
        finished_at=finished_at or _now(),
        output=None,
        metadata=dict(metadata or {}),
        error_type=None,
        error_message=reason,
        attempts=attempts,
        max_attempts=None,
        retry_policy=None,
    )


def timeout(
    *,
    execution_id: str,
    task_id: str,
    started_at: datetime,
    timeout_seconds: float,
    finished_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    attempts: int = 1,
) -> ExecutionResult:
    """Build a timed-out :class:`ExecutionResult`.

    Args:
        execution_id: Identifier correlating this result to its
            execution attempt.
        task_id: Identifier of the task that was executed.
        started_at: When the execution attempt began.
        timeout_seconds: The timeout threshold, in seconds, that was
            exceeded.
        finished_at: When the timeout was detected. Defaults to the
            current UTC time.
        metadata: Free-form structured metadata for observability.
        attempts: Number of attempts made before timing out.

    Returns:
        An ``ExecutionResult`` with ``outcome=ExecutionOutcome.TIMEOUT``.
    """
    return ExecutionResult(
        execution_id=execution_id,
        task_id=task_id,
        outcome=ExecutionOutcome.TIMEOUT,
        started_at=started_at,
        finished_at=finished_at or _now(),
        output=None,
        metadata=dict(metadata or {}),
        error_type="TimeoutError",
        error_message=f"Execution exceeded timeout of {timeout_seconds}s",
        attempts=attempts,
        max_attempts=None,
        retry_policy=None,
    )


def retry_exhausted(
    *,
    execution_id: str,
    task_id: str,
    started_at: datetime,
    attempts: int,
    retry_policy: RetryPolicy,
    last_error: BaseException | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    finished_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Build an :class:`ExecutionResult` for retries that never succeeded.

    Args:
        execution_id: Identifier correlating this result to its
            execution attempt.
        task_id: Identifier of the task that was executed.
        started_at: When the first attempt began.
        attempts: Total number of attempts made.
        retry_policy: The ``RetryPolicy`` that governed the retries.
        last_error: The exception raised by the final attempt, if
            available. Used to derive ``error_type``/``error_message``
            when they are not supplied explicitly.
        error_type: The class name of the final attempt's exception.
            Derived from ``last_error`` when omitted.
        error_message: A human-readable description of the final
            failure. Derived from ``last_error`` when omitted.
        finished_at: When the final attempt failed. Defaults to the
            current UTC time.
        metadata: Free-form structured metadata for observability.

    Returns:
        An ``ExecutionResult`` with
        ``outcome=ExecutionOutcome.RETRY_EXHAUSTED``.
    """
    resolved_type = error_type or (
        type(last_error).__name__ if last_error else "RetriesExhaustedError"
    )
    resolved_message = error_message or (
        str(last_error) if last_error else f"Execution failed after {attempts} attempts"
    )
    return ExecutionResult(
        execution_id=execution_id,
        task_id=task_id,
        outcome=ExecutionOutcome.RETRY_EXHAUSTED,
        started_at=started_at,
        finished_at=finished_at or _now(),
        output=None,
        metadata=dict(metadata or {}),
        error_type=resolved_type,
        error_message=resolved_message,
        attempts=attempts,
        max_attempts=retry_policy.max_attempts,
        retry_policy=retry_policy,
    )


async def from_awaitable(
    awaitable: Awaitable[Any],
    *,
    execution_id: str,
    task_id: str,
    timeout_seconds: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Await a coroutine and translate its outcome into an ExecutionResult.

    Cancellation is intentionally left to propagate rather than being
    converted into a ``cancelled()`` result: per ``asyncio`` semantics,
    ``CancelledError`` must reach the enclosing task so cancellation
    unwinds correctly. Callers that catch ``asyncio.CancelledError``
    around the awaited task should build the ``cancelled()`` result
    themselves once unwinding is complete.

    Args:
        awaitable: The coroutine or future representing the execution.
        execution_id: Identifier correlating this result to its
            execution attempt.
        task_id: Identifier of the task being executed.
        timeout_seconds: Optional timeout, in seconds, applied to
            ``awaitable``.
        metadata: Free-form structured metadata for observability.

    Returns:
        A ``success`` result if ``awaitable`` completes normally, a
        ``timeout`` result if it exceeds ``timeout_seconds``, or a
        ``failure`` result if it raises any other exception.
    """
    started_at = _now()
    try:
        if timeout_seconds is not None:
            output = await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        else:
            output = await awaitable
    except asyncio.TimeoutError:
        return timeout(
            execution_id=execution_id,
            task_id=task_id,
            started_at=started_at,
            timeout_seconds=timeout_seconds or 0.0,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 - captured as a failure result
        return failure(
            execution_id=execution_id,
            task_id=task_id,
            started_at=started_at,
            error=exc,
            metadata=metadata,
        )
    return success(
        execution_id=execution_id,
        task_id=task_id,
        started_at=started_at,
        output=output,
        metadata=metadata,
    )


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


def to_event(
    result: ExecutionResult, *, execution_id: str | None = None
) -> AnyExecutionEvent:
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
    resolved_execution_id = execution_id or result.execution_id

    if result.outcome is ExecutionOutcome.SUCCESS:
        return ExecutionSucceededEvent(
            task_id=result.task_id,
            execution_id=resolved_execution_id,
            result=result,
            duration_seconds=_duration_seconds(result),
        )
    if result.outcome is ExecutionOutcome.CANCELLED:
        return ExecutionCancelledEvent(
            task_id=result.task_id,
            execution_id=resolved_execution_id,
            reason=result.error_message or "Execution cancelled",
        )
    if result.outcome in (
        ExecutionOutcome.FAILURE,
        ExecutionOutcome.TIMEOUT,
        ExecutionOutcome.RETRY_EXHAUSTED,
    ):
        return ExecutionFailedEvent(
            task_id=result.task_id,
            execution_id=resolved_execution_id,
            error_type=result.error_type or result.outcome.value,
            error_message=result.error_message or "Execution failed",
            duration_seconds=_duration_seconds(result),
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
    resolved_level = level or (
        "info" if result.outcome is ExecutionOutcome.SUCCESS else "error"
    )
    log_method = getattr(logger, resolved_level)
    log_method(f"execution_result.{result.outcome.value}", **to_dict(result))