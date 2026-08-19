"""Execution lifecycle event definitions for Kodiak's orchestration layer.

This module defines the structured events emitted over the lifetime of a
single execution attempt -- from start, through progress and retries, to
a terminal success, failure, or cancellation. Events are immutable,
fully typed, and JSON-serializable, so they can be logged, persisted, or
published to subscribers (the CLI's progress display, the Review Engine,
webhooks, etc.) without any subscriber needing to know about the
Execution Engine itself.

This module intentionally contains only event definitions. It reuses
``Task``, ``ExecutionContext``, ``ExecutionResult``, and ``RetryPolicy``
from ``kodiak.orchestration.execution`` rather than redefining them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from structlog.typing import FilteringBoundLogger

from kodiak.db.models.task import Task
from kodiak.orchestration.execution import (
    ExecutionContext,
    ExecutionResult,
    RetryPolicy,
)


class ExecutionEventType(StrEnum):
    """Discriminator identifying the kind of execution event."""

    STARTED = "started"
    PROGRESS = "progress"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _serialize(value: Any) -> Any:
    """Recursively convert a value into a JSON-safe representation.

    Args:
        value: Any value that may appear as an execution event field,
            including enums, timestamps, paths, and nested dataclasses
            such as ``Task``, ``ExecutionContext``, ``ExecutionResult``,
            or ``RetryPolicy``.

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


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionEvent:
    """Base class for every execution lifecycle event.

    Attributes:
        event_type: Discriminator identifying the concrete event kind.
        task_id: Identifier of the task being executed.
        execution_id: Identifier correlating every event emitted during a
            single execution attempt.
        timestamp: UTC timestamp of when the event occurred. Defaults to
            the moment the event is constructed.
        task: The full ``Task`` this event relates to, if the emitter has
            it on hand. Optional, since ``task_id`` alone is sufficient
            for correlation.
        metadata: Free-form structured metadata for observability,
            tracing, or subscriber-specific enrichment.
    """

    event_type: ExecutionEventType
    task_id: str
    execution_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    task: Task | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def log_event_name(self) -> str:
        """The dot-namespaced event name used for structured logging."""
        return f"execution.{self.event_type.value}"

    def to_dict(self) -> dict[str, Any]:
        """Convert this event into a JSON-safe dictionary.

        Returns:
            A dictionary containing every field of this event, with
            enums, timestamps, paths, and nested dataclasses converted
            into JSON-safe primitives.
        """
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    def to_json(self) -> str:
        """Serialize this event to a JSON string.

        Returns:
            A JSON-encoded representation of this event.
        """
        return json.dumps(self.to_dict(), default=str)

    def log(self, logger: FilteringBoundLogger, *, level: str = "info") -> None:
        """Emit this event through a structlog-compatible logger.

        Args:
            logger: A bound structlog logger to emit the event through.
            level: The log level to emit at (e.g. ``"info"``,
                ``"warning"``, ``"error"``). Must name a method on
                ``logger``.
        """
        log_method = getattr(logger, level)
        log_method(self.log_event_name, **self.to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionStartedEvent(ExecutionEvent):
    """Emitted when an execution attempt begins running.

    Attributes:
        attempt: The 1-indexed attempt number for this execution.
        context: The ``ExecutionContext`` the attempt is running under,
            if available at emission time.
    """

    event_type: ExecutionEventType = field(default=ExecutionEventType.STARTED, init=False)
    attempt: int = 1
    context: ExecutionContext | None = None

    def __post_init__(self) -> None:
        """Validate that the attempt number is well-formed."""
        if self.attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {self.attempt!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionProgressEvent(ExecutionEvent):
    """Emitted to report incremental progress during execution.

    Attributes:
        progress: Fractional completion in the inclusive range
            ``[0.0, 1.0]``.
        current_step: A short label for the step currently in progress.
        message: A human-readable progress message.
    """

    event_type: ExecutionEventType = field(default=ExecutionEventType.PROGRESS, init=False)
    progress: float
    current_step: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate that progress is a well-formed fraction."""
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"progress must be within [0.0, 1.0], got {self.progress!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionRetryingEvent(ExecutionEvent):
    """Emitted before a failed execution attempt is retried.

    Attributes:
        attempt: The 1-indexed attempt number that just failed.
        max_attempts: The maximum number of attempts permitted.
        delay_seconds: How long the engine will wait before retrying.
        reason: A human-readable description of why the attempt failed.
        retry_policy: The ``RetryPolicy`` governing this retry, if
            available at emission time.
    """

    event_type: ExecutionEventType = field(default=ExecutionEventType.RETRYING, init=False)
    attempt: int
    max_attempts: int
    delay_seconds: float
    reason: str
    retry_policy: RetryPolicy | None = None

    def __post_init__(self) -> None:
        """Validate attempt bookkeeping and delay are well-formed."""
        if self.attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {self.attempt!r}")
        if self.max_attempts < self.attempt:
            raise ValueError("max_attempts must be >= attempt")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCancelledEvent(ExecutionEvent):
    """Emitted when an execution attempt is cancelled before completion.

    Attributes:
        reason: A human-readable description of why execution was
            cancelled.
        cancelled_by: Identifier of the actor that requested cancellation
            (e.g. a user id, ``"system"``, or ``"timeout"``).
        graceful: Whether the execution engine was able to unwind cleanly,
            as opposed to being forcibly terminated.
    """

    event_type: ExecutionEventType = field(default=ExecutionEventType.CANCELLED, init=False)
    reason: str
    cancelled_by: str | None = None
    graceful: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionSucceededEvent(ExecutionEvent):
    """Emitted when an execution attempt completes successfully.

    Attributes:
        result: The ``ExecutionResult`` produced by the attempt.
        duration_seconds: Total wall-clock time the attempt took.
    """

    event_type: ExecutionEventType = field(default=ExecutionEventType.SUCCEEDED, init=False)
    result: ExecutionResult
    duration_seconds: float

    def __post_init__(self) -> None:
        """Validate that duration is well-formed."""
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionFailedEvent(ExecutionEvent):
    """Emitted when an execution attempt fails with no further retries.

    Attributes:
        error_type: The class name of the exception that caused failure.
        error_message: A human-readable description of the failure.
        duration_seconds: Total wall-clock time the attempt took before
            failing.
        attempt: The 1-indexed attempt number that failed terminally.
        final: Whether this failure is terminal (no further retries will
            be attempted). Always ``True`` in practice; retryable
            failures are represented by ``ExecutionRetryingEvent``
            instead.
    """

    event_type: ExecutionEventType = field(default=ExecutionEventType.FAILED, init=False)
    error_type: str
    error_message: str
    duration_seconds: float
    attempt: int = 1
    final: bool = True

    def __post_init__(self) -> None:
        """Validate that duration and attempt bookkeeping are well-formed."""
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {self.attempt!r}")


type AnyExecutionEvent = (
    ExecutionStartedEvent
    | ExecutionProgressEvent
    | ExecutionRetryingEvent
    | ExecutionCancelledEvent
    | ExecutionSucceededEvent
    | ExecutionFailedEvent
)
"""Union of every concrete execution event, for typing event handlers."""


__all__ = [
    "AnyExecutionEvent",
    "ExecutionCancelledEvent",
    "ExecutionEvent",
    "ExecutionEventType",
    "ExecutionFailedEvent",
    "ExecutionProgressEvent",
    "ExecutionRetryingEvent",
    "ExecutionStartedEvent",
    "ExecutionSucceededEvent",
]
