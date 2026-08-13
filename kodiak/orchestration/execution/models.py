"""
kodiak/execution/models.py

Domain objects for the Execution Engine. These describe the runtime
lifecycle of a single Task execution and are deliberately independent of
persistence concerns (SQLAlchemy sessions) and of whatever internal
representation the Agent Manager eventually uses. The only external
dependency is the existing `Task` / `TaskStatus` model, which is reused
as-is rather than re-derived.
"""

from __future__ import annotations

import asyncio
import enum
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from kodiak.db.models.task import Task, TaskStatus


class ExecutionOutcome(enum.StrEnum):
    """Terminal outcome of a single ExecutionEngine.execute() run.

    Distinct from `TaskStatus`: this is an application-layer concept
    describing *why* execution ended, which the engine then maps onto the
    persisted `TaskStatus` vocabulary via `outcome_to_task_status`.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRY_EXHAUSTED = "retry_exhausted"


_OUTCOME_TO_TASK_STATUS: dict[ExecutionOutcome, TaskStatus] = {
    ExecutionOutcome.SUCCESS: TaskStatus.COMPLETED,
    ExecutionOutcome.FAILURE: TaskStatus.FAILED,
    ExecutionOutcome.TIMEOUT: TaskStatus.FAILED,
    ExecutionOutcome.CANCELLED: TaskStatus.CANCELLED,
    ExecutionOutcome.RETRY_EXHAUSTED: TaskStatus.FAILED,
}


def outcome_to_task_status(outcome: ExecutionOutcome) -> TaskStatus:
    """Map an ExecutionOutcome onto the persisted TaskStatus vocabulary."""
    return _OUTCOME_TO_TASK_STATUS[outcome]


class ExecutionEventType(enum.StrEnum):
    """Event types emitted by the Execution Engine for observability/hooks."""

    TASK_STARTED = "task_started"
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_SUCCEEDED = "attempt_succeeded"
    ATTEMPT_FAILED = "attempt_failed"
    RETRY_SCHEDULED = "retry_scheduled"
    TIMEOUT = "timeout"
    PROGRESS = "progress"
    CANCELLED = "cancelled"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """A single lifecycle event, delivered to any hooks registered for its type."""

    type: ExecutionEventType
    task_id: str
    attempt: int
    timestamp: float = field(default_factory=time.time)
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionProgress:
    """A progress update reported by the Agent Manager mid-attempt."""

    task_id: str
    attempt: int
    percent: float | None
    message: str
    timestamp: float = field(default_factory=time.time)


ExecutionHook = Callable[[ExecutionEvent], Awaitable[None]]
ProgressCallback = Callable[[ExecutionProgress], Awaitable[None]]


@dataclass(slots=True)
class RetryPolicy:
    """Exponential backoff retry policy for a single execution.

    Attributes:
        max_attempts: Total attempts allowed, including the first (a value
            of 3 means up to 2 retries after the initial attempt).
        base_delay_seconds: Delay before the first retry.
        max_delay_seconds: Upper bound on backoff delay.
        backoff_multiplier: Multiplier applied to the delay after each
            failed attempt.
        jitter_seconds: Random jitter (0..jitter_seconds) added to each
            delay to avoid thundering-herd retries.
        retryable_exceptions: Exception types considered retryable. Any
            exception not matching this tuple is treated as terminal
            regardless of remaining attempts.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_seconds: float = 0.5
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)

    @classmethod
    def from_task(cls, task: Task, **overrides: Any) -> "RetryPolicy":
        """Derive a policy from `Task.max_retries`, the existing retry field.

        Args:
            task: The task whose `max_retries` sets the attempt budget.
            **overrides: Any other `RetryPolicy` field to override.

        Returns:
            A RetryPolicy with `max_attempts = task.max_retries + 1`.
        """
        policy = cls(max_attempts=max(task.max_retries, 0) + 1)
        for key, value in overrides.items():
            setattr(policy, key, value)
        return policy

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute the backoff delay before retrying after `attempt`."""
        raw = self.base_delay_seconds * (self.backoff_multiplier ** max(attempt - 1, 0))
        capped = min(raw, self.max_delay_seconds)
        return capped + random.uniform(0, self.jitter_seconds)

    def is_retryable(self, exc: BaseException) -> bool:
        """Whether `exc` matches this policy's retryable exception types."""
        return isinstance(exc, self.retryable_exceptions)


class CancellationToken:
    """Cooperative cancellation signal shared across an execution's attempts.

    Checked between attempts, during backoff waits, and raced against the
    Agent Manager's in-flight coroutine so cancellation takes effect
    promptly rather than only at attempt boundaries.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Signal cancellation. Idempotent."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        """Suspend until cancellation is signalled."""
        await self._event.wait()


@dataclass(slots=True)
class ExecutionContext:
    """Per-execution state threaded through the engine and into the Agent Manager.

    Attributes:
        task: The task being executed. Mutated in place as execution
            progresses (status, retry_count, result, error, cost fields).
        correlation_id: Unique id for this execution run, bound into all
            structured logs for cross-service tracing.
        attempt: 1-indexed current attempt number.
        cancellation_token: Shared token the Agent Manager should poll
            cooperatively for long-running work.
        progress_callback: Optional sink for progress updates; the Agent
            Manager calls `context.report_progress(...)` to use it.
        metadata: Free-form bag for engine/agent-manager-specific extras.
    """

    task: Task
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attempt: int = 0
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    progress_callback: ProgressCallback | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    async def report_progress(self, message: str, percent: float | None = None) -> None:
        """Report progress for the current attempt, if a callback is wired up."""
        if self.progress_callback is None:
            return
        await self.progress_callback(
            ExecutionProgress(
                task_id=str(self.task.id),
                attempt=self.attempt,
                percent=percent,
                message=message,
            )
        )


@dataclass(slots=True)
class AgentManagerResult:
    """What the Agent Manager returns to the Execution Engine on success.

    Intentionally minimal — the Agent Manager milestone will likely extend
    this internally, but the engine only needs an output payload plus
    optional usage accounting to fold into the Task record.
    """

    output: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    cost_usd: float | None = None


@dataclass(slots=True)
class ExecutionResult:
    """Terminal result of `ExecutionEngine.execute()`, returned to the Supervisor."""

    task_id: str
    outcome: ExecutionOutcome
    attempts: int
    duration_seconds: float
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    final_status: TaskStatus = TaskStatus.FAILED
    verification: dict[str, Any] | None = None
    reflection: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        return self.outcome is ExecutionOutcome.SUCCESS