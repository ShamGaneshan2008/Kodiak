"""
kodiak/execution/interfaces.py

Ports the Execution Engine depends on. Concrete adapters (the real Agent
Manager, a SQLAlchemy-backed Task repository) are supplied by the caller
via dependency injection at construction time — this module, and
engine.py, never import infrastructure code directly.

Neither of these interfaces exists elsewhere in the codebase yet, so they
are defined here as the contract the next milestones (Agent Manager,
persistence layer) must satisfy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kodiak.db.models.task import Task, TaskStatus
from kodiak.orchestration.execution.models import AgentManagerResult, ExecutionContext


@runtime_checkable
class AgentManager(Protocol):
    """Runs the specialized-agent pipeline for a single Task attempt.

    Implemented by the Agent Manager milestone. The Execution Engine only
    needs this single entrypoint: hand it an execution context, get back a
    result or have it raise. Implementations should poll
    `context.cancellation_token` cooperatively during long-running work and
    call `context.report_progress(...)` to surface progress upstream.
    """

    async def run(self, context: ExecutionContext) -> AgentManagerResult:
        """Execute one attempt for `context.task` and return its result.

        Raises:
            Exception: Any failure. Retryability is determined by the
                engine's `RetryPolicy` unless the implementation raises
                `kodiak.execution.exceptions.NonRetryableExecutionError`
                to force a terminal failure.
        """
        ...


@runtime_checkable
class TaskRepository(Protocol):
    """Persistence port for Task lifecycle updates.

    The Execution Engine reports status transitions through this
    interface rather than owning a database session directly, keeping
    orchestration logic independent of the persistence layer. Passing
    `task_repository=None` to `ExecutionEngine` is valid; the engine still
    mutates the in-memory `Task` correctly and the caller becomes
    responsible for persisting it.
    """

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        **fields: object,
    ) -> None:
        """Persist a status transition (and any accompanying fields) for `task_id`."""
        ...

    async def save(self, task: Task) -> None:
        """Persist the full current state of `task`."""
        ...
