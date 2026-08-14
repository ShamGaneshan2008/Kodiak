"""
kodiak/orchestration/state.py

Defines the canonical runtime state for a Kodiak agent task execution.
All orchestration components read from and write to these models, making
the state the single source of truth for a running task.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Lifecycle status of a top-level task."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    REPAIRING = "repairing"
    REPLANNING = "replanning"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(StrEnum):
    """Runtime status of a registered agent instance."""

    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    STOPPED = "stopped"


# Valid orchestration state transitions for the autonomous task loop.
TASK_STATUS_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED}),
    TaskStatus.PLANNING: frozenset(
        {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.REPLANNING}
    ),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.VERIFYING, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PAUSED}
    ),
    TaskStatus.VERIFYING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.REFLECTING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.REFLECTING: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.REPAIRING,
            TaskStatus.PLANNING,
            TaskStatus.REPLANNING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.REPAIRING: frozenset(
        {TaskStatus.RUNNING, TaskStatus.VERIFYING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.REPLANNING: frozenset(
        {TaskStatus.PLANNING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.AWAITING_APPROVAL: frozenset(
        {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.PAUSED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTaskStatusTransition(ValueError):
    """Raised when an orchestration component attempts an illegal state change."""


def transition_task_status(current: TaskStatus, new: TaskStatus) -> None:
    """Validate and apply a task status transition.

    Raises:
        InvalidTaskStatusTransition: If ``new`` is not allowed from ``current``.
    """
    allowed = TASK_STATUS_TRANSITIONS.get(current, frozenset())
    if new not in allowed:
        raise InvalidTaskStatusTransition(
            f"Cannot transition task status from {current.value!r} to {new.value!r}"
        )


class StepStatus(StrEnum):
    """Lifecycle status of an individual execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentRole(StrEnum):
    """The functional role of an agent within the orchestration graph."""

    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    RESEARCHER = "researcher"
    DEBUGGER = "debugger"
    RETRIEVAL = "retrieval"
    REFLECTION = "reflection"
    MEMORY = "memory"


class ApprovalStatus(StrEnum):
    """Status of a human-in-the-loop approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A record of a single tool invocation made by an agent."""

    tool_call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str = Field(..., description="Registered name of the tool.")
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = Field(None, description="Serialisable result returned by the tool.")
    error: str | None = Field(None, description="Error message if the call failed.")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    duration_ms: int | None = None

    def mark_finished(self, result: Any | None = None, error: str | None = None) -> None:
        """Record the completion time and outcome of this tool call."""
        self.finished_at = datetime.now(UTC)
        delta = self.finished_at - self.started_at
        self.duration_ms = int(delta.total_seconds() * 1000)
        self.result = result
        self.error = error


class ExecutionStep(BaseModel):
    """One discrete unit of work within the overall task plan."""

    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    index: int = Field(..., description="Zero-based position in the plan sequence.")
    description: str = Field(..., description="Human-readable description of the step.")
    agent_role: AgentRole = Field(..., description="Which agent type owns this step.")
    status: StepStatus = Field(StepStatus.PENDING)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output: str | None = Field(None, description="Textual output produced by the step.")
    error: str | None = None
    depends_on: list[str] = Field(
        default_factory=list,
        description="step_id values that must complete before this step runs.",
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def start(self) -> None:
        """Transition the step to the running state."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, output: str) -> None:
        """Mark the step as successfully completed."""
        self.status = StepStatus.COMPLETED
        self.output = output
        self.finished_at = datetime.now(UTC)

    def fail(self, error: str) -> None:
        """Mark the step as failed with a reason."""
        self.status = StepStatus.FAILED
        self.error = error
        self.finished_at = datetime.now(UTC)

    def skip(self) -> None:
        """Mark the step as skipped (e.g. dependency failed)."""
        self.status = StepStatus.SKIPPED
        self.finished_at = datetime.now(UTC)

    @property
    def is_terminal(self) -> bool:
        """Return True if the step has reached a final state."""
        return self.status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}

    @property
    def duration_ms(self) -> int | None:
        """Wall-clock duration in milliseconds, or None if not yet finished."""
        if self.started_at and self.finished_at:
            delta = self.finished_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class ApprovalRequest(BaseModel):
    """A human-in-the-loop gate that must be resolved before execution continues."""

    approval_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    reason: str = Field(..., description="Why approval is required.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information presented to the approver.",
    )
    status: ApprovalStatus = Field(ApprovalStatus.PENDING)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    resolved_by: str | None = Field(None, description="User ID or system that resolved this.")
    resolution_note: str | None = None

    def approve(self, resolved_by: str, note: str | None = None) -> None:
        """Record an approval decision."""
        self.status = ApprovalStatus.APPROVED
        self.resolved_at = datetime.now(UTC)
        self.resolved_by = resolved_by
        self.resolution_note = note

    def reject(self, resolved_by: str, note: str | None = None) -> None:
        """Record a rejection decision."""
        self.status = ApprovalStatus.REJECTED
        self.resolved_at = datetime.now(UTC)
        self.resolved_by = resolved_by
        self.resolution_note = note

    def timeout(self) -> None:
        """Mark the request as timed out when no human responds in time."""
        self.status = ApprovalStatus.TIMED_OUT
        self.resolved_at = datetime.now(UTC)


class ReflectionEntry(BaseModel):
    """A structured reflection recorded after a significant execution event."""

    reflection_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    step_id: str | None = Field(None, description="The step this reflection is linked to.")
    agent_role: AgentRole
    summary: str = Field(..., description="What happened and what was learned.")
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Concrete follow-up actions derived from the reflection.",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the reflection (0 = low, 1 = high).",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TokenUsage(BaseModel):
    """Accumulated LLM token consumption for cost/quota tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        """Add token counts from a single LLM call."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


# ---------------------------------------------------------------------------
# Root state model
# ---------------------------------------------------------------------------


class TaskState(BaseModel):
    """
    The complete, mutable runtime state of a Kodiak agent task.

    This model is the single source of truth shared across all orchestration
    components (supervisor, scheduler, planner, reflection loop, approval
    gate, etc.).  It is designed to be serialisable so it can be persisted
    to a database or event store between orchestration ticks.
    """

    # Identity
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this specific execution attempt.",
    )
    repository_id: str | None = Field(
        None, description="The GitHub repository this task is scoped to."
    )
    pull_request_id: int | None = None
    issue_number: int | None = None

    # User-facing description
    title: str = Field(..., description="Short human-readable title of the task.")
    objective: str = Field(..., description="Full description of what the agent must achieve.")
    tags: list[str] = Field(default_factory=list)

    # Lifecycle
    status: TaskStatus = Field(TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Plan & execution
    steps: list[ExecutionStep] = Field(default_factory=list)
    current_step_index: int | None = Field(
        None, description="Index of the step currently being executed."
    )

    # Approval gates
    pending_approval: ApprovalRequest | None = Field(
        None, description="Active approval request blocking further execution."
    )
    approval_history: list[ApprovalRequest] = Field(default_factory=list)

    # Reflections
    reflections: list[ReflectionEntry] = Field(default_factory=list)

    # Memory / context
    context_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value context packed by the context manager.",
    )
    working_memory: dict[str, Any] = Field(
        default_factory=dict,
        description="Ephemeral key-value store for inter-step data passing.",
    )

    # Metrics
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    retry_count: int = 0
    max_retries: int = 3

    # Final output
    result: str | None = Field(
        None, description="Final textual result produced when the task completes."
    )
    error: str | None = Field(None, description="Top-level error message if the task failed.")

    # Arbitrary metadata (agent versions, feature flags, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("steps")
    @classmethod
    def steps_indices_are_consistent(cls, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        """Ensure step indices match their position in the list."""
        for i, step in enumerate(steps):
            if step.index != i:
                raise ValueError(f"Step at position {i} has index={step.index}; expected {i}.")
        return steps

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Transition the task to the running state."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, result: str) -> None:
        """Mark the task as successfully completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.finished_at = datetime.now(UTC)

    def fail(self, error: str) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.finished_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Cancel the task, regardless of current state."""
        self.status = TaskStatus.CANCELLED
        self.finished_at = datetime.now(UTC)

    def transition_to(self, new_status: TaskStatus) -> None:
        """Validate and apply a lifecycle status transition."""
        transition_task_status(self.status, new_status)
        self.status = new_status
        if new_status is TaskStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now(UTC)
        if new_status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            self.finished_at = datetime.now(UTC)

    def pause(self) -> None:
        """Pause execution (e.g. waiting for an external signal)."""
        self.status = TaskStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused task."""
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING

    # --- Steps ---

    @property
    def current_step(self) -> ExecutionStep | None:
        """Return the step currently being executed, if any."""
        if self.current_step_index is not None and self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def completed_steps(self) -> list[ExecutionStep]:
        """Return all steps that have reached the COMPLETED state."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    @property
    def failed_steps(self) -> list[ExecutionStep]:
        """Return all steps that have reached the FAILED state."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    @property
    def pending_steps(self) -> list[ExecutionStep]:
        """Return all steps still waiting to be executed."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def advance_step(self) -> ExecutionStep | None:
        """
        Move to the next pending step and return it.

        Returns None if no more pending steps are available.
        """
        for i, step in enumerate(self.steps):
            if step.status == StepStatus.PENDING:
                self.current_step_index = i
                return step
        self.current_step_index = None
        return None

    def get_step(self, step_id: str) -> ExecutionStep | None:
        """Look up a step by its ID."""
        return next((s for s in self.steps if s.step_id == step_id), None)

    # --- Approval ---

    def request_approval(
        self, reason: str, context: dict[str, Any] | None = None
    ) -> ApprovalRequest:
        """
        Create and register a new approval gate.

        Transitions the task status to AWAITING_APPROVAL and stores the
        request as the active pending approval.
        """
        request = ApprovalRequest(reason=reason, context=context or {})
        self.pending_approval = request
        self.status = TaskStatus.AWAITING_APPROVAL
        return request

    def resolve_approval(self, approval: ApprovalRequest) -> None:
        """
        Move a resolved approval from pending to history.

        If approved, the task resumes; if rejected or timed out, it fails.
        """
        self.approval_history.append(approval)
        self.pending_approval = None

        if approval.status == ApprovalStatus.APPROVED:
            self.status = TaskStatus.RUNNING
        elif approval.status in {ApprovalStatus.REJECTED, ApprovalStatus.TIMED_OUT}:
            self.fail(f"Approval {approval.status.value}: {approval.resolution_note or ''}")

    # --- Reflections ---

    def add_reflection(self, entry: ReflectionEntry) -> None:
        """Append a reflection entry to the task's reflection log."""
        self.reflections.append(entry)

    # --- Working memory ---

    def set_memory(self, key: str, value: Any) -> None:
        """Write a value into the ephemeral working memory store."""
        self.working_memory[key] = value

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Read a value from the ephemeral working memory store."""
        return self.working_memory.get(key, default)

    def clear_memory(self) -> None:
        """Wipe the ephemeral working memory (e.g. between planning phases)."""
        self.working_memory.clear()

    # --- Metrics ---

    @property
    def is_terminal(self) -> bool:
        """Return True if the task has reached a terminal state."""
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }

    @property
    def duration_ms(self) -> int | None:
        """Total wall-clock duration in milliseconds, or None if not finished."""
        if self.started_at and self.finished_at:
            delta = self.finished_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    @property
    def progress_pct(self) -> float:
        """
        Rough completion percentage based on step statuses.

        Returns a value between 0.0 and 100.0.
        """
        if not self.steps:
            return 0.0
        terminal = sum(1 for s in self.steps if s.is_terminal)
        return round(terminal / len(self.steps) * 100, 1)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary dict suitable for logging or API responses."""
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "title": self.title,
            "status": self.status.value,
            "progress_pct": self.progress_pct,
            "steps_total": len(self.steps),
            "steps_completed": len(self.completed_steps),
            "steps_failed": len(self.failed_steps),
            "duration_ms": self.duration_ms,
            "token_usage": {
                "prompt": self.token_usage.prompt_tokens,
                "completion": self.token_usage.completion_tokens,
                "total": self.token_usage.total_tokens,
            },
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Legacy scheduler / supervisor compatibility models
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Runtime state for a single agent tracked by the legacy supervisor."""

    name: str
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: uuid.UUID | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0


class ScheduledTaskRecord(BaseModel):
    """Lightweight task record used by the scheduler and reflection loop."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    agent_type: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[uuid.UUID] = Field(default_factory=list)
    retry_count: int = 0
    error: str | None = None
    completed_at: datetime | None = None


class ExecutionState(BaseModel):
    """Shared mutable state for scheduler, supervisor, and reflection loop."""

    tasks: list[ScheduledTaskRecord] = Field(default_factory=list)
    agents: dict[str, AgentState] = Field(default_factory=dict)
    current_task_id: uuid.UUID | None = None

    def get_task(self, task_id: uuid.UUID) -> ScheduledTaskRecord | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    def update_task(self, task_id: uuid.UUID, **fields: Any) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, value)
