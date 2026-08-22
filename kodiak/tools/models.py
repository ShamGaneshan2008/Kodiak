"""Domain models and data structures for the Tool & Capability System."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kodiak.orchestration.execution.models import ExecutionResult


class PermissionLevel(StrEnum):
    """Permission boundaries for tool execution."""

    ALLOWED = "allowed"
    DENIED = "denied"
    RESTRICTED = "restricted"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Strongly typed representation and metadata of a tool.

    Attributes:
        name: Stable identifier of the tool.
        description: Clear explanation of what the tool does.
        version: Version string for compatibility and tracking.
        input_schema: Optional JSON schema or parameter specification for input validation.
        output_schema: Optional JSON schema or output specification.
        capabilities: Set of capability strings associated with or satisfied by this tool.
        permissions: Default permission level required for this tool.
        execution_metadata: Additional operational metadata.
        timeout_seconds: Optional execution time cap.
        is_async: Whether the tool executes asynchronously.
        is_safe: Whether the tool is considered safe for automated invocation.
        is_restricted: Whether access to the tool is strictly restricted to privileged agents.
    """

    name: str
    description: str
    version: str = "1.0.0"
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    permissions: PermissionLevel = PermissionLevel.ALLOWED
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    is_async: bool = True
    is_safe: bool = True
    is_restricted: bool = False


@dataclass(slots=True)
class ToolExecutionContext:
    """Contextual information provided to a tool during invocation.

    Attributes:
        agent_name: Identity or role of the invoking agent.
        task_id: Unique identity of the task.
        plan_id: Identity of the overall plan.
        plan_step_id: Identity of the specific plan step.
        permissions: Set of permissions granted in this context.
        granted_capabilities: Set of capabilities possessed by the calling agent.
        timeout_seconds: Request-specific timeout override.
        correlation_id: Unique correlation identifier for tracing.
        metadata: Extra contextual key-value pairs.
    """

    agent_name: str | None = None
    task_id: str | None = None
    plan_id: str | None = None
    plan_step_id: str | None = None
    permissions: frozenset[PermissionLevel] = field(
        default_factory=lambda: frozenset({PermissionLevel.ALLOWED})
    )
    granted_capabilities: frozenset[str] = field(default_factory=frozenset)
    timeout_seconds: float | None = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by a tool invocation.

    Attributes:
        success: True if tool execution succeeded without unhandled error.
        output: Structured output dictionary or payload produced by the tool.
        error: Detailed error message if execution failed.
        execution_metadata: Diagnostic metadata captured during tool run.
        duration_seconds: Monotonic runtime of the invocation.
        tool_name: Stable identifier of the executed tool.
        correlation_id: Correlation identifier matching the execution context.
    """

    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    tool_name: str = ""
    correlation_id: str | None = None

    def to_execution_result(self, task_id: str | None = None) -> ExecutionResult:
        """Convert this ToolResult into Kodiak's canonical ExecutionResult."""
        from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult

        tid = task_id or (self.correlation_id or "tool_task")
        outcome = ExecutionOutcome.SUCCESS if self.success else ExecutionOutcome.FAILURE
        error_dict = {"message": self.error} if self.error else None
        return ExecutionResult(
            task_id=tid,
            outcome=outcome,
            attempts=1,
            duration_seconds=self.duration_seconds,
            result=self.output,
            error=error_dict,
        )


@dataclass(frozen=True, slots=True)
class Capability:
    """Representation of an agent or tool capability.

    Attributes:
        identifier: Unique capability identifier (e.g. 'write_code', 'run_tests').
        description: Human readable capability description.
        supported_tools: Identifiers of tools that support or implement this capability.
        metadata: Arbitrary capability metadata.
    """

    identifier: str
    description: str
    supported_tools: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PermissionLevel",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolResult",
    "Capability",
]
