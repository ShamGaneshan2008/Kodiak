"""
kodiak/execution

Autonomous Task Execution: the orchestration layer between the Supervisor
and the Agent Manager.

    Supervisor -> ExecutionEngine -> AgentManager -> ... -> ExecutionResult
"""

from kodiak.orchestration.execution.engine import ExecutionEngine

from kodiak.orchestration.execution.exceptions import (
    ExecutionCancelledError,
    ExecutionEngineError,
    ExecutionTimeoutError,
    NonRetryableExecutionError,
    RetryExhaustedError,
)

from kodiak.orchestration.execution.interfaces import (
    AgentManager,
    TaskRepository,
)

from kodiak.orchestration.execution.models import (
    AgentManagerResult,
    CancellationToken,
    ExecutionContext,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionHook,
    ExecutionOutcome,
    ExecutionProgress,
    ExecutionResult,
    ProgressCallback,
    RetryPolicy,
    outcome_to_task_status,
)

__all__ = [
    "ExecutionEngine",
    "ExecutionCancelledError",
    "ExecutionEngineError",
    "ExecutionTimeoutError",
    "NonRetryableExecutionError",
    "RetryExhaustedError",
    "AgentManager",
    "TaskRepository",
    "AgentManagerResult",
    "CancellationToken",
    "ExecutionContext",
    "ExecutionEvent",
    "ExecutionEventType",
    "ExecutionHook",
    "ExecutionOutcome",
    "ExecutionProgress",
    "ExecutionResult",
    "ProgressCallback",
    "RetryPolicy",
    "outcome_to_task_status",
]