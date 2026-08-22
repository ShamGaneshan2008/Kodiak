"""Domain models for reflection and self-repair."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import (
    ExecutionContext,
    ExecutionResult,
)
from kodiak.orchestration.verification.models import VerificationResult


class ReflectionOutcome(enum.StrEnum):
    """High-level reflection decision."""

    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    NON_RETRYABLE_FAILURE = "non_retryable_failure"
    REPLAN_REQUIRED = "replan_required"
    MAX_RETRIES_REACHED = "max_retries_reached"


class FailureCategory(enum.StrEnum):
    """Structured failure classification derived from evidence."""

    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    TYPE_ERROR = "type_error"
    LINT_FAILURE = "lint_failure"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    PERMISSION_FAILURE = "permission_failure"
    TIMEOUT = "timeout"
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"
    MISSING_ARTIFACT = "missing_artifact"
    INCORRECT_IMPLEMENTATION = "incorrect_implementation"
    EXECUTION_FAILURE = "execution_failure"
    UNKNOWN = "unknown"


class RepairStrategy(enum.StrEnum):
    """Recommended next action after reflection."""

    RETRY = "retry"
    REPAIR = "repair"
    REPLAN = "replan"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Structured output from the reflection subsystem."""

    outcome: ReflectionOutcome
    category: FailureCategory
    strategy: RepairStrategy
    root_cause: str
    suggested_correction: str
    evidence: dict[str, Any] = field(default_factory=dict)
    affected_files: tuple[str, ...] = field(default_factory=tuple)
    affected_tool: str | None = None
    confidence: float = 0.5
    should_retry: bool = False
    replan_required: bool = False
    attempt: int = 1
    max_attempts: int = 1
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "outcome": self.outcome.value,
            "category": self.category.value,
            "strategy": self.strategy.value,
            "root_cause": self.root_cause,
            "suggested_correction": self.suggested_correction,
            "evidence": dict(self.evidence),
            "affected_files": list(self.affected_files),
            "affected_tool": self.affected_tool,
            "confidence": self.confidence,
            "should_retry": self.should_retry,
            "replan_required": self.replan_required,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "summary": self.summary,
        }

    def to_memory_payload(self) -> dict[str, Any]:
        """Payload suitable for future memory persistence."""
        return {
            "failure_category": self.category.value,
            "root_cause": self.root_cause,
            "correction": self.suggested_correction,
            "strategy": self.strategy.value,
            "evidence": dict(self.evidence),
            "affected_files": list(self.affected_files),
        }


@dataclass(slots=True)
class ReflectionContext:
    """Inputs for analyzing a failed or inconclusive execution."""

    task: Task
    execution_result: ExecutionResult
    execution_context: ExecutionContext | None = None
    verification_result: VerificationResult | None = None
    attempt: int = 1
    max_attempts: int = 1

    @classmethod
    def from_execution(
        cls,
        task: Task,
        execution_result: ExecutionResult,
        *,
        execution_context: ExecutionContext | None = None,
        verification_result: VerificationResult | None = None,
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> ReflectionContext:
        """Build reflection context from execution artifacts."""
        return cls(
            task=task,
            execution_result=execution_result,
            execution_context=execution_context,
            verification_result=verification_result,
            attempt=attempt,
            max_attempts=max(1, max_attempts),
        )


__all__ = [
    "FailureCategory",
    "ReflectionContext",
    "ReflectionOutcome",
    "ReflectionResult",
    "RepairStrategy",
]
