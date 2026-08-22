"""Domain models for task verification."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import (
    ExecutionContext,
    ExecutionOutcome,
    ExecutionResult,
)


class VerificationStatus(enum.StrEnum):
    """Outcome of verifying whether a task actually succeeded."""

    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Structured evidence produced by a single verifier."""

    verifier: str
    status: VerificationStatus
    duration_seconds: float = 0.0
    message: str | None = None
    command: str | None = None
    exit_code: int | None = None
    stdout_summary: str | None = None
    stderr_summary: str | None = None
    files_checked: tuple[str, ...] = field(default_factory=tuple)
    artifacts_checked: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "verifier": self.verifier,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_summary": self.stdout_summary,
            "stderr_summary": self.stderr_summary,
            "files_checked": list(self.files_checked),
            "artifacts_checked": list(self.artifacts_checked),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Aggregated verification outcome for a task execution."""

    status: VerificationStatus
    evidence: tuple[VerificationEvidence, ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0
    summary: str | None = None
    retry_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "status": self.status.value,
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "retry_recommended": self.retry_recommended,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class VerificationContext:
    """Inputs available to verification strategies."""

    task: Task
    execution_result: ExecutionResult
    execution_context: ExecutionContext | None = None
    workspace_root: Path | None = None
    success_criteria: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution(
        cls,
        task: Task,
        execution_result: ExecutionResult,
        *,
        execution_context: ExecutionContext | None = None,
        workspace_root: Path | None = None,
    ) -> VerificationContext:
        """Build verification context from a completed execution."""
        criteria = dict(task.context.get("verification", {}))
        root = workspace_root
        if root is None:
            raw_root = criteria.get("workspace_root") or task.context.get("repository_path")
            if raw_root:
                root = Path(str(raw_root))
        return cls(
            task=task,
            execution_result=execution_result,
            execution_context=execution_context,
            workspace_root=root,
            success_criteria=criteria,
        )

    @property
    def agent_output(self) -> dict[str, Any]:
        """Agent output payload from the execution result."""
        return dict(self.execution_result.result or {})

    @property
    def execution_succeeded(self) -> bool:
        """Whether the agent execution reported success."""
        return self.execution_result.outcome is ExecutionOutcome.SUCCESS


def aggregate_evidence(evidence: list[VerificationEvidence]) -> VerificationResult:
    """Combine verifier evidence into a single verification result."""
    if not evidence:
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            summary="No verification evidence was collected.",
            retry_recommended=True,
        )

    duration = sum(item.duration_seconds for item in evidence)
    statuses = {item.status for item in evidence}

    if VerificationStatus.FAILED in statuses:
        failed = [item for item in evidence if item.status is VerificationStatus.FAILED]
        summary = failed[0].message or f"{len(failed)} verifier(s) failed."
        return VerificationResult(
            status=VerificationStatus.FAILED,
            evidence=tuple(evidence),
            duration_seconds=duration,
            summary=summary,
            retry_recommended=True,
        )

    if all(item.status is VerificationStatus.VERIFIED for item in evidence):
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            evidence=tuple(evidence),
            duration_seconds=duration,
            summary="All configured verifiers passed.",
        )

    return VerificationResult(
        status=VerificationStatus.INCONCLUSIVE,
        evidence=tuple(evidence),
        duration_seconds=duration,
        summary="Verification produced mixed or insufficient evidence.",
        retry_recommended=True,
    )


__all__ = [
    "VerificationContext",
    "VerificationEvidence",
    "VerificationResult",
    "VerificationStatus",
    "aggregate_evidence",
]
