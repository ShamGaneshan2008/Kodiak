"""Verification subsystem for autonomous task execution.

All types that were previously spread across the ``verification/`` package
(VerificationEngine, Verifier, VerificationEvidence, VerificationContext,
and the four built-in verifiers) are consolidated here so that both
``autonomous_loop.py`` and ``execution/engine.py`` can import from a
single, non-ambiguous module.
"""

from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import (
    ExecutionContext,
    ExecutionOutcome,
    ExecutionResult,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VerificationStatus(enum.StrEnum):
    """Outcome of verifying whether a task actually succeeded."""

    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Evidence (from package)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# VerificationResult — unified API
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Aggregated verification outcome for a task execution.

    Supports both the simple API (``.message``, ``.model_dump()``) used by
    ``autonomous_loop.py`` / ``reflection.py`` and the richer fields
    (``.evidence``, ``.duration_seconds``, ``.retry_recommended``,
    ``.to_dict()``) used by ``execution/engine.py`` and the verification
    engine.
    """

    status: VerificationStatus
    message: str = ""
    evidence: dict[str, Any] | tuple[VerificationEvidence, ...] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    summary: str | None = None
    retry_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        if isinstance(self.evidence, tuple):
            evidence_list = [item.to_dict() for item in self.evidence]
        else:
            evidence_list = (
                self.evidence.get("evidence", []) if isinstance(self.evidence, dict) else []
            )
        return {
            "status": self.status.value,
            "summary": self.summary or self.message,
            "duration_seconds": self.duration_seconds,
            "retry_recommended": self.retry_recommended,
            "evidence": evidence_list,
        }


# ---------------------------------------------------------------------------
# VerificationContext (from package)
# ---------------------------------------------------------------------------


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
            message="No verification evidence was collected.",
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
            message=summary,
            summary=summary,
            retry_recommended=True,
        )

    if all(item.status is VerificationStatus.VERIFIED for item in evidence):
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            evidence=tuple(evidence),
            duration_seconds=duration,
            message="All configured verifiers passed.",
            summary="All configured verifiers passed.",
        )

    return VerificationResult(
        status=VerificationStatus.INCONCLUSIVE,
        evidence=tuple(evidence),
        duration_seconds=duration,
        message="Verification produced mixed or insufficient evidence.",
        summary="Verification produced mixed or insufficient evidence.",
        retry_recommended=True,
    )


# ---------------------------------------------------------------------------
# Verifier base class and implementations
# ---------------------------------------------------------------------------


class Verifier(ABC):
    """Evaluates one aspect of whether a task actually succeeded."""

    name: str

    def applies(self, context: VerificationContext) -> bool:
        """Return True when this verifier should run for the given context."""
        return True

    @abstractmethod
    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        """Run verification and return structured evidence."""


def _summarize(text: str | None, limit: int = 500) -> str | None:
    if not text:
        return None
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 3] + "..."


class OutputVerifier(Verifier):
    """Validate required fields and types in agent output."""

    name = "output"

    def applies(self, context: VerificationContext) -> bool:
        criteria = context.success_criteria
        return bool(
            criteria.get("required_output_fields")
            or criteria.get("required_fields")
            or criteria.get("output_schema")
        )

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        start = time.monotonic()
        criteria = context.success_criteria
        required = criteria.get("required_output_fields") or criteria.get("required_fields") or []
        output = context.agent_output

        missing = [field for field in required if field not in output]
        if missing:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=f"Missing required output fields: {', '.join(missing)}",
                metadata={"missing_fields": missing, "output_keys": sorted(output.keys())},
            )

        invalid = self._validate_schema(output, criteria.get("output_schema"))
        if invalid:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=invalid,
                metadata={"output_keys": sorted(output.keys())},
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.VERIFIED,
            duration_seconds=time.monotonic() - start,
            message="Agent output contains all required fields.",
            metadata={"validated_fields": list(required)},
        )

    @staticmethod
    def _validate_schema(output: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
        if not schema:
            return None
        properties = schema.get("properties", {})
        for fld, spec in properties.items():
            if fld not in output:
                continue
            expected_type = spec.get("type")
            value = output[fld]
            if expected_type == "object" and not isinstance(value, dict):
                return f"Field {fld!r} expected object, got {type(value).__name__}"
            if expected_type == "array" and not isinstance(value, list):
                return f"Field {fld!r} expected array, got {type(value).__name__}"
            if expected_type == "string" and not isinstance(value, str):
                return f"Field {fld!r} expected string, got {type(value).__name__}"
            if expected_type == "integer" and not isinstance(value, int):
                return f"Field {fld!r} expected integer, got {type(value).__name__}"
        return None


class FileVerifier(Verifier):
    """Verify expected files exist and unexpected files were not modified."""

    name = "file"

    def applies(self, context: VerificationContext) -> bool:
        criteria = context.success_criteria
        return bool(
            criteria.get("expected_files")
            or criteria.get("required_artifacts")
            or criteria.get("unexpected_files")
        )

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        start = time.monotonic()
        root = context.workspace_root or Path.cwd()
        criteria = context.success_criteria

        expected = list(criteria.get("expected_files") or [])
        artifacts = list(criteria.get("required_artifacts") or [])
        unexpected = set(criteria.get("unexpected_files") or [])

        checked: list[str] = []
        missing: list[str] = []

        for rel_path in expected + artifacts:
            path = Path(rel_path)
            if not path.is_absolute():
                path = root / path
            checked.append(str(path))
            if not path.exists():
                missing.append(str(rel_path))

        if missing:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=f"Missing expected files: {', '.join(missing)}",
                files_checked=tuple(checked),
                artifacts_checked=tuple(artifacts),
            )

        changed_unexpected = [rel for rel in unexpected if (root / rel).exists()]
        if changed_unexpected:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=f"Unexpected files present: {', '.join(changed_unexpected)}",
                files_checked=tuple(checked),
                metadata={"unexpected_files": changed_unexpected},
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.VERIFIED,
            duration_seconds=time.monotonic() - start,
            message="All expected files are present.",
            files_checked=tuple(checked),
            artifacts_checked=tuple(artifacts),
        )


class TestVerifier(Verifier):
    """Run configured tests through the existing ToolRouter boundary."""

    name = "test"

    def __init__(self, tool_router: Any | None = None) -> None:
        self._tool_router = tool_router

    def applies(self, context: VerificationContext) -> bool:
        return bool(context.success_criteria.get("run_tests"))

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        from kodiak.tools.models import ToolExecutionContext

        start = time.monotonic()
        config = context.success_criteria.get("run_tests", {})
        if not isinstance(config, dict):
            config = {"test_target": str(config)}

        if self._tool_router is None:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.INCONCLUSIVE,
                duration_seconds=time.monotonic() - start,
                message="Test verification requested but ToolRouter is not configured.",
            )

        test_target = config.get("test_target", "tests")
        options = config.get("options", [])
        inputs: dict[str, Any] = {"test_target": test_target, "options": options}

        tool_context = ToolExecutionContext(
            agent_name="verification",
            task_id=str(context.task.id),
            granted_capabilities=frozenset({"run_tests", "test_execution"}),
            timeout_seconds=config.get("timeout_seconds"),
        )

        result = await self._tool_router.execute("test_runner", inputs, tool_context)
        duration = time.monotonic() - start
        stdout = _summarize(result.output.get("stdout") if result.output else None)
        stderr = _summarize(result.output.get("stderr") if result.output else None)
        exit_code = result.output.get("returncode") if result.output else None

        if result.success:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.VERIFIED,
                duration_seconds=duration,
                message=f"Tests passed for target {test_target!r}.",
                command=f"pytest {test_target}",
                exit_code=exit_code,
                stdout_summary=stdout,
                stderr_summary=stderr,
                metadata={"test_target": test_target},
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.FAILED,
            duration_seconds=duration,
            message=result.error or f"Tests failed for target {test_target!r}.",
            command=f"pytest {test_target}",
            exit_code=exit_code,
            stdout_summary=stdout,
            stderr_summary=stderr,
            metadata={"test_target": test_target},
        )


class CommandVerifier(Verifier):
    """Run allowed validation commands through ToolRouter."""

    name = "command"

    def __init__(self, tool_router: Any | None = None) -> None:
        self._tool_router = tool_router

    def applies(self, context: VerificationContext) -> bool:
        return bool(context.success_criteria.get("commands"))

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        from kodiak.tools.models import ToolExecutionContext

        start = time.monotonic()
        commands = context.success_criteria.get("commands", [])
        if not isinstance(commands, list) or not commands:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.INCONCLUSIVE,
                duration_seconds=time.monotonic() - start,
                message="No commands configured for command verification.",
            )

        if self._tool_router is None:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.INCONCLUSIVE,
                duration_seconds=time.monotonic() - start,
                message="Command verification requested but ToolRouter is not configured.",
            )

        last_failure: VerificationEvidence | None = None
        for entry in commands:
            if isinstance(entry, str):
                command = entry
                args: list[str] = []
            elif isinstance(entry, dict):
                command = str(entry.get("command", ""))
                args = [str(arg) for arg in entry.get("args", [])]
            else:
                continue

            if not command:
                continue

            tool_context = ToolExecutionContext(
                agent_name="verification",
                task_id=str(context.task.id),
                granted_capabilities=frozenset({"command_execution", "terminal"}),
                timeout_seconds=(entry.get("timeout_seconds") if isinstance(entry, dict) else None),
            )
            result = await self._tool_router.execute(
                "command_runner",
                {"command": command, "args": args},
                tool_context,
            )
            command_label = " ".join([command, *args]).strip()
            stdout = _summarize(result.output.get("stdout") if result.output else None)
            stderr = _summarize(result.output.get("stderr") if result.output else None)
            exit_code = result.output.get("returncode") if result.output else None

            if not result.success:
                last_failure = VerificationEvidence(
                    verifier=self.name,
                    status=VerificationStatus.FAILED,
                    duration_seconds=time.monotonic() - start,
                    message=result.error or f"Command failed: {command_label}",
                    command=command_label,
                    exit_code=exit_code,
                    stdout_summary=stdout,
                    stderr_summary=stderr,
                )
                break

        duration = time.monotonic() - start
        if last_failure is not None:
            return VerificationEvidence(
                verifier=last_failure.verifier,
                status=last_failure.status,
                duration_seconds=duration,
                message=last_failure.message,
                command=last_failure.command,
                exit_code=last_failure.exit_code,
                stdout_summary=last_failure.stdout_summary,
                stderr_summary=last_failure.stderr_summary,
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.VERIFIED,
            duration_seconds=duration,
            message="All configured validation commands passed.",
            metadata={"commands_run": len(commands)},
        )


# ---------------------------------------------------------------------------
# VerificationEngine
# ---------------------------------------------------------------------------


def default_verifiers(tool_router: Any | None = None) -> list[Verifier]:
    """Return the standard set of verification strategies."""
    return [
        OutputVerifier(),
        FileVerifier(),
        TestVerifier(tool_router=tool_router),
        CommandVerifier(tool_router=tool_router),
    ]


class VerificationEngine:
    """Evaluates whether an agent execution actually satisfied the task."""

    def __init__(
        self,
        verifiers: list[Verifier] | None = None,
        tool_router: Any | None = None,
    ) -> None:
        self._tool_router = tool_router
        self._verifiers = verifiers if verifiers is not None else default_verifiers(tool_router)
        self._logger = logger.bind(component="verification_engine")

    @staticmethod
    def should_verify(task: Task) -> bool:
        """Return True when the task defines verification criteria."""
        verification = task.context.get("verification")
        return isinstance(verification, dict) and bool(verification)

    async def verify(
        self,
        task: Task | None = None,
        execution_result: ExecutionResult | None = None,
        *,
        execution_context: ExecutionContext | None = None,
        goal: str | None = None,
        plan: Any = None,
        task_state: Any = None,
    ) -> VerificationResult:
        """Run applicable verifiers and aggregate evidence."""
        if task is None:
            if execution_result is None or task_state is None:
                raise TypeError(
                    "verify() requires either a Task plus ExecutionResult, or "
                    "execution_result and task_state for autonomous-loop verification."
                )
            return await TaskVerifier().verify(
                goal=goal or getattr(task_state, "objective", ""),
                plan=plan,
                execution_result=execution_result,
                task_state=task_state,
            )

        if execution_result is None:
            raise TypeError("verify() missing required argument: 'execution_result'")

        context = VerificationContext.from_execution(
            task,
            execution_result,
            execution_context=execution_context,
        )
        log = self._logger.bind(task_id=str(task.id))
        started = time.monotonic()

        if not context.execution_succeeded:
            result = VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                message="Agent execution did not succeed; verification skipped.",
                duration_seconds=time.monotonic() - started,
            )
            log.info("verification_skipped_execution_failed", status=result.status.value)
            return result

        active = [verifier for verifier in self._verifiers if verifier.applies(context)]
        if not active:
            result = VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                message="No applicable verifiers for configured criteria.",
                duration_seconds=time.monotonic() - started,
                retry_recommended=True,
            )
            log.info("verification_inconclusive_no_verifiers", status=result.status.value)
            return result

        evidence: list[VerificationEvidence] = []
        for verifier in active:
            log.info("verification_started", verifier=verifier.name)
            item = await verifier.verify(context)
            evidence.append(item)
            log.info(
                "verification_completed",
                verifier=verifier.name,
                status=item.status.value,
                duration_seconds=item.duration_seconds,
            )

        agg = aggregate_evidence(evidence)
        result = VerificationResult(
            status=agg.status,
            evidence=evidence,
            duration_seconds=time.monotonic() - started,
            message=agg.message,
            summary=agg.summary,
            retry_recommended=agg.retry_recommended,
        )
        log.info(
            "verification_finished",
            status=result.status.value,
            verifier_count=len(evidence),
            duration_seconds=result.duration_seconds,
        )
        return result


# ---------------------------------------------------------------------------
# TaskVerifier — simple API for autonomous_loop.py
# ---------------------------------------------------------------------------


class TaskVerifier:
    """Evaluate whether an execution outcome satisfies the planned task goal.

    This is the simpler verifier used by ``autonomous_loop.py``. It does
    not use the full VerificationEngine pipeline.
    """

    async def verify(
        self,
        *,
        goal: str,
        plan: Any,
        execution_result: ExecutionResult,
        task_state: Any,
    ) -> VerificationResult:
        logger.info(
            "verification_started",
            task_id=task_state.task_id,
            execution_id=task_state.run_id,
            outcome=execution_result.outcome.value,
        )

        if execution_result.outcome is not ExecutionOutcome.SUCCESS:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                message="Execution did not complete successfully.",
                evidence={
                    "outcome": execution_result.outcome.value,
                    "error": execution_result.error,
                },
            )

        output = execution_result.result or {}
        explicit_status = str(output.get("verification_status", "")).lower()
        if explicit_status == "failed":
            return VerificationResult(
                status=VerificationStatus.FAILED,
                message=str(output.get("verification_message", "Explicit verification failure.")),
                evidence={"output": output},
            )
        if explicit_status == "inconclusive":
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                message=str(output.get("verification_message", "Verification inconclusive.")),
                evidence={"output": output},
            )

        if not output and not task_state.result:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                message="Execution completed without output evidence.",
                evidence={"goal": goal},
            )

        if plan and hasattr(plan, "acceptance_criteria") and plan.acceptance_criteria:
            missing = [
                criterion
                for criterion in plan.acceptance_criteria
                if criterion.lower() not in str(output).lower()
                and criterion.lower() not in goal.lower()
            ]
            if missing and output.get("strict_acceptance", False):
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    message="Acceptance criteria not satisfied.",
                    evidence={"missing_criteria": missing, "output": output},
                )

        logger.info(
            "verification_completed",
            task_id=task_state.task_id,
            status=VerificationStatus.VERIFIED.value,
        )
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            message="Task execution verified.",
            evidence={"output": output, "goal": goal},
        )


__all__ = [
    # Enums
    "VerificationStatus",
    # Evidence
    "VerificationEvidence",
    # Result
    "VerificationResult",
    # Context
    "VerificationContext",
    "aggregate_evidence",
    # Verifier base + implementations
    "Verifier",
    "OutputVerifier",
    "FileVerifier",
    "TestVerifier",
    "CommandVerifier",
    # Engines
    "VerificationEngine",
    "TaskVerifier",
    "default_verifiers",
]
