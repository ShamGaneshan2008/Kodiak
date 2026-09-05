"""Reflection and self-repair subsystem.

All types that were previously spread across the ``reflection/`` package
(ReflectionEngine, FailureAnalyzer, SelfRepairLoop, RepairStrategy, etc.)
are consolidated here so that both ``autonomous_loop.py`` (which uses the
file-based API) and ``execution/engine.py`` (which used the package API)
can import from a single, non-ambiguous module.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from kodiak.db.models.task import Task, TaskStatus
from kodiak.orchestration.execution.models import (
    ExecutionContext,
    ExecutionOutcome,
    ExecutionResult,
    RetryPolicy,
)
from kodiak.orchestration.state import TaskState
from kodiak.orchestration.verification import (
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from kodiak.orchestration.execution.engine import ExecutionEngine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReflectionAction(enum.StrEnum):
    """Follow-up action after verification or execution failure."""

    RETRY = "retry"
    REPAIR = "repair"
    REPLAN = "replan"
    STOP = "stop"


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


# ---------------------------------------------------------------------------
# Pydantic-based ReflectionResult (used by autonomous_loop.py)
# ---------------------------------------------------------------------------


class ReflectionResult(BaseModel):
    """Structured output from the lightweight reflection service.

    This is the Pydantic-based result used by ``autonomous_loop.py`` and
    ``reflection.py``'s ``ReflectionService``.
    """

    action: ReflectionAction
    root_cause: str
    retryable: bool = True
    repair_required: bool = False
    replan_required: bool = False
    strategy: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """JSON-safe serialization."""
        return {
            "action": self.action.value,
            "root_cause": self.root_cause,
            "retryable": self.retryable,
            "repair_required": self.repair_required,
            "replan_required": self.replan_required,
            "strategy": self.strategy,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Dataclass-based models (used by ReflectionEngine and FailureAnalyzer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReflectionResultDetailed:
    """Detailed structured output from the reflection engine.

    This is the dataclass-based result used by ``ReflectionEngine`` and
    ``FailureAnalyzer`` for richer failure analysis.
    """

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

    @property
    def action(self) -> RepairStrategy:
        """Compatibility alias for autonomous-loop callers."""
        return self.strategy

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


# ---------------------------------------------------------------------------
# FailureAnalyzer
# ---------------------------------------------------------------------------


class FailureAnalyzer:
    """Derives structured failure diagnosis from available evidence."""

    def analyze(self, context: ReflectionContext) -> ReflectionResultDetailed:
        """Analyze failure evidence and produce a reflection result."""
        if context.execution_result.outcome is ExecutionOutcome.SUCCESS:
            verification = context.verification_result
            if verification is None or verification.status is VerificationStatus.VERIFIED:
                return ReflectionResultDetailed(
                    outcome=ReflectionOutcome.SUCCESS,
                    category=FailureCategory.UNKNOWN,
                    strategy=RepairStrategy.STOP,
                    root_cause="Task completed and verified successfully.",
                    suggested_correction="No corrective action required.",
                    should_retry=False,
                    attempt=context.attempt,
                    max_attempts=context.max_attempts,
                    summary="Success",
                    confidence=1.0,
                )

        evidence = self._collect_evidence(context)
        category = self._categorize(context, evidence)
        root_cause = self._root_cause(context, category, evidence)
        correction = self._suggested_correction(category, evidence)
        strategy, outcome = self._strategy_for(category, context, evidence)

        affected_files = tuple(evidence.get("affected_files", ()))
        affected_tool = evidence.get("affected_tool")

        return ReflectionResultDetailed(
            outcome=outcome,
            category=category,
            strategy=strategy,
            root_cause=root_cause,
            suggested_correction=correction,
            evidence=evidence,
            affected_files=affected_files,
            affected_tool=affected_tool,
            confidence=self._confidence(category, evidence),
            should_retry=strategy is RepairStrategy.RETRY,
            replan_required=strategy is RepairStrategy.REPLAN,
            attempt=context.attempt,
            max_attempts=context.max_attempts,
            summary=evidence.get("summary"),
        )

    def _collect_evidence(self, context: ReflectionContext) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        execution_error = context.execution_result.error or {}
        evidence["execution_error"] = execution_error
        evidence["execution_outcome"] = context.execution_result.outcome.value

        verification_dict = context.execution_result.verification
        if verification_dict:
            evidence["verification"] = verification_dict
        elif context.verification_result is not None:
            evidence["verification"] = context.verification_result.to_dict()

        verification = evidence.get("verification", {})
        failed_items = [
            item
            for item in verification.get("evidence", [])
            if item.get("status") == VerificationStatus.FAILED.value
        ]
        if failed_items:
            evidence["failed_verifiers"] = failed_items
            evidence["summary"] = failed_items[0].get("message") or verification.get("summary")
            files: list[str] = []
            for item in failed_items:
                files.extend(item.get("files_checked", []))
                files.extend(item.get("artifacts_checked", []))
            if files:
                evidence["affected_files"] = tuple(dict.fromkeys(files))
            evidence["affected_tool"] = failed_items[0].get("verifier")
            evidence["stdout_summary"] = failed_items[0].get("stdout_summary")
            evidence["stderr_summary"] = failed_items[0].get("stderr_summary")
        elif execution_error:
            evidence["summary"] = execution_error.get("message", str(execution_error))

        return evidence

    def _categorize(self, context: ReflectionContext, evidence: dict[str, Any]) -> FailureCategory:
        text = " ".join(
            filter(
                None,
                [
                    str(evidence.get("summary", "")),
                    str((evidence.get("execution_error") or {}).get("message", "")),
                    str(evidence.get("stdout_summary", "")),
                    str(evidence.get("stderr_summary", "")),
                ],
            )
        ).lower()

        failed_verifiers = evidence.get("failed_verifiers") or []
        if failed_verifiers:
            verifier = failed_verifiers[0].get("verifier", "")
            if verifier == "test":
                return FailureCategory.TEST_FAILURE
            if verifier == "file":
                return FailureCategory.MISSING_ARTIFACT
            if verifier == "output":
                return FailureCategory.INCORRECT_IMPLEMENTATION
            if verifier == "command":
                if "lint" in text or "ruff" in text:
                    return FailureCategory.LINT_FAILURE
                if "mypy" in text:
                    return FailureCategory.TYPE_ERROR
                return FailureCategory.EXECUTION_FAILURE

        if context.execution_result.outcome is ExecutionOutcome.TIMEOUT:
            return FailureCategory.TIMEOUT
        if "permission" in text or "denied" in text:
            return FailureCategory.PERMISSION_FAILURE
        if "timeout" in text or "timed out" in text:
            return FailureCategory.TIMEOUT
        if "syntax" in text or "syntaxerror" in text:
            return FailureCategory.SYNTAX_ERROR
        if "pytest" in text or "test" in text or "assert" in text:
            return FailureCategory.TEST_FAILURE
        if "type error" in text or "typeerror" in text:
            return FailureCategory.TYPE_ERROR
        if "modulenotfounderror" in text or "missing dependency" in text:
            return FailureCategory.MISSING_DEPENDENCY
        if "invalid" in text and "argument" in text:
            return FailureCategory.INVALID_TOOL_ARGUMENTS
        if context.execution_result.outcome is ExecutionOutcome.FAILURE:
            return FailureCategory.EXECUTION_FAILURE
        return FailureCategory.UNKNOWN

    def _root_cause(
        self,
        context: ReflectionContext,
        category: FailureCategory,
        evidence: dict[str, Any],
    ) -> str:
        summary = evidence.get("summary")
        if summary:
            if category is FailureCategory.TEST_FAILURE:
                return f"Tests did not pass: {summary}"
            if category is FailureCategory.MISSING_ARTIFACT:
                return f"Expected artifact or file missing: {summary}"
            if category is FailureCategory.INCORRECT_IMPLEMENTATION:
                return f"Agent output did not satisfy requirements: {summary}"
            return str(summary)

        error_message = (evidence.get("execution_error") or {}).get("message")
        if error_message:
            return str(error_message)

        return f"Execution ended with outcome {context.execution_result.outcome.value}."

    def _suggested_correction(
        self,
        category: FailureCategory,
        evidence: dict[str, Any],
    ) -> str:
        mapping: dict[FailureCategory, str] = {
            FailureCategory.TEST_FAILURE: (
                "Inspect failing tests, fix implementation, and rerun focused tests."
            ),
            FailureCategory.MISSING_ARTIFACT: (
                "Create or restore the missing artifact before retrying."
            ),
            FailureCategory.INCORRECT_IMPLEMENTATION: (
                "Adjust the implementation to satisfy required outputs."
            ),
            FailureCategory.SYNTAX_ERROR: ("Fix syntax errors in affected files before retrying."),
            FailureCategory.TYPE_ERROR: ("Resolve type errors and rerun type checks."),
            FailureCategory.LINT_FAILURE: ("Fix lint violations and rerun lint validation."),
            FailureCategory.PERMISSION_FAILURE: (
                "Adjust permissions or use an authorized agent config."
            ),
            FailureCategory.TIMEOUT: ("Reduce scope or increase timeout, retry narrower."),
            FailureCategory.MISSING_DEPENDENCY: (
                "Install or declare the missing dependency, then retry."
            ),
            FailureCategory.INVALID_TOOL_ARGUMENTS: (
                "Correct tool inputs and retry the operation."
            ),
            FailureCategory.EXECUTION_FAILURE: ("Review execution error details and apply a fix."),
        }
        base = mapping.get(category, "Review failure evidence and apply a targeted correction.")
        affected = evidence.get("affected_files")
        if affected:
            return f"{base} Affected files: {', '.join(affected)}."
        return base

    def _strategy_for(
        self,
        category: FailureCategory,
        context: ReflectionContext,
        evidence: dict[str, Any],
    ) -> tuple[RepairStrategy, ReflectionOutcome]:
        if context.attempt >= context.max_attempts:
            return RepairStrategy.STOP, ReflectionOutcome.MAX_RETRIES_REACHED

        non_retryable = {
            FailureCategory.PERMISSION_FAILURE,
            FailureCategory.INVALID_TOOL_ARGUMENTS,
            FailureCategory.MISSING_DEPENDENCY,
        }
        if category in non_retryable:
            return RepairStrategy.STOP, ReflectionOutcome.NON_RETRYABLE_FAILURE

        replan_categories = {FailureCategory.INCORRECT_IMPLEMENTATION}
        if category in replan_categories and context.attempt >= max(2, context.max_attempts - 1):
            return RepairStrategy.REPLAN, ReflectionOutcome.REPLAN_REQUIRED

        repeated_test_failure = category is FailureCategory.TEST_FAILURE and context.attempt >= 2
        if repeated_test_failure:
            return RepairStrategy.REPLAN, ReflectionOutcome.REPLAN_REQUIRED

        if category is FailureCategory.UNKNOWN and context.attempt >= 2:
            return RepairStrategy.REPLAN, ReflectionOutcome.REPLAN_REQUIRED

        return RepairStrategy.RETRY, ReflectionOutcome.RETRYABLE_FAILURE

    @staticmethod
    def _confidence(category: FailureCategory, evidence: dict[str, Any]) -> float:
        if evidence.get("failed_verifiers"):
            return 0.85
        if category is FailureCategory.UNKNOWN:
            return 0.3
        return 0.7


# ---------------------------------------------------------------------------
# ReflectionEngine
# ---------------------------------------------------------------------------


class ReflectionEngine:
    """Analyzes failed executions and recommends repair strategies."""

    def __init__(self, analyzer: FailureAnalyzer | None = None) -> None:
        self._analyzer = analyzer or FailureAnalyzer()
        self._logger = logger.bind(component="reflection_engine")

    @staticmethod
    def should_reflect(task: Task, execution_result: ExecutionResult) -> bool:
        """Return True when reflection should run for this execution."""
        if not execution_result.is_success:
            return True
        verification = execution_result.verification or {}
        status = verification.get("status")
        return status in {"failed", "inconclusive"}

    async def reflect(
        self,
        task: Task,
        execution_result: ExecutionResult,
        *,
        execution_context: ExecutionContext | None = None,
        verification_result: VerificationResult | None = None,
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> ReflectionResultDetailed:
        """Analyze execution/verification outcome and return structured reflection."""
        context = ReflectionContext.from_execution(
            task,
            execution_result,
            execution_context=execution_context,
            verification_result=verification_result,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        started = time.monotonic()
        result = self._analyzer.analyze(context)
        self._logger.info(
            "reflection_completed",
            task_id=str(task.id),
            outcome=result.outcome.value,
            category=result.category.value,
            strategy=result.strategy.value,
            should_retry=result.should_retry,
            duration_seconds=time.monotonic() - started,
        )
        return result


# ---------------------------------------------------------------------------
# SelfRepairLoop
# ---------------------------------------------------------------------------


class SelfRepairLoop:
    """Coordinates execution, verification, reflection, retry, and replanning."""

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        reflection_engine: ReflectionEngine | None = None,
        planner: Any | None = None,
        *,
        max_cycles: int = 3,
    ) -> None:
        self._execution_engine = execution_engine
        self._reflection_engine = reflection_engine or ReflectionEngine()
        self._planner = planner
        self._max_cycles = max(1, max_cycles)
        self._logger = logger.bind(component="self_repair_loop")

    async def run(self, task: Task, **execute_kwargs: Any) -> ExecutionResult:
        """Run the self-repair loop until success, stop, or cycle limit."""
        policy = RetryPolicy.from_task(task)
        max_attempts = policy.max_attempts
        last_result: ExecutionResult | None = None

        for cycle in range(1, self._max_cycles + 1):
            result = await self._execution_engine.execute(task, **execute_kwargs)
            last_result = result

            if result.is_success and not ReflectionEngine.should_reflect(task, result):
                return result

            reflection = await self._reflection_engine.reflect(
                task,
                result,
                attempt=result.attempts,
                max_attempts=max_attempts,
            )
            result = self._attach_reflection(result, reflection.to_dict())

            if reflection.outcome is ReflectionOutcome.SUCCESS:
                return result

            if reflection.strategy is RepairStrategy.STOP:
                return result

            if reflection.strategy is RepairStrategy.REPLAN:
                replanned = await self._maybe_replan(task, result, reflection.to_dict())
                if replanned:
                    self._inject_reflection(task, reflection.to_dict())
                    continue
                return result

            if reflection.strategy is RepairStrategy.RETRY:
                if cycle >= self._max_cycles or result.attempts >= max_attempts:
                    return self._mark_max_retries(result, reflection.to_dict())
                self._inject_reflection(task, reflection.to_dict())
                task.status = TaskStatus.PENDING
                continue

            return result

        if last_result is not None:
            return last_result
        raise RuntimeError("Self-repair loop completed without producing a result.")

    async def _maybe_replan(
        self,
        task: Task,
        result: ExecutionResult,
        reflection_payload: dict[str, Any],
    ) -> bool:
        if self._planner is None:
            return False
        current_plan = task.context.get("execution_plan")
        if current_plan is None:
            return False
        try:
            replan_input = {
                "task_id": result.task_id,
                "error": result.error or {},
                "reflection": reflection_payload,
            }
            updated_plan = await self._planner.replan(current_plan, replan_input)
            task.context["execution_plan"] = updated_plan
            task.context["replanned"] = True
            self._logger.info("self_repair_replan_applied", task_id=result.task_id)
            return True
        except Exception:
            self._logger.exception("self_repair_replan_failed", task_id=result.task_id)
            return False

    @staticmethod
    def _inject_reflection(task: Task, reflection_payload: dict[str, Any]) -> None:
        history = list(task.context.get("reflection_history", []))
        history.append(reflection_payload)
        task.context["reflection_history"] = history
        task.context["reflection"] = reflection_payload
        corrections = task.context.setdefault("correction_context", {})
        corrections.update(
            {
                "root_cause": reflection_payload.get("root_cause"),
                "suggested_correction": reflection_payload.get("suggested_correction"),
                "category": reflection_payload.get("category"),
                "strategy": reflection_payload.get("strategy"),
            }
        )

    @staticmethod
    def _attach_reflection(
        result: ExecutionResult,
        reflection_payload: dict[str, Any],
    ) -> ExecutionResult:
        merged_result = dict(result.result)
        merged_result["reflection"] = reflection_payload
        return ExecutionResult(
            task_id=result.task_id,
            outcome=result.outcome,
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            result=merged_result,
            error=result.error,
            final_status=result.final_status,
            verification=result.verification,
            reflection=reflection_payload,
        )

    @staticmethod
    def _mark_max_retries(
        result: ExecutionResult,
        reflection_payload: dict[str, Any],
    ) -> ExecutionResult:
        reflection_payload = {
            **reflection_payload,
            "outcome": ReflectionOutcome.MAX_RETRIES_REACHED.value,
            "strategy": RepairStrategy.STOP.value,
        }
        return ExecutionResult(
            task_id=result.task_id,
            outcome=ExecutionOutcome.RETRY_EXHAUSTED,
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            result={**result.result, "reflection": reflection_payload},
            error=result.error,
            final_status=TaskStatus.FAILED,
            verification=result.verification,
            reflection=reflection_payload,
        )


# ---------------------------------------------------------------------------
# Lightweight ReflectionService (used by autonomous_loop.py)
# ---------------------------------------------------------------------------


class ReflectionService:
    """Determine follow-up action after verification or execution failure.

    When a ``StrategyMemory`` is provided, the service consults stored
    strategies before making a recovery decision.  This enables the
    system to learn from past successes and failures rather than relying
    purely on rule-based heuristics.
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_replans: int = 2,
        strategy_memory: Any | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._max_replans = max_replans
        self._strategy_memory = strategy_memory

    async def reflect(
        self,
        *,
        task_state: TaskState,
        verification_result: VerificationResult,
        execution_result: ExecutionResult,
        attempt: int,
        replan_count: int,
    ) -> ReflectionResult:
        logger.info(
            "reflection_started",
            task_id=task_state.task_id,
            verification_status=verification_result.status.value,
            attempt=attempt,
        )

        if verification_result.status is VerificationStatus.VERIFIED:
            return ReflectionResult(
                action=ReflectionAction.STOP,
                root_cause="Task already verified.",
                retryable=False,
                evidence={"verification": verification_result.model_dump()},
            )

        error = execution_result.error or {}
        error_type = str(error.get("type", ""))
        error_message = str(error.get("message", ""))
        root_cause = verification_result.message or error_message or "Unknown failure"

        if execution_result.outcome is ExecutionOutcome.FAILURE and (
            error_type.endswith("NonRetryableExecutionError")
            or "non-retryable" in error_message.lower()
        ):
            return ReflectionResult(
                action=ReflectionAction.STOP,
                root_cause=root_cause,
                retryable=False,
                evidence={"error": error},
            )

        if task_state.retry_count >= self._max_retries:
            return ReflectionResult(
                action=ReflectionAction.STOP,
                root_cause="Retry budget exhausted.",
                retryable=False,
                evidence={"retry_count": task_state.retry_count},
            )

        if replan_count >= self._max_replans:
            if task_state.retry_count < self._max_retries:
                return ReflectionResult(
                    action=ReflectionAction.RETRY,
                    root_cause=root_cause,
                    retryable=True,
                    strategy="Retry execution with failure evidence.",
                    evidence={"verification": verification_result.model_dump()},
                )
            return ReflectionResult(
                action=ReflectionAction.STOP,
                root_cause="Replanning budget exhausted.",
                retryable=False,
                evidence={"replan_count": replan_count},
            )

        if verification_result.status is VerificationStatus.INCONCLUSIVE:
            return ReflectionResult(
                action=ReflectionAction.RETRY,
                root_cause=root_cause,
                retryable=True,
                strategy="Retry with additional context.",
                evidence={"verification": verification_result.model_dump()},
            )

        if attempt >= 2 or error.get("replan_required"):
            return ReflectionResult(
                action=ReflectionAction.REPLAN,
                root_cause=root_cause,
                retryable=True,
                replan_required=True,
                strategy="Replan using reflection evidence.",
                evidence={
                    "verification": verification_result.model_dump(),
                    "execution": {
                        "outcome": execution_result.outcome.value,
                        "error": execution_result.error,
                        "result": execution_result.result,
                    },
                },
            )

        if error.get("repair_required") or "repair" in root_cause.lower():
            return ReflectionResult(
                action=ReflectionAction.REPAIR,
                root_cause=root_cause,
                retryable=True,
                repair_required=True,
                strategy="Execute corrective repair action.",
                evidence={"verification": verification_result.model_dump()},
            )

        return ReflectionResult(
            action=ReflectionAction.RETRY,
            root_cause=root_cause,
            retryable=True,
            strategy="Retry execution with failure evidence.",
            evidence={"verification": verification_result.model_dump()},
        )

    async def _query_strategy_memory(
        self,
        failure_category: str,
        root_cause: str,
    ) -> dict[str, Any] | None:
        """Look up relevant strategies from the strategy memory.

        Returns a dict with ``strategy_id``, ``name``, ``approach``,
        and ``action`` if a relevant strategy is found, or ``None``.
        """
        if self._strategy_memory is None:
            return None

        try:
            from kodiak.orchestration.strategy import ProblemClass

            # Map failure category string to ProblemClass
            category_map = {
                "syntax_error": ProblemClass.SYNTAX_ERROR,
                "test_failure": ProblemClass.TEST_FAILURE,
                "type_error": ProblemClass.TYPE_ERROR,
                "lint_failure": ProblemClass.LINT_FAILURE,
                "missing_dependency": ProblemClass.MISSING_DEPENDENCY,
                "permission_failure": ProblemClass.PERMISSION_FAILURE,
                "timeout": ProblemClass.TIMEOUT,
                "incorrect_implementation": ProblemClass.INCORRECT_IMPLEMENTATION,
                "missing_artifact": ProblemClass.MISSING_ARTIFACT,
                "execution_failure": ProblemClass.EXECUTION_FAILURE,
            }
            problem_class = category_map.get(failure_category, ProblemClass.UNKNOWN)

            strategies = self._strategy_memory.retrieve_for_problem(problem_class, limit=3)
            if not strategies:
                return None

            best = strategies[0]
            return {
                "strategy_id": best.strategy_id,
                "name": best.name,
                "approach": best.approach,
                "effectiveness_score": best.effectiveness_score(),
                "use_count": best.use_count,
                "action": ("repair" if best.expected_success_probability >= 0.6 else "replan"),
            }
        except Exception:
            logger.debug("strategy_memory_query_failed", exc_info=True)
            return None

    async def _record_strategy_outcome(
        self,
        strategy_id: str,
        success: bool,
    ) -> None:
        """Record the outcome of a strategy in memory."""
        if self._strategy_memory is None or not strategy_id:
            return
        try:
            from kodiak.orchestration.strategy import StrategyOutcome

            outcome = StrategyOutcome.SUCCESS if success else StrategyOutcome.FAILURE
            self._strategy_memory.record_outcome(strategy_id, outcome)
        except Exception:
            logger.debug("strategy_outcome_record_failed", exc_info=True)


# ---------------------------------------------------------------------------
# Re-export all public names
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "FailureCategory",
    "ReflectionAction",
    "ReflectionOutcome",
    "RepairStrategy",
    # Pydantic-based (autonomous_loop API)
    "ReflectionResult",
    "ReflectionService",
    # Dataclass-based (engine API)
    "ReflectionContext",
    "ReflectionResultDetailed",
    # Engines
    "FailureAnalyzer",
    "ReflectionEngine",
    "SelfRepairLoop",
    # Strategy (Phase 4)
    "EngineeringStrategy",
    "StrategyMemory",
    "StrategyOutcome",
    "ProblemClass",
]


# Lazy re-exports from strategy module for convenience
from kodiak.orchestration.strategy import (  # noqa: E402
    EngineeringStrategy,
    ProblemClass,
    StrategyMemory,
    StrategyOutcome,
)
