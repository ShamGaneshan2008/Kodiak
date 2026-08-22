"""Verification engine orchestrating task outcome validation."""

from __future__ import annotations

import time

import structlog

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import ExecutionContext, ExecutionResult
from kodiak.orchestration.state import TaskState
from kodiak.orchestration.task_planner import ExecutionPlan
from kodiak.orchestration.verification.base import Verifier
from kodiak.orchestration.verification.models import (
    VerificationContext,
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
    aggregate_evidence,
)
from kodiak.orchestration.verification.verifiers import (
    CommandVerifier,
    FileVerifier,
    OutputVerifier,
    TestVerifier,
)
from kodiak.tools.router import ToolRouter

logger = structlog.get_logger(__name__)


def default_verifiers(tool_router: ToolRouter | None = None) -> list[Verifier]:
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
        tool_router: ToolRouter | None = None,
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
        plan: ExecutionPlan | None = None,
        task_state: TaskState | None = None,
    ) -> VerificationResult:
        """Run applicable verifiers and aggregate evidence.

        The autonomous loop verifies a workflow-level result rather than a
        single persisted ``Task``. Retain that call shape alongside the
        task-level verification API.
        """
        if task is None:
            if execution_result is None or goal is None or task_state is None:
                raise TypeError(
                    "workflow verification requires goal, execution_result, and task_state"
                )
            return self._verify_workflow_result(
                goal=goal,
                plan=plan,
                execution_result=execution_result,
                task_state=task_state,
            )

        if execution_result is None:
            raise TypeError("task verification requires an execution_result")

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
                summary="Agent execution did not succeed; verification skipped.",
                duration_seconds=time.monotonic() - started,
            )
            log.info("verification_skipped_execution_failed", status=result.status.value)
            return result

        active = [verifier for verifier in self._verifiers if verifier.applies(context)]
        if not active:
            result = VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                summary="No applicable verifiers for configured criteria.",
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

        result = aggregate_evidence(evidence)
        result = VerificationResult(
            status=result.status,
            evidence=result.evidence,
            duration_seconds=time.monotonic() - started,
            summary=result.summary,
            retry_recommended=result.retry_recommended,
        )
        log.info(
            "verification_finished",
            status=result.status.value,
            verifier_count=len(evidence),
            duration_seconds=result.duration_seconds,
        )
        return result

    @staticmethod
    def _verify_workflow_result(
        *,
        goal: str,
        plan: ExecutionPlan | None,
        execution_result: ExecutionResult,
        task_state: TaskState,
    ) -> VerificationResult:
        """Verify the aggregate result produced by ``AutonomousTaskLoop``."""
        output = execution_result.result or {}
        status = VerificationStatus.VERIFIED
        summary = "Task execution verified."
        retry_recommended = False

        if not execution_result.is_success:
            status = VerificationStatus.FAILED
            summary = "Execution did not complete successfully."
            retry_recommended = True
        else:
            explicit_status = str(output.get("verification_status", "")).lower()
            if explicit_status == VerificationStatus.FAILED.value:
                status = VerificationStatus.FAILED
                summary = str(output.get("verification_message", "Explicit verification failure."))
                retry_recommended = True
            elif explicit_status == VerificationStatus.INCONCLUSIVE.value:
                status = VerificationStatus.INCONCLUSIVE
                summary = str(output.get("verification_message", "Verification inconclusive."))
                retry_recommended = True
            elif not output and not getattr(task_state, "result", None):
                status = VerificationStatus.INCONCLUSIVE
                summary = "Execution completed without output evidence."
                retry_recommended = True
            elif plan and getattr(plan, "acceptance_criteria", None):
                missing = [
                    criterion
                    for criterion in plan.acceptance_criteria
                    if criterion.lower() not in str(output).lower()
                    and criterion.lower() not in goal.lower()
                ]
                if missing and output.get("strict_acceptance", False):
                    status = VerificationStatus.FAILED
                    summary = "Acceptance criteria not satisfied."
                    retry_recommended = True

        evidence = VerificationEvidence(
            verifier="autonomous_output",
            status=status,
            message=summary,
            metadata={"goal": goal, "output": output},
        )
        return VerificationResult(
            status=status,
            evidence=(evidence,),
            summary=summary,
            retry_recommended=retry_recommended,
        )


__all__ = ["VerificationEngine", "default_verifiers"]
