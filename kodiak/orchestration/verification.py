"""Verification subsystem for autonomous task execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.orchestration.state import TaskState
from kodiak.orchestration.task_planner import ExecutionPlan

logger = structlog.get_logger(__name__)


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class VerificationResult(BaseModel):
    status: VerificationStatus
    message: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class TaskVerifier:
    """Evaluate whether an execution outcome satisfies the planned task goal."""

    async def verify(
        self,
        *,
        goal: str,
        plan: ExecutionPlan | None,
        execution_result: ExecutionResult,
        task_state: TaskState,
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

        if plan and plan.acceptance_criteria:
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
