"""Structured reflection decisions for the autonomous task loop."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.orchestration.state import TaskState
from kodiak.orchestration.verification import VerificationResult, VerificationStatus

logger = structlog.get_logger(__name__)


class ReflectionAction(StrEnum):
    RETRY = "retry"
    REPAIR = "repair"
    REPLAN = "replan"
    STOP = "stop"


class ReflectionResult(BaseModel):
    action: ReflectionAction
    root_cause: str
    retryable: bool = True
    repair_required: bool = False
    replan_required: bool = False
    strategy: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReflectionService:
    """Determine follow-up action after verification or execution failure."""

    def __init__(self, max_retries: int = 3, max_replans: int = 2) -> None:
        self._max_retries = max_retries
        self._max_replans = max_replans

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
