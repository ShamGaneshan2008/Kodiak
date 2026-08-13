"""Verification engine orchestrating task outcome validation."""

from __future__ import annotations

import time
from typing import Any

import structlog

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import ExecutionContext, ExecutionResult
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
        task: Task,
        execution_result: ExecutionResult,
        *,
        execution_context: ExecutionContext | None = None,
    ) -> VerificationResult:
        """Run applicable verifiers and aggregate evidence."""
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


__all__ = ["VerificationEngine", "default_verifiers"]
