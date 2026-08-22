"""Reflection engine for failure analysis and repair decisions."""

from __future__ import annotations

import time

import structlog

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import ExecutionContext, ExecutionResult
from kodiak.orchestration.reflection.analyzer import FailureAnalyzer
from kodiak.orchestration.reflection.models import (
    ReflectionContext,
    ReflectionResult,
)
from kodiak.orchestration.verification.models import VerificationResult

logger = structlog.get_logger(__name__)


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
    ) -> ReflectionResult:
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


__all__ = ["ReflectionEngine"]
