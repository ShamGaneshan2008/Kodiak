"""Focused tests for reflection and self-repair."""

from __future__ import annotations

from uuid import uuid4

import pytest

from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.orchestration.reflection.analyzer import FailureAnalyzer
from kodiak.orchestration.reflection.engine import ReflectionEngine
from kodiak.orchestration.reflection.models import (
    FailureCategory,
    ReflectionContext,
    ReflectionOutcome,
    RepairStrategy,
)
from kodiak.orchestration.verification.models import VerificationStatus


def _task(**context: object) -> Task:
    return Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="reflect",
        description="reflection test",
        source=TaskSource.API,
        status=TaskStatus.FAILED,
        priority=TaskPriority.MEDIUM,
        max_retries=2,
        context=dict(context),
    )


def _execution_result(
    *,
    outcome: ExecutionOutcome = ExecutionOutcome.FAILURE,
    result: dict | None = None,
    error: dict | None = None,
    verification: dict | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        task_id="task-1",
        outcome=outcome,
        attempts=1,
        duration_seconds=0.1,
        result=result or {},
        error=error,
        verification=verification,
        final_status=TaskStatus.FAILED,
    )


def test_successful_verification_requires_no_repair() -> None:
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(outcome=ExecutionOutcome.SUCCESS),
        attempt=1,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.outcome is ReflectionOutcome.SUCCESS
    assert result.strategy is RepairStrategy.STOP


def test_retryable_test_failure() -> None:
    verification = {
        "status": VerificationStatus.FAILED.value,
        "summary": "Tests failed",
        "evidence": [
            {
                "verifier": "test",
                "status": VerificationStatus.FAILED.value,
                "message": "pytest failed: expected 200, received 500",
                "stdout_summary": "AssertionError",
            }
        ],
    }
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(
            outcome=ExecutionOutcome.FAILURE,
            verification=verification,
            error={"message": "verification failed"},
        ),
        attempt=1,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.category is FailureCategory.TEST_FAILURE
    assert result.outcome is ReflectionOutcome.RETRYABLE_FAILURE
    assert result.strategy is RepairStrategy.RETRY
    assert "500" in result.root_cause or "Tests" in result.root_cause


def test_non_retryable_permission_failure() -> None:
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(
            error={"message": "Permission DENIED for agent 'coder' executing tool 'write_file'."},
        ),
        attempt=1,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.category is FailureCategory.PERMISSION_FAILURE
    assert result.outcome is ReflectionOutcome.NON_RETRYABLE_FAILURE
    assert result.strategy is RepairStrategy.STOP


def test_replan_required_after_repeated_test_failure() -> None:
    verification = {
        "status": VerificationStatus.FAILED.value,
        "evidence": [
            {
                "verifier": "test",
                "status": VerificationStatus.FAILED.value,
                "message": "pytest failed again",
            }
        ],
    }
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(outcome=ExecutionOutcome.FAILURE, verification=verification),
        attempt=2,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.outcome is ReflectionOutcome.REPLAN_REQUIRED
    assert result.strategy is RepairStrategy.REPLAN


def test_timeout_failure_is_retryable() -> None:
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(
            outcome=ExecutionOutcome.TIMEOUT,
            error={"message": "attempt exceeded 30s timeout"},
        ),
        attempt=1,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.category is FailureCategory.TIMEOUT
    assert result.strategy is RepairStrategy.RETRY


def test_missing_artifact_failure() -> None:
    verification = {
        "status": VerificationStatus.FAILED.value,
        "evidence": [
            {
                "verifier": "file",
                "status": VerificationStatus.FAILED.value,
                "message": "Missing expected files: output.txt",
                "artifacts_checked": ["output.txt"],
            }
        ],
    }
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(outcome=ExecutionOutcome.FAILURE, verification=verification),
        attempt=1,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.category is FailureCategory.MISSING_ARTIFACT
    assert "output.txt" in result.suggested_correction or "output.txt" in result.root_cause


def test_max_retries_reached() -> None:
    context = ReflectionContext.from_execution(
        _task(),
        _execution_result(error={"message": "still failing"}),
        attempt=3,
        max_attempts=3,
    )
    result = FailureAnalyzer().analyze(context)
    assert result.outcome is ReflectionOutcome.MAX_RETRIES_REACHED
    assert result.strategy is RepairStrategy.STOP


@pytest.mark.asyncio
async def test_reflection_engine_structured_output() -> None:
    engine = ReflectionEngine()
    task = _task()
    execution = _execution_result(
        verification={
            "status": VerificationStatus.FAILED.value,
            "evidence": [
                {
                    "verifier": "output",
                    "status": VerificationStatus.FAILED.value,
                    "message": "Missing required output fields: analysis",
                }
            ],
        }
    )
    result = await engine.reflect(task, execution, attempt=1, max_attempts=3)
    payload = result.to_dict()
    assert payload["category"] == FailureCategory.INCORRECT_IMPLEMENTATION.value
    assert payload["root_cause"]
    assert payload["suggested_correction"]
    assert "evidence" in payload or result.evidence


def test_reflection_should_reflect_rules() -> None:
    success = _execution_result(outcome=ExecutionOutcome.SUCCESS)
    assert ReflectionEngine.should_reflect(_task(), success) is False

    failed = _execution_result(
        outcome=ExecutionOutcome.FAILURE,
        verification={"status": VerificationStatus.FAILED.value},
    )
    assert ReflectionEngine.should_reflect(_task(), failed) is True
