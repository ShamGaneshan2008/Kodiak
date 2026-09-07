import asyncio
import enum
import platform
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .metrics import MetricsCollector


class EvaluationTaskType(enum.StrEnum):
    REPOSITORY_UNDERSTANDING = "repository_understanding"
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    TEST_GENERATION = "test_generation"
    REFACTOR = "refactor"
    SECURITY = "security"
    RECOVERY = "recovery"
    GIT_WORKFLOW = "git_workflow"


# Backward-compatible public name used by existing callers.
TaskType = EvaluationTaskType


class BenchmarkDifficulty(enum.StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReleaseReadiness(enum.StrEnum):
    NOT_READY = "not_ready"
    ALPHA_READY = "alpha_ready"
    BETA_READY = "beta_ready"
    RELEASE_CANDIDATE = "release_candidate"


@dataclass
class EvaluationCase:
    id: str
    name: str
    description: str
    task_type: EvaluationTaskType
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    difficulty: BenchmarkDifficulty = BenchmarkDifficulty.EASY
    repository: str = "fixture"
    expected_files: tuple[str, ...] = ()
    requires_approval: bool = False


@dataclass
class EvaluationResult:
    case_id: str
    case_name: str
    passed: bool
    output: Any
    execution_time: float
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    category: str = "general"
    difficulty: BenchmarkDifficulty = BenchmarkDifficulty.EASY
    repository: str = "fixture"
    verification_status: str = "unknown"
    attempts: int = 1
    repair_attempts: int = 0
    files_changed: tuple[str, ...] = ()
    tests_executed: tuple[str, ...] = ()
    tool_calls: int = 0
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    expected_approval: bool = False
    unexpected_human_intervention: bool = False
    regression: bool = False
    false_success: bool = False
    failure_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "category": self.category,
            "difficulty": self.difficulty.value,
            "repository": self.repository,
            "passed": self.passed,
            "verification_status": self.verification_status,
            "execution_time": self.execution_time,
            "error": self.error,
            "attempts": self.attempts,
            "repair_attempts": self.repair_attempts,
            "files_changed": list(self.files_changed),
            "tests_executed": list(self.tests_executed),
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "expected_approval": self.expected_approval,
            "unexpected_human_intervention": self.unexpected_human_intervention,
            "regression": self.regression,
            "false_success": self.false_success,
            "failure_category": self.failure_category,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


class EvaluationHarness:
    def __init__(self, llm_provider: Any | None = None):
        self.llm_provider = llm_provider
        self.metrics = MetricsCollector()
        self.results: list[EvaluationResult] = []
        self.validators: dict[str, Callable] = {}

    def register_validator(self, task_type: str, validator: Callable):
        self.validators[task_type] = validator

    async def run_case(
        self,
        case: EvaluationCase,
        agent_execute_fn: Callable,
    ) -> EvaluationResult:
        start_time = time.time()
        error = None
        output = None
        passed = False

        try:
            output = await asyncio.wait_for(
                agent_execute_fn(case.input_data),
                timeout=case.timeout_seconds,
            )

            if case.task_type.value in self.validators:
                validator = self.validators[case.task_type.value]
                passed = validator(output, case.expected_output)
            else:
                passed = False

        except TimeoutError:
            error = f"Execution timeout after {case.timeout_seconds}s"
        except Exception as e:
            error = str(e)

        execution_time = time.time() - start_time

        metadata = output if isinstance(output, dict) else {}
        verification_status = str(metadata.get("verification_status", "unknown"))
        false_success = bool(metadata.get("claimed_success")) and not passed
        result = EvaluationResult(
            case_id=case.id,
            case_name=case.name,
            passed=passed,
            output=output,
            execution_time=execution_time,
            error=error,
            metrics=self.metrics.get_snapshot(),
            category=case.category,
            difficulty=case.difficulty,
            repository=case.repository,
            verification_status=verification_status,
            attempts=max(1, int(metadata.get("attempts", 1))),
            repair_attempts=max(0, int(metadata.get("repair_attempts", 0))),
            files_changed=tuple(metadata.get("files_changed", ())),
            tests_executed=tuple(metadata.get("tests_executed", ())),
            tool_calls=max(0, int(metadata.get("tool_calls", 0))),
            model_calls=max(0, int(metadata.get("model_calls", 0))),
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            estimated_cost_usd=metadata.get("estimated_cost_usd"),
            expected_approval=case.requires_approval,
            unexpected_human_intervention=bool(metadata.get("human_intervention", False)),
            regression=bool(metadata.get("regression", False)),
            false_success=false_success,
            failure_category=(
                str(metadata["failure_category"]) if metadata.get("failure_category") else None
            ),
        )

        self.results.append(result)
        return result

    async def run_suite(
        self,
        cases: list[EvaluationCase],
        agent_execute_fn: Callable,
    ) -> dict[str, Any]:
        results = []
        for case in cases:
            result = await self.run_case(case, agent_execute_fn)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        avg_execution_time = sum(r.execution_time for r in results) / len(results) if results else 0

        return {
            "total_cases": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": pass_rate,
            "avg_execution_time": avg_execution_time,
            "results": results,
        }

    def get_results(self) -> list[EvaluationResult]:
        return self.results

    def clear_results(self):
        self.results = []
        self.metrics.reset()

    def scorecard(self, results: list[EvaluationResult] | None = None) -> dict[str, Any]:
        """Aggregate transparent, verification-based benchmark measurements."""
        measured = list(results if results is not None else self.results)
        total = len(measured)
        passed = sum(result.passed for result in measured)
        repair_runs = [result for result in measured if result.repair_attempts > 0]
        latencies = sorted(result.execution_time for result in measured)
        failure_counts: dict[str, int] = {}
        for result in measured:
            if not result.passed:
                category = result.failure_category or "unclassified"
                failure_counts[category] = failure_counts.get(category, 0) + 1
        blockers = []
        if any(result.false_success for result in measured):
            blockers.append("false_verification_success")
        if any(
            result.failure_category in {"secret_leak", "destructive_action"} for result in measured
        ):
            blockers.append("critical_security_failure")
        readiness = ReleaseReadiness.NOT_READY
        if total and not blockers and passed == total:
            categories = {result.category for result in measured}
            readiness = (
                ReleaseReadiness.BETA_READY
                if {"security", "crash_recovery", "git", "architecture"}.issubset(categories)
                else ReleaseReadiness.ALPHA_READY
            )
        return {
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "total_tasks": total,
            "verified_success": passed,
            "verified_success_rate": passed / total if total else 0.0,
            "first_try_success_rate": (
                sum(result.passed and result.repair_attempts == 0 for result in measured) / total
                if total
                else 0.0
            ),
            "self_repair_success_rate": (
                sum(result.passed for result in repair_runs) / len(repair_runs)
                if repair_runs
                else None
            ),
            "regression_rate": sum(result.regression for result in measured) / total
            if total
            else 0.0,
            "false_success_count": sum(result.false_success for result in measured),
            "security_failures": sum(
                result.category == "security" and not result.passed for result in measured
            ),
            "crash_recovery_failures": sum(
                result.category == "crash_recovery" and not result.passed for result in measured
            ),
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "p95_latency_seconds": self._percentile(latencies, 0.95),
            "tool_calls": sum(result.tool_calls for result in measured),
            "model_calls": sum(result.model_calls for result in measured),
            "input_tokens": self._optional_sum(result.input_tokens for result in measured),
            "output_tokens": self._optional_sum(result.output_tokens for result in measured),
            "estimated_cost_usd": self._optional_sum(
                result.estimated_cost_usd for result in measured
            ),
            "expected_approvals": sum(result.expected_approval for result in measured),
            "unexpected_human_interventions": sum(
                result.unexpected_human_intervention for result in measured
            ),
            "top_failure_categories": sorted(
                failure_counts.items(), key=lambda item: (-item[1], item[0])
            ),
            "release_blockers": blockers,
            "release_readiness": readiness.value,
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        return values[min(int((len(values) - 1) * percentile), len(values) - 1)]

    @staticmethod
    def _optional_sum(values: Any) -> float | int | None:
        available = [value for value in values if value is not None]
        return sum(available) if available else None
