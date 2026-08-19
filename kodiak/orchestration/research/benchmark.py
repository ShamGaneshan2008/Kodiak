"""Benchmark infrastructure for evaluating strategies.

Provides reusable benchmark tasks and suites for systematic strategy
evaluation.  Each benchmark task has an objective, initial state,
constraints, acceptance criteria, and verification procedure.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from kodiak.orchestration.research.models import ExperimentResult

logger = structlog.get_logger(__name__)


class BenchmarkTaskCategory(enum.StrEnum):
    """Categories of benchmark tasks."""

    BUG_FIXING = "bug_fixing"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    REFACTORING = "refactoring"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    TEST_REPAIR = "test_repair"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_HARDENING = "security_hardening"
    MULTI_FILE_ARCHITECTURAL = "multi_file_architectural"
    DEBUGGING = "debugging"
    REPOSITORY_MIGRATION = "repository_migration"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """A reusable benchmark task for strategy evaluation.

    Attributes:
        task_id: Unique identifier.
        title: Short human-readable title.
        category: Task category.
        objective: What the task should accomplish.
        initial_state: Description of the starting state.
        constraints: Constraints on the solution.
        acceptance_criteria: How to verify the task was completed.
        verification_procedure: Steps to verify the solution.
        difficulty: Difficulty estimate (1-5).
        tags: Searchable tags.
        metadata: Arbitrary additional data.
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    category: BenchmarkTaskCategory = BenchmarkTaskCategory.UNKNOWN
    objective: str = ""
    initial_state: str = ""
    constraints: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    verification_procedure: tuple[str, ...] = ()
    difficulty: int = 3
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "category": self.category.value,
            "objective": self.objective,
            "initial_state": self.initial_state,
            "constraints": list(self.constraints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "verification_procedure": list(self.verification_procedure),
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Result of running a strategy on a benchmark task."""

    task_id: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    success: bool = False
    primary_metric: float = 0.0
    duration_seconds: float = 0.0
    tool_calls: int = 0
    verification_passed: bool = False
    error_message: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "success": self.success,
            "primary_metric": self.primary_metric,
            "duration_seconds": self.duration_seconds,
            "tool_calls": self.tool_calls,
            "verification_passed": self.verification_passed,
            "error_message": self.error_message,
            "measurements": dict(self.measurements),
        }


@dataclass
class BenchmarkSuite:
    """A collection of benchmark tasks for strategy evaluation.

    Suites group related benchmark tasks for systematic comparison.
    """

    suite_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    description: str = ""
    tasks: tuple[BenchmarkTask, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({t.category.value for t in self.tasks}))

    def tasks_by_category(
        self, category: BenchmarkTaskCategory
    ) -> tuple[BenchmarkTask, ...]:
        return tuple(t for t in self.tasks if t.category == category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
            "task_count": self.task_count,
            "categories": list(self.categories),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


class BenchmarkRunner:
    """Runs a strategy against a benchmark suite and collects results.

    The runner is a framework — callers must provide the actual
    strategy execution function.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="benchmark_runner")

    def aggregate_results(
        self, results: list[BenchmarkResult]
    ) -> ExperimentResult:
        """Aggregate benchmark results into an ExperimentResult."""
        if not results:
            return ExperimentResult(
                primary_metric=0.0,
                total_tasks=0,
            )

        successful = sum(1 for r in results if r.success)
        total = len(results)
        total_duration = sum(r.duration_seconds for r in results)
        total_tool_calls = sum(r.tool_calls for r in results)

        primary_metric = successful / total if total > 0 else 0.0

        failures = tuple(
            r.error_message for r in results if not r.success and r.error_message
        )

        strategy_name = results[0].strategy_name if results else "unknown"
        strategy_id = results[0].strategy_id if results else ""

        return ExperimentResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            primary_metric=primary_metric,
            task_success_rate=primary_metric,
            total_tasks=total,
            successful_tasks=successful,
            failed_tasks=total - successful,
            total_duration_seconds=total_duration,
            average_duration_seconds=total_duration / total if total > 0 else 0.0,
            tool_calls=total_tool_calls,
            failures=failures,
        )

    def compare_suite_results(
        self,
        baseline_results: list[BenchmarkResult],
        candidate_results: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Compare baseline and candidate results from a benchmark suite."""
        baseline_agg = self.aggregate_results(baseline_results)
        candidate_agg = self.aggregate_results(candidate_results)

        improvement = 0.0
        if baseline_agg.primary_metric > 0:
            improvement = (
                (candidate_agg.primary_metric - baseline_agg.primary_metric)
                / baseline_agg.primary_metric
            )

        return {
            "baseline": baseline_agg.to_dict(),
            "candidate": candidate_agg.to_dict(),
            "improvement": improvement,
            "baseline_tasks": baseline_agg.total_tasks,
            "candidate_tasks": candidate_agg.total_tasks,
            "conclusion": (
                f"Candidate improved by {improvement:.1%}"
                if improvement > 0
                else f"Candidate degraded by {abs(improvement):.1%}"
                if improvement < 0
                else "No difference"
            ),
        }


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "BenchmarkTaskCategory",
]
