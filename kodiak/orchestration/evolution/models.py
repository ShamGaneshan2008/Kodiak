"""Core evolution models — structured representations of system evaluation.

Every evaluation has multiple dimensions.  The system is never reduced
to a single score.  Each dimension carries evidence and provenance.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class EvaluationDimension(enum.StrEnum):
    """Dimensions along which Kodiak evaluates itself."""

    PLANNING_QUALITY = "planning_quality"
    STRATEGY_SELECTION = "strategy_selection"
    AGENT_SELECTION = "agent_selection"
    TOOL_SELECTION = "tool_selection"
    MEMORY_RETRIEVAL = "memory_retrieval"
    VERIFICATION_QUALITY = "verification_quality"
    RECOVERY_BEHAVIOR = "recovery_behavior"
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    RESEARCH_EFFECTIVENESS = "research_effectiveness"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    EXECUTION_RELIABILITY = "execution_reliability"
    REGRESSION_RATE = "regression_rate"
    HUMAN_INTERVENTION = "human_intervention"
    OVERALL_TASK_SUCCESS = "overall_task_success"


class EvaluationVerdict(enum.StrEnum):
    """High-level verdict for an evaluation dimension."""

    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    FAILING = "failing"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """Score for a single evaluation dimension.

    Carries the numeric score, a verdict, and supporting evidence.
    """

    dimension: EvaluationDimension
    score: float  # 0.0-1.0
    verdict: EvaluationVerdict
    evidence: str = ""
    confidence: float = 0.5
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "verdict": self.verdict.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "measurements": dict(self.measurements),
        }


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    """Structured evaluation of a single task execution.

    Captures what worked, what failed, where effort was wasted,
    and which assumptions were wrong.
    """

    evaluation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    goal: str = ""

    # Multi-dimensional scores
    dimension_scores: tuple[DimensionScore, ...] = ()

    # Qualitative assessment
    what_worked: tuple[str, ...] = ()
    what_failed: tuple[str, ...] = ()
    wasted_effort: tuple[str, ...] = ()
    wrong_assumptions: tuple[str, ...] = ()
    failure_component: str = ""
    better_strategy: str = ""

    # Memory usefulness
    memory_helped: bool = False
    planning_helped: bool = False
    verification_caught_failure: bool = False

    # Agent and tool assessment
    agent_appropriate: bool = True
    tool_appropriate: bool = True

    # Overall
    overall_score: float = 0.0
    overall_verdict: EvaluationVerdict = EvaluationVerdict.UNKNOWN
    summary: str = ""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def avg_score(self) -> float:
        if not self.dimension_scores:
            return self.overall_score
        return sum(d.score for d in self.dimension_scores) / len(self.dimension_scores)

    @property
    def weak_dimensions(self) -> tuple[EvaluationDimension, ...]:
        return tuple(
            d.dimension
            for d in self.dimension_scores
            if d.verdict in {EvaluationVerdict.WEAK, EvaluationVerdict.FAILING}
        )

    @property
    def strong_dimensions(self) -> tuple[EvaluationDimension, ...]:
        return tuple(
            d.dimension
            for d in self.dimension_scores
            if d.verdict == EvaluationVerdict.STRONG
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "task_id": self.task_id,
            "goal": self.goal,
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "what_worked": list(self.what_worked),
            "what_failed": list(self.what_failed),
            "wasted_effort": list(self.wasted_effort),
            "wrong_assumptions": list(self.wrong_assumptions),
            "failure_component": self.failure_component,
            "better_strategy": self.better_strategy,
            "memory_helped": self.memory_helped,
            "planning_helped": self.planning_helped,
            "verification_caught_failure": self.verification_caught_failure,
            "agent_appropriate": self.agent_appropriate,
            "tool_appropriate": self.tool_appropriate,
            "overall_score": self.overall_score,
            "overall_verdict": self.overall_verdict.value,
            "summary": self.summary,
            "avg_score": self.avg_score,
            "weak_dimensions": [d.value for d in self.weak_dimensions],
            "strong_dimensions": [d.value for d in self.strong_dimensions],
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SystemEvaluation:
    """Aggregate evaluation across multiple task evaluations.

    Provides a system-level view of Kodiak's current performance
    across all dimensions.
    """

    evaluation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_evaluations_count: int = 0
    dimension_averages: tuple[DimensionScore, ...] = ()
    overall_score: float = 0.0
    overall_verdict: EvaluationVerdict = EvaluationVerdict.UNKNOWN
    weakest_dimensions: tuple[EvaluationDimension, ...] = ()
    strongest_dimensions: tuple[EvaluationDimension, ...] = ()
    recurring_failures: tuple[str, ...] = ()
    improvement_opportunities: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "task_evaluations_count": self.task_evaluations_count,
            "dimension_averages": [d.to_dict() for d in self.dimension_averages],
            "overall_score": self.overall_score,
            "overall_verdict": self.overall_verdict.value,
            "weakest_dimensions": [d.value for d in self.weakest_dimensions],
            "strongest_dimensions": [d.value for d in self.strongest_dimensions],
            "recurring_failures": list(self.recurring_failures),
            "improvement_opportunities": list(self.improvement_opportunities),
            "created_at": self.created_at.isoformat(),
        }


__all__ = [
    "DimensionScore",
    "EvaluationDimension",
    "EvaluationVerdict",
    "SystemEvaluation",
    "TaskEvaluation",
]
