"""Deterministic, explainable agent selection for Kodiak."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from kodiak.db.models.task import TaskPriority

logger = structlog.get_logger(__name__)

__all__ = [
    "AgentCandidate",
    "AgentHealthStatus",
    "AgentScore",
    "AgentSelectionStrategy",
    "AgentSelector",
    "NoSuitableAgentError",
    "SelectionContext",
    "SelectionResult",
]


class AgentHealthStatus(StrEnum):
    """Health state used by the selector."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AgentSelectionStrategy(StrEnum):
    """Selection strategy used to produce a decision."""

    CAPABILITY_BASED = "capability_based"
    WEIGHTED_RANKING = "weighted_ranking"


class NoSuitableAgentError(Exception):
    """Raised when no registered agent can satisfy a task."""


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Task-side selection requirements."""

    required_capabilities: frozenset[str] = frozenset()
    priority: TaskPriority = TaskPriority.MEDIUM
    task_type: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentCandidate:
    """Agent-side selection facts supplied by AgentManager."""

    agent_id: str
    capabilities: frozenset[str] = frozenset()
    priority: int = 0
    enabled: bool = True
    health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    in_flight: int = 0
    success_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentScore:
    """Explainable score for one candidate."""

    agent_id: str
    total_score: float
    matched_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    extra_capabilities: tuple[str, ...]
    capability_score: float
    priority_score: float
    health_score: float
    load_score: float
    confidence_score: float
    is_available: bool
    is_compatible: bool
    health_status: AgentHealthStatus
    reason: str
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """The final agent selection and candidate ranking."""

    selected_agent_id: str
    selected_score: float
    matched_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    candidate_ranking: tuple[AgentScore, ...]
    selection_strategy: AgentSelectionStrategy
    reason: str


class AgentSelector:
    """Select the best available compatible agent.

    The selector is intentionally pure and deterministic. It does not run
    health checks, instantiate agents, execute tasks, or mutate workload
    counters; the AgentManager supplies those facts and owns execution.
    """

    def __init__(self) -> None:
        """Initialize an agent selector."""
        self._log = logger.bind(component="agent_selector")

    def select(
        self,
        context: SelectionContext,
        candidates: list[AgentCandidate],
    ) -> SelectionResult:
        """Select the highest-ranked compatible candidate.

        Args:
            context: Task requirements and metadata.
            candidates: Registered agent facts.

        Returns:
            Explainable selection result.

        Raises:
            NoSuitableAgentError: If no candidate is both available and
                capability-compatible.
        """
        if not candidates:
            raise NoSuitableAgentError(
                f"No agents are registered for task {context.task_id or 'unknown'}."
            )

        scored = [self.score(context, candidate) for candidate in candidates]
        ranked = self._rank(scored)
        valid = [score for score in ranked if score.is_available and score.is_compatible]

        if not valid:
            missing = sorted(
                set(context.required_capabilities)
                - {cap for score in ranked for cap in score.matched_capabilities}
            )
            raise NoSuitableAgentError(
                "No suitable agent found"
                f" for task {context.task_id or 'unknown'}"
                f"; missing capabilities: {missing}"
            )

        selected = valid[0]
        strategy = (
            AgentSelectionStrategy.CAPABILITY_BASED
            if context.required_capabilities
            else AgentSelectionStrategy.WEIGHTED_RANKING
        )
        self._log.info(
            "agent_selected",
            task_id=context.task_id,
            agent_id=selected.agent_id,
            score=selected.total_score,
            strategy=strategy.value,
            matched_capabilities=selected.matched_capabilities,
        )
        return SelectionResult(
            selected_agent_id=selected.agent_id,
            selected_score=selected.total_score,
            matched_capabilities=selected.matched_capabilities,
            missing_capabilities=selected.missing_capabilities,
            candidate_ranking=tuple(ranked),
            selection_strategy=strategy,
            reason=selected.reason,
        )

    def score(self, context: SelectionContext, candidate: AgentCandidate) -> AgentScore:
        """Score a candidate against a task context."""
        required = context.required_capabilities
        matched = tuple(sorted(required & candidate.capabilities))
        missing = tuple(sorted(required - candidate.capabilities))
        extra = tuple(sorted(candidate.capabilities - required))
        is_compatible = not missing

        capability_score = len(matched) / len(required) if required else 1.0
        priority_score = self._priority_score(candidate.priority)
        health_score = self._health_score(candidate.health_status)
        load_score = 1.0 / (candidate.in_flight + 1)
        confidence_score = candidate.success_rate if candidate.success_rate is not None else 0.5
        confidence_score = max(0.0, min(confidence_score, 1.0))
        is_available = (
            candidate.enabled and candidate.health_status is not AgentHealthStatus.UNHEALTHY
        )

        if not is_available or not is_compatible:
            total_score = 0.0
        else:
            total_score = (
                capability_score,
                priority_score,
                health_score,
                load_score,
                confidence_score,
            )
            total_score = sum(total_score)

        reason = self._reason(candidate, matched, missing, is_available)
        return AgentScore(
            agent_id=candidate.agent_id,
            total_score=round(total_score, 6),
            matched_capabilities=matched,
            missing_capabilities=missing,
            extra_capabilities=extra,
            capability_score=round(capability_score, 6),
            priority_score=round(priority_score, 6),
            health_score=round(health_score, 6),
            load_score=round(load_score, 6),
            confidence_score=round(confidence_score, 6),
            is_available=is_available,
            is_compatible=is_compatible,
            health_status=candidate.health_status,
            reason=reason,
        )

    @staticmethod
    def _rank(scores: list[AgentScore]) -> list[AgentScore]:
        ordered = sorted(
            scores,
            key=lambda score: (
                not score.is_available,
                not score.is_compatible,
                -score.total_score,
                -score.priority_score,
                score.agent_id,
            ),
        )
        return [
            AgentScore(
                agent_id=score.agent_id,
                total_score=score.total_score,
                matched_capabilities=score.matched_capabilities,
                missing_capabilities=score.missing_capabilities,
                extra_capabilities=score.extra_capabilities,
                capability_score=score.capability_score,
                priority_score=score.priority_score,
                health_score=score.health_score,
                load_score=score.load_score,
                confidence_score=score.confidence_score,
                is_available=score.is_available,
                is_compatible=score.is_compatible,
                health_status=score.health_status,
                reason=score.reason,
                rank=index,
            )
            for index, score in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _priority_score(priority: int) -> float:
        return max(0.0, min(float(priority), 100.0)) / 100.0

    @staticmethod
    def _health_score(status: AgentHealthStatus) -> float:
        if status is AgentHealthStatus.HEALTHY:
            return 1.0
        if status is AgentHealthStatus.UNKNOWN:
            return 0.5
        return 0.0

    @staticmethod
    def _reason(
        candidate: AgentCandidate,
        matched: tuple[str, ...],
        missing: tuple[str, ...],
        is_available: bool,
    ) -> str:
        if missing:
            return f"Rejected because capabilities are missing: {', '.join(missing)}."
        if not is_available:
            return "Rejected because the agent is disabled or unhealthy."
        if matched:
            return f"Selected with exact required capability match: {', '.join(matched)}."
        return "Selected from available agents using priority, health, load, and confidence."
