"""Meta-strategy selection — strategies for choosing strategies.

The meta-level determines which strategy-selection mechanism is
appropriate for a given task.  This creates a controlled meta-level
that prevents the system from always using the same approach.

Example routing:
    simple task → direct retrieval
    uncertain task → experimentation
    high-risk task → multiple verification paths
    novel problem → research mode
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class StrategySelectionMethod(enum.StrEnum):
    """Methods for selecting a strategy."""

    HISTORICAL_RETRIEVAL = "historical_retrieval"
    BENCHMARK_RANKING = "benchmark_ranking"
    MULTI_STRATEGY_EXPERIMENT = "multi_strategy_experiment"
    RESEARCH_DRIVEN = "research_driven"
    COMPOSITION_BASED = "composition_based"
    DIRECT_APPLICATION = "direct_application"


class TaskComplexity(enum.StrEnum):
    """Estimated task complexity."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    NOVEL = "novel"


class RiskLevel(enum.StrEnum):
    """Risk level of a task."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class MetaStrategyProfile:
    """A profile describing how to select strategies for a task class.

    Each profile maps task characteristics to a selection method.
    """

    profile_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    description: str = ""
    task_complexity: TaskComplexity = TaskComplexity.MODERATE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    selection_method: StrategySelectionMethod = StrategySelectionMethod.HISTORICAL_RETRIEVAL
    confidence_threshold: float = 0.5
    max_strategy_candidates: int = 3
    require_verification: bool = True
    allow_experimentation: bool = False
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "task_complexity": self.task_complexity.value,
            "risk_level": self.risk_level.value,
            "selection_method": self.selection_method.value,
            "confidence_threshold": self.confidence_threshold,
            "max_strategy_candidates": self.max_strategy_candidates,
            "require_verification": self.require_verification,
            "allow_experimentation": self.allow_experimentation,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class MetaStrategyDecision:
    """The output of meta-strategy selection."""

    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    profile_used: str = ""
    selection_method: StrategySelectionMethod = StrategySelectionMethod.HISTORICAL_RETRIEVAL
    reasoning: str = ""
    confidence: float = 0.5
    alternatives: tuple[StrategySelectionMethod, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "task_id": self.task_id,
            "profile_used": self.profile_used,
            "selection_method": self.selection_method.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "alternatives": [a.value for a in self.alternatives],
            "metadata": dict(self.metadata),
        }


class MetaStrategySelector:
    """Selects the appropriate strategy-selection method for a task.

    Maintains profiles that map task characteristics to selection
    methods.  Learns from outcomes to improve routing over time.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, MetaStrategyProfile] = {}
        self._decisions: list[MetaStrategyDecision] = []
        self._method_outcomes: dict[str, list[bool]] = {
            method.value: [] for method in StrategySelectionMethod
        }
        self._log = logger.bind(component="meta_strategy_selector")
        self._initialize_default_profiles()

    def _initialize_default_profiles(self) -> None:
        """Create sensible default profiles."""
        defaults = [
            MetaStrategyProfile(
                name="trivial_task",
                description="Trivial tasks need minimal reasoning",
                task_complexity=TaskComplexity.TRIVIAL,
                risk_level=RiskLevel.LOW,
                selection_method=StrategySelectionMethod.DIRECT_APPLICATION,
                confidence_threshold=0.3,
                require_verification=False,
                tags=("trivial", "quick"),
            ),
            MetaStrategyProfile(
                name="simple_known",
                description="Simple tasks with known solutions",
                task_complexity=TaskComplexity.SIMPLE,
                risk_level=RiskLevel.LOW,
                selection_method=StrategySelectionMethod.HISTORICAL_RETRIEVAL,
                confidence_threshold=0.5,
                tags=("simple", "known"),
            ),
            MetaStrategyProfile(
                name="moderate_task",
                description="Moderate complexity tasks",
                task_complexity=TaskComplexity.MODERATE,
                risk_level=RiskLevel.MEDIUM,
                selection_method=StrategySelectionMethod.BENCHMARK_RANKING,
                confidence_threshold=0.6,
                allow_experimentation=True,
                tags=("moderate",),
            ),
            MetaStrategyProfile(
                name="complex_task",
                description="Complex tasks requiring deeper analysis",
                task_complexity=TaskComplexity.COMPLEX,
                risk_level=RiskLevel.HIGH,
                selection_method=StrategySelectionMethod.MULTI_STRATEGY_EXPERIMENT,
                confidence_threshold=0.7,
                require_verification=True,
                allow_experimentation=True,
                max_strategy_candidates=5,
                tags=("complex", "high_risk"),
            ),
            MetaStrategyProfile(
                name="novel_problem",
                description="Novel problems requiring research",
                task_complexity=TaskComplexity.NOVEL,
                risk_level=RiskLevel.HIGH,
                selection_method=StrategySelectionMethod.RESEARCH_DRIVEN,
                confidence_threshold=0.8,
                require_verification=True,
                allow_experimentation=True,
                max_strategy_candidates=5,
                tags=("novel", "research"),
            ),
        ]
        for profile in defaults:
            self._profiles[profile.profile_id] = profile

    def register_profile(self, profile: MetaStrategyProfile) -> None:
        self._profiles[profile.profile_id] = profile
        self._log.info(
            "profile_registered",
            profile_id=profile.profile_id,
            name=profile.name,
        )

    def select_method(
        self,
        *,
        task_id: str,
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        has_known_strategies: bool = True,
        strategy_confidence: float = 0.5,
        tags: tuple[str, ...] = (),
    ) -> MetaStrategyDecision:
        """Select the appropriate strategy-selection method for a task.

        Examines task characteristics, matches against profiles, and
        returns a decision with reasoning.
        """
        # Find matching profiles
        candidates = self._match_profiles(
            complexity=complexity,
            risk_level=risk_level,
            tags=tags,
        )

        if not candidates:
            # Fallback to default
            method = StrategySelectionMethod.HISTORICAL_RETRIEVAL
            profile_name = "default_fallback"
            reasoning = "No matching profile found; using historical retrieval."
            confidence = 0.4
        else:
            # Pick best profile
            best = candidates[0]
            method = best.selection_method
            profile_name = best.name
            confidence = self._compute_confidence(best, strategy_confidence, has_known_strategies)
            reasoning = self._build_reasoning(best, complexity, risk_level, has_known_strategies, strategy_confidence)

            # Override method based on runtime signals
            method = self._maybe_override_method(
                method=method,
                has_known_strategies=has_known_strategies,
                strategy_confidence=strategy_confidence,
                risk_level=risk_level,
                best=best,
            )

        alternatives = tuple(
            a.selection_method for a in candidates[1:3] if a.selection_method != method
        )

        decision = MetaStrategyDecision(
            task_id=task_id,
            profile_used=profile_name,
            selection_method=method,
            reasoning=reasoning,
            confidence=confidence,
            alternatives=alternatives,
            metadata={
                "complexity": complexity.value,
                "risk_level": risk_level.value,
                "has_known_strategies": has_known_strategies,
                "strategy_confidence": strategy_confidence,
            },
        )

        self._decisions.append(decision)
        self._log.info(
            "meta_strategy_selected",
            task_id=task_id,
            method=method.value,
            confidence=round(confidence, 3),
            profile=profile_name,
        )

        return decision

    def record_outcome(self, method: StrategySelectionMethod, success: bool) -> None:
        """Record the outcome of a strategy selection method."""
        self._method_outcomes[method.value].append(success)
        # Keep only last 100 outcomes per method
        if len(self._method_outcomes[method.value]) > 100:
            self._method_outcomes[method.value] = self._method_outcomes[method.value][-100:]

    def method_success_rate(self, method: StrategySelectionMethod) -> float:
        """Get the historical success rate for a selection method."""
        outcomes = self._method_outcomes.get(method.value, [])
        if not outcomes:
            return 0.5  # Unknown
        return sum(outcomes) / len(outcomes)

    def all_profiles(self) -> list[MetaStrategyProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.name)

    def recent_decisions(self, limit: int = 10) -> list[MetaStrategyDecision]:
        return self._decisions[-limit:]

    def method_report(self) -> dict[str, dict[str, Any]]:
        """Report on each selection method's performance."""
        report: dict[str, dict[str, Any]] = {}
        for method in StrategySelectionMethod:
            outcomes = self._method_outcomes.get(method.value, [])
            total = len(outcomes)
            successes = sum(outcomes)
            report[method.value] = {
                "total_uses": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else None,
            }
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_profiles(
        self,
        *,
        complexity: TaskComplexity,
        risk_level: RiskLevel,
        tags: tuple[str, ...],
    ) -> list[MetaStrategyProfile]:
        """Find profiles matching the given task characteristics."""
        scored: list[tuple[float, MetaStrategyProfile]] = []
        tag_set = set(tags)

        complexity_order = list(TaskComplexity)
        risk_order = list(RiskLevel)

        for profile in self._profiles.values():
            score = 0.0

            # Complexity match (closer = better)
            complexity_diff = abs(
                complexity_order.index(profile.task_complexity)
                - complexity_order.index(complexity)
            )
            score += max(0.0, 1.0 - complexity_diff * 0.3)

            # Risk match
            risk_diff = abs(
                risk_order.index(profile.risk_level)
                - risk_order.index(risk_level)
            )
            score += max(0.0, 1.0 - risk_diff * 0.3)

            # Tag overlap
            if tag_set and profile.tags:
                overlap = len(tag_set & set(profile.tags))
                score += overlap * 0.1

            scored.append((score, profile))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [profile for _, profile in scored if scored[0][0] > 0 or profile == scored[0][1]]

    def _compute_confidence(
        self,
        profile: MetaStrategyProfile,
        strategy_confidence: float,
        has_known_strategies: bool,
    ) -> float:
        """Compute confidence in the meta-strategy decision."""
        base = profile.confidence_threshold
        if has_known_strategies:
            base = min(base + 0.1, 1.0)
        if strategy_confidence > 0.7:
            base = min(base + 0.05, 1.0)
        return base

    def _build_reasoning(
        self,
        profile: MetaStrategyProfile,
        complexity: TaskComplexity,
        risk_level: RiskLevel,
        has_known_strategies: bool,
        strategy_confidence: float,
    ) -> str:
        parts = [
            f"Profile '{profile.name}' matched for {complexity.value} complexity, {risk_level.value} risk.",
            f"Selected method: {profile.selection_method.value}.",
        ]
        if has_known_strategies:
            parts.append(f"Known strategies available (confidence: {strategy_confidence:.2f}).")
        else:
            parts.append("No known strategies; may need experimentation or research.")
        if profile.allow_experimentation:
            parts.append("Experimentation allowed.")
        return " ".join(parts)

    def _maybe_override_method(
        self,
        *,
        method: StrategySelectionMethod,
        has_known_strategies: bool,
        strategy_confidence: float,
        risk_level: RiskLevel,
        best: MetaStrategyProfile,
    ) -> StrategySelectionMethod:
        """Override the profile method based on runtime signals."""
        # If no known strategies and low confidence, force research
        if not has_known_strategies and strategy_confidence < 0.3:
            return StrategySelectionMethod.RESEARCH_DRIVEN

        # If very high risk and low confidence, force multi-strategy experiment
        if risk_level == RiskLevel.CRITICAL and strategy_confidence < 0.5:
            return StrategySelectionMethod.MULTI_STRATEGY_EXPERIMENT

        # If high confidence and known strategies, can use faster methods
        if strategy_confidence > 0.8 and has_known_strategies:
            if method == StrategySelectionMethod.MULTI_STRATEGY_EXPERIMENT:
                return StrategySelectionMethod.BENCHMARK_RANKING

        return method


__all__ = [
    "MetaStrategyDecision",
    "MetaStrategyProfile",
    "MetaStrategySelector",
    "RiskLevel",
    "StrategySelectionMethod",
    "TaskComplexity",
]
