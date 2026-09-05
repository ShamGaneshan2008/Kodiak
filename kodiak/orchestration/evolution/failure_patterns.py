"""Failure-pattern mining — detect recurring failures across executions.

Periodically analyzes execution history for recurring patterns:
same failure, same recovery, same unnecessary step, same tool mistake,
same verification gap, same planning error.
"""

from __future__ import annotations

import enum
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class FailurePatternSeverity(enum.StrEnum):
    """Severity of a detected failure pattern."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class FailurePattern:
    """A recurring failure pattern detected across executions."""

    pattern_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""
    category: str = ""
    severity: FailurePatternSeverity = FailurePatternSeverity.MEDIUM
    occurrence_count: int = 0
    affected_task_ids: tuple[str, ...] = ()
    affected_components: tuple[str, ...] = ()
    suggested_improvement: str = ""
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_recurring(self) -> bool:
        return self.occurrence_count >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "category": self.category,
            "severity": self.severity.value,
            "occurrence_count": self.occurrence_count,
            "affected_task_ids": list(self.affected_task_ids),
            "affected_components": list(self.affected_components),
            "suggested_improvement": self.suggested_improvement,
            "confidence": self.confidence,
            "is_recurring": self.is_recurring,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


class FailurePatternMiner:
    """Analyzes execution evaluations for recurring failure patterns.

    Examines task evaluations to identify systemic issues that repeat
    across multiple executions.  Repeated patterns indicate potential
    system improvements.
    """

    def __init__(self, min_occurrences: int = 2) -> None:
        self._min_occurrences = min_occurrences
        self._patterns: dict[str, FailurePattern] = {}
        self._log = logger.bind(component="failure_pattern_miner")

    def analyze_evaluations(
        self,
        evaluations: list[dict[str, Any]],
    ) -> list[FailurePattern]:
        """Analyze a batch of task evaluations for failure patterns.

        Each evaluation should be a dict with keys like:
        - task_id, goal, what_failed, wasted_effort, wrong_assumptions,
          failure_component, dimension_scores
        """
        if not evaluations:
            return []

        # Collect failure signals
        failure_components: Counter[str] = Counter()
        failure_messages: Counter[str] = Counter()
        wasted_efforts: Counter[str] = Counter()
        weak_dims: Counter[str] = Counter()
        task_ids_by_failure: dict[str, list[str]] = {}
        task_ids_by_waste: dict[str, list[str]] = {}

        for eval_ in evaluations:
            task_id = eval_.get("task_id", "unknown")

            component = eval_.get("failure_component", "")
            if component:
                failure_components[component] += 1
                task_ids_by_failure.setdefault(component, []).append(task_id)

            for msg in eval_.get("what_failed", []):
                failure_messages[msg] += 1
                task_ids_by_failure.setdefault(msg, []).append(task_id)

            for waste in eval_.get("wasted_effort", []):
                wasted_efforts[waste] += 1
                task_ids_by_waste.setdefault(waste, []).append(task_id)

            for ds in eval_.get("dimension_scores", []):
                if isinstance(ds, dict) and ds.get("verdict") in ("weak", "failing"):
                    weak_dims[ds.get("dimension", "unknown")] += 1

        patterns: list[FailurePattern] = []

        # Detect recurring failure components
        for component, count in failure_components.most_common():
            if count >= self._min_occurrences:
                pattern = FailurePattern(
                    description=f"Component '{component}' fails repeatedly",
                    category="component_failure",
                    severity=self._severity_for_count(count),
                    occurrence_count=count,
                    affected_task_ids=tuple(dict.fromkeys(task_ids_by_failure.get(component, []))),
                    affected_components=(component,),
                    suggested_improvement=f"Investigate and improve '{component}' component",
                    confidence=min(count / 10.0, 0.95),
                )
                patterns.append(pattern)

        # Detect recurring failure messages
        for msg, count in failure_messages.most_common(5):
            if count >= self._min_occurrences:
                pattern = FailurePattern(
                    description=f"Recurring failure: {msg[:80]}",
                    category="recurring_failure",
                    severity=self._severity_for_count(count),
                    occurrence_count=count,
                    affected_task_ids=tuple(dict.fromkeys(task_ids_by_failure.get(msg, []))),
                    suggested_improvement=f"Address root cause of: {msg[:80]}",
                    confidence=min(count / 10.0, 0.9),
                )
                patterns.append(pattern)

        # Detect recurring waste patterns
        for waste, count in wasted_efforts.most_common(3):
            if count >= self._min_occurrences:
                pattern = FailurePattern(
                    description=f"Recurring wasted effort: {waste}",
                    category="wasted_effort",
                    severity=FailurePatternSeverity.LOW,
                    occurrence_count=count,
                    affected_task_ids=tuple(dict.fromkeys(task_ids_by_waste.get(waste, []))),
                    suggested_improvement=f"Eliminate: {waste}",
                    confidence=min(count / 10.0, 0.8),
                )
                patterns.append(pattern)

        # Detect weak dimensions across evaluations
        for dim, count in weak_dims.most_common(3):
            if count >= self._min_occurrences:
                pattern = FailurePattern(
                    description=f"Dimension '{dim}' is consistently weak",
                    category="weak_dimension",
                    severity=self._severity_for_count(count),
                    occurrence_count=count,
                    affected_components=(dim,),
                    suggested_improvement=f"Improve system capability: {dim}",
                    confidence=min(count / 10.0, 0.85),
                )
                patterns.append(pattern)

        # Store patterns
        for pattern in patterns:
            self._patterns[pattern.pattern_id] = pattern

        self._log.info(
            "failure_patterns_detected",
            count=len(patterns),
            evaluations_analyzed=len(evaluations),
        )

        return patterns

    def all_patterns(self) -> list[FailurePattern]:
        return sorted(
            self._patterns.values(),
            key=lambda p: (p.occurrence_count, p.confidence),
            reverse=True,
        )

    def recurring_patterns(self) -> list[FailurePattern]:
        return [p for p in self.all_patterns() if p.is_recurring]

    def critical_patterns(self) -> list[FailurePattern]:
        return [
            p
            for p in self.all_patterns()
            if p.severity in {FailurePatternSeverity.CRITICAL, FailurePatternSeverity.HIGH}
        ]

    def patterns_for_component(self, component: str) -> list[FailurePattern]:
        return [p for p in self.all_patterns() if component in p.affected_components]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patterns": [p.to_dict() for p in self.all_patterns()],
            "total": len(self._patterns),
            "recurring": len(self.recurring_patterns()),
            "critical": len(self.critical_patterns()),
        }

    @staticmethod
    def _severity_for_count(count: int) -> FailurePatternSeverity:
        if count >= 5:
            return FailurePatternSeverity.CRITICAL
        if count >= 3:
            return FailurePatternSeverity.HIGH
        return FailurePatternSeverity.MEDIUM


__all__ = [
    "FailurePattern",
    "FailurePatternMiner",
    "FailurePatternSeverity",
]
