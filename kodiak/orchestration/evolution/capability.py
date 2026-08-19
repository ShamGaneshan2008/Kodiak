"""Capability model — structured representation of Kodiak's capabilities.

Each capability has measurable evidence.  Do not claim a capability
merely because a corresponding module exists — it must demonstrate
value through execution history.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CapabilityCategory(enum.StrEnum):
    """High-level categories of capabilities."""

    PLANNING = "planning"
    DEBUGGING = "debugging"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    TESTING = "testing"
    MEMORY_RETRIEVAL = "memory_retrieval"
    VERIFICATION = "verification"
    RESEARCH = "research"
    STRATEGY_DISCOVERY = "strategy_discovery"
    TOOL_USE = "tool_use"
    REPOSITORY_ANALYSIS = "repository_analysis"
    RECOVERY = "recovery"
    META_LEARNING = "meta_learning"


@dataclass(frozen=True, slots=True)
class CapabilityPerformance:
    """Measurable performance data for a capability."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    avg_duration_seconds: float = 0.0
    avg_cost: float = 0.0
    common_failure_modes: tuple[str, ...] = ()
    task_categories: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate": round(self.success_rate, 4),
            "avg_duration_seconds": self.avg_duration_seconds,
            "avg_cost": self.avg_cost,
            "common_failure_modes": list(self.common_failure_modes),
            "task_categories": list(self.task_categories),
        }


@dataclass
class Capability:
    """A measurable system capability.

    Tracks performance, common failure modes, and evidence.
    """

    capability_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    category: CapabilityCategory = CapabilityCategory.PLANNING
    description: str = ""
    performance: CapabilityPerformance = field(default_factory=CapabilityPerformance)
    evidence: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_evaluated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def health_score(self) -> float:
        """Health score based on performance and evidence."""
        base = self.performance.success_rate
        evidence_bonus = min(len(self.evidence) * 0.02, 0.15)
        limitation_penalty = min(len(self.known_limitations) * 0.03, 0.2)
        return max(0.0, min(1.0, base + evidence_bonus - limitation_penalty))

    @property
    def is_healthy(self) -> bool:
        return self.health_score >= 0.5 and self.is_active

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "performance": self.performance.to_dict(),
            "health_score": round(self.health_score, 4),
            "is_healthy": self.is_healthy,
            "evidence": list(self.evidence),
            "known_limitations": list(self.known_limitations),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_evaluated_at": self.last_evaluated_at.isoformat()
            if self.last_evaluated_at
            else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CapabilityEvaluation:
    """A structured evaluation of a single capability."""

    evaluation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    capability_id: str = ""
    score: float = 0.0
    evidence: str = ""
    limitations_found: tuple[str, ...] = ()
    recommendation: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "capability_id": self.capability_id,
            "score": self.score,
            "evidence": self.evidence,
            "limitations_found": list(self.limitations_found),
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
        }


class CapabilityTracker:
    """Tracks and evaluates system capabilities.

    Maintains a registry of capabilities and updates them based on
    execution history.  Answers: "I am strong at X but weak at Y."
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._evaluations: list[CapabilityEvaluation] = []
        self._log = logger.bind(component="capability_tracker")

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.capability_id] = capability
        self._log.info(
            "capability_registered",
            capability_id=capability.capability_id,
            name=capability.name,
            category=capability.category.value,
        )

    def get(self, capability_id: str) -> Capability | None:
        return self._capabilities.get(capability_id)

    def get_by_name(self, name: str) -> Capability | None:
        for cap in self._capabilities.values():
            if cap.name == name:
                return cap
        return None

    def all_capabilities(self) -> list[Capability]:
        return sorted(
            self._capabilities.values(),
            key=lambda c: c.health_score,
            reverse=True,
        )

    def record_outcome(
        self,
        capability_id: str,
        *,
        success: bool,
        duration_seconds: float = 0.0,
        failure_mode: str | None = None,
    ) -> None:
        """Record an execution outcome for a capability."""
        cap = self._capabilities.get(capability_id)
        if cap is None:
            return

        # Update performance in-place
        old = cap.performance
        total = old.total_attempts + 1
        successful = old.successful_attempts + (1 if success else 0)
        failed = old.failed_attempts + (0 if success else 1)
        avg_duration = (
            (old.avg_duration_seconds * old.total_attempts + duration_seconds) / total
            if total > 0
            else duration_seconds
        )

        failure_modes = list(old.common_failure_modes)
        if not success and failure_mode and failure_mode not in failure_modes:
            failure_modes.append(failure_mode)

        cap.performance = CapabilityPerformance(
            total_attempts=total,
            successful_attempts=successful,
            failed_attempts=failed,
            avg_duration_seconds=avg_duration,
            avg_cost=old.avg_cost,
            common_failure_modes=tuple(failure_modes[:5]),
            task_categories=old.task_categories,
        )
        cap.last_evaluated_at = datetime.now(UTC)

        self._log.info(
            "capability_outcome_recorded",
            capability_id=capability_id,
            success=success,
            new_success_rate=round(cap.performance.success_rate, 3),
        )

    def evaluate(self, capability_id: str) -> CapabilityEvaluation | None:
        """Produce a structured evaluation of a capability."""
        cap = self._capabilities.get(capability_id)
        if cap is None:
            return None

        score = cap.health_score
        evidence_parts = []
        if cap.performance.total_attempts > 0:
            evidence_parts.append(
                f"Success rate: {cap.performance.success_rate:.1%} "
                f"over {cap.performance.total_attempts} attempts"
            )
        if cap.evidence:
            evidence_parts.append(f"{len(cap.evidence)} evidence items")
        if cap.known_limitations:
            evidence_parts.append(f"{len(cap.known_limitations)} known limitations")

        limitations_found = cap.known_limitations
        if cap.performance.success_rate < 0.5 and cap.performance.total_attempts >= 3:
            limitations_found = limitations_found + ("Low success rate",)

        recommendation = ""
        if score < 0.4:
            recommendation = f"Capability '{cap.name}' needs significant improvement."
        elif score < 0.6:
            recommendation = f"Capability '{cap.name}' has room for improvement."
        else:
            recommendation = f"Capability '{cap.name}' is performing adequately."

        evaluation = CapabilityEvaluation(
            capability_id=capability_id,
            score=score,
            evidence="; ".join(evidence_parts) if evidence_parts else "No execution data",
            limitations_found=tuple(limitations_found),
            recommendation=recommendation,
        )

        self._evaluations.append(evaluation)
        self._log.info(
            "capability_evaluated",
            capability_id=capability_id,
            score=round(score, 3),
        )
        return evaluation

    def strong_capabilities(self, min_health: float = 0.7) -> list[Capability]:
        return [c for c in self.all_capabilities() if c.health_score >= min_health]

    def weak_capabilities(self, max_health: float = 0.5) -> list[Capability]:
        return [c for c in self.all_capabilities() if c.health_score <= max_health]

    def missing_capabilities(
        self, required: frozenset[str]
    ) -> frozenset[str]:
        """Identify required capabilities that are not registered."""
        registered = {c.name for c in self._capabilities.values() if c.is_active}
        return required - registered

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": [c.to_dict() for c in self.all_capabilities()],
            "total": len(self._capabilities),
            "healthy": sum(1 for c in self._capabilities.values() if c.is_healthy),
            "unhealthy": sum(
                1 for c in self._capabilities.values() if not c.is_healthy
            ),
        }


__all__ = [
    "Capability",
    "CapabilityCategory",
    "CapabilityEvaluation",
    "CapabilityPerformance",
    "CapabilityTracker",
]
