"""System health dashboard — evidence-based health tracking.

Builds a high-level health model for Kodiak that reflects actual
evidence rather than aspirational metrics.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class HealthDimension(enum.StrEnum):
    """Dimensions of system health."""

    RELIABILITY = "reliability"
    ADAPTABILITY = "adaptability"
    AUTONOMY = "autonomy"
    VERIFICATION_QUALITY = "verification_quality"
    MEMORY_QUALITY = "memory_quality"
    RESEARCH_EFFECTIVENESS = "research_effectiveness"
    STRATEGY_QUALITY = "strategy_quality"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    REGRESSION_RATE = "regression_rate"
    HUMAN_INTERVENTION = "human_intervention"


class HealthStatus(enum.StrEnum):
    """Health status for a dimension or overall system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


def _status_for_score(score: float) -> HealthStatus:
    if score >= 0.7:
        return HealthStatus.HEALTHY
    if score >= 0.4:
        return HealthStatus.DEGRADED
    if score > 0.0:
        return HealthStatus.UNHEALTHY
    return HealthStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class HealthMetric:
    """A single health metric with evidence."""

    dimension: HealthDimension
    score: float  # 0.0-1.0
    status: HealthStatus
    evidence: str = ""
    confidence: float = 0.5
    trend: str = "stable"  # "improving", "stable", "declining"
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "status": self.status.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "trend": self.trend,
            "measurements": dict(self.measurements),
        }


@dataclass(frozen=True, slots=True)
class SystemHealth:
    """System-wide health snapshot."""

    health_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    overall_score: float = 0.0
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    metrics: tuple[HealthMetric, ...] = ()
    weakest_dimension: HealthDimension | None = None
    strongest_dimension: HealthDimension | None = None
    alerts: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_id": self.health_id,
            "overall_score": self.overall_score,
            "overall_status": self.overall_status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "weakest_dimension": self.weakest_dimension.value
            if self.weakest_dimension
            else None,
            "strongest_dimension": self.strongest_dimension.value
            if self.strongest_dimension
            else None,
            "alerts": list(self.alerts),
            "created_at": self.created_at.isoformat(),
        }


class SystemHealthDashboard:
    """Computes and tracks system health from evaluation data.

    Health is evidence-based: each metric carries measurements,
    confidence, and trend data.
    """

    def __init__(self) -> None:
        self._history: list[SystemHealth] = []
        self._log = logger.bind(component="system_health_dashboard")

    def compute_health(
        self,
        *,
        total_tasks: int = 0,
        successful_tasks: int = 0,
        failed_tasks: int = 0,
        total_replans: int = 0,
        total_retries: int = 0,
        verification_pass_rate: float = 0.0,
        memory_usefulness: float = 0.0,
        research_tasks: int = 0,
        strategy_use_count: int = 0,
        avg_duration_seconds: float = 0.0,
        human_interventions: int = 0,
        dimension_scores: dict[str, float] | None = None,
    ) -> SystemHealth:
        """Compute system health from aggregate metrics."""
        metrics: list[HealthMetric] = []

        # Reliability
        reliability = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        metrics.append(HealthMetric(
            dimension=HealthDimension.RELIABILITY,
            score=reliability,
            status=_status_for_score(reliability),
            evidence=f"{successful_tasks}/{total_tasks} tasks succeeded",
            measurements={"success_rate": reliability, "total": total_tasks},
        ))

        # Adaptability (inferred from replan rate)
        if total_tasks > 0:
            adaptability = max(0.0, 1.0 - (total_replans / total_tasks) * 0.5)
        else:
            adaptability = 0.5
        metrics.append(HealthMetric(
            dimension=HealthDimension.ADAPTABILITY,
            score=adaptability,
            status=_status_for_score(adaptability),
            evidence=f"{total_replans} replan(s) across {total_tasks} tasks",
            measurements={"replan_rate": total_replans / max(total_tasks, 1)},
        ))

        # Autonomy (inferred from human intervention rate)
        if total_tasks > 0:
            autonomy = max(0.0, 1.0 - (human_interventions / total_tasks))
        else:
            autonomy = 0.8
        metrics.append(HealthMetric(
            dimension=HealthDimension.AUTONOMY,
            score=autonomy,
            status=_status_for_score(autonomy),
            evidence=f"{human_interventions} human intervention(s) in {total_tasks} tasks",
            measurements={"intervention_rate": human_interventions / max(total_tasks, 1)},
        ))

        # Verification quality
        metrics.append(HealthMetric(
            dimension=HealthDimension.VERIFICATION_QUALITY,
            score=verification_pass_rate,
            status=_status_for_score(verification_pass_rate),
            evidence=f"Verification pass rate: {verification_pass_rate:.1%}",
        ))

        # Memory quality
        metrics.append(HealthMetric(
            dimension=HealthDimension.MEMORY_QUALITY,
            score=memory_usefulness,
            status=_status_for_score(memory_usefulness),
            evidence=f"Memory usefulness: {memory_usefulness:.1%}",
        ))

        # Research effectiveness
        research_score = min(research_tasks / max(total_tasks, 1) * 2, 1.0)
        metrics.append(HealthMetric(
            dimension=HealthDimension.RESEARCH_EFFECTIVENESS,
            score=research_score,
            status=_status_for_score(research_score),
            evidence=f"{research_tasks} research tasks completed",
        ))

        # Strategy quality
        strategy_score = min(strategy_use_count / max(total_tasks, 1), 1.0)
        metrics.append(HealthMetric(
            dimension=HealthDimension.STRATEGY_QUALITY,
            score=strategy_score,
            status=_status_for_score(strategy_score),
            evidence=f"Strategy used in {strategy_use_count} of {total_tasks} tasks",
        ))

        # Resource efficiency
        efficiency = max(0.0, 1.0 - avg_duration_seconds / 300.0) if avg_duration_seconds > 0 else 0.7
        metrics.append(HealthMetric(
            dimension=HealthDimension.RESOURCE_EFFICIENCY,
            score=efficiency,
            status=_status_for_score(efficiency),
            evidence=f"Average duration: {avg_duration_seconds:.1f}s",
            measurements={"avg_duration_seconds": avg_duration_seconds},
        ))

        # Regression rate
        regression_rate = failed_tasks / total_tasks if total_tasks > 0 else 0.0
        regression_health = 1.0 - regression_rate
        metrics.append(HealthMetric(
            dimension=HealthDimension.REGRESSION_RATE,
            score=regression_health,
            status=_status_for_score(regression_health),
            evidence=f"Failure rate: {regression_rate:.1%}",
        ))

        # Human intervention
        intervention_health = autonomy  # Same calculation
        metrics.append(HealthMetric(
            dimension=HealthDimension.HUMAN_INTERVENTION,
            score=intervention_health,
            status=_status_for_score(intervention_health),
            evidence=f"{human_interventions} intervention(s)",
        ))

        # Override with dimension_scores if provided
        if dimension_scores:
            for metric in metrics:
                if metric.dimension.value in dimension_scores:
                    override = dimension_scores[metric.dimension.value]
                    metrics[metrics.index(metric)] = HealthMetric(
                        dimension=metric.dimension,
                        score=override,
                        status=_status_for_score(override),
                        evidence=metric.evidence,
                        confidence=metric.confidence,
                        trend=metric.trend,
                        measurements=metric.measurements,
                    )

        # Overall
        overall_score = sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
        overall_status = _status_for_score(overall_score)

        # Find weakest and strongest
        sorted_metrics = sorted(metrics, key=lambda m: m.score)
        weakest = sorted_metrics[0].dimension if sorted_metrics else None
        strongest = sorted_metrics[-1].dimension if sorted_metrics else None

        # Generate alerts
        alerts: list[str] = []
        for m in metrics:
            if m.status == HealthStatus.UNHEALTHY:
                alerts.append(f"ALERT: {m.dimension.value} is unhealthy ({m.score:.2f})")
            elif m.status == HealthStatus.DEGRADED and m.score < 0.5:
                alerts.append(f"WARNING: {m.dimension.value} is degraded ({m.score:.2f})")

        health = SystemHealth(
            overall_score=overall_score,
            overall_status=overall_status,
            metrics=tuple(metrics),
            weakest_dimension=weakest,
            strongest_dimension=strongest,
            alerts=tuple(alerts),
        )

        self._history.append(health)
        self._log.info(
            "health_computed",
            overall_score=round(overall_score, 3),
            overall_status=overall_status.value,
            alerts=len(alerts),
        )

        return health

    def latest_health(self) -> SystemHealth | None:
        return self._history[-1] if self._history else None

    def health_history(self, limit: int = 10) -> list[SystemHealth]:
        return self._history[-limit:]

    def trend(self, dimension: HealthDimension) -> str:
        """Determine trend for a dimension over recent history."""
        if len(self._history) < 2:
            return "insufficient_data"

        recent = self._history[-5:]
        scores = []
        for h in recent:
            for m in h.metrics:
                if m.dimension == dimension:
                    scores.append(m.score)
                    break

        if len(scores) < 2:
            return "insufficient_data"

        avg_first = sum(scores[: len(scores) // 2]) / max(len(scores) // 2, 1)
        avg_second = sum(scores[len(scores) // 2 :]) / max(len(scores) - len(scores) // 2, 1)

        if avg_second > avg_first + 0.05:
            return "improving"
        if avg_second < avg_first - 0.05:
            return "declining"
        return "stable"

    def to_dict(self) -> dict[str, Any]:
        latest = self.latest_health()
        return {
            "latest": latest.to_dict() if latest else None,
            "history_count": len(self._history),
        }


__all__ = [
    "HealthDimension",
    "HealthMetric",
    "HealthStatus",
    "SystemHealth",
    "SystemHealthDashboard",
]
