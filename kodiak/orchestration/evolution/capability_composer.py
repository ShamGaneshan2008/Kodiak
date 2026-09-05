"""Capability composition — combine existing capabilities.

Allow existing capabilities to be combined into composite capabilities.
Test whether compositions actually outperform individual capabilities.

Example:
    repository_analysis
    + dependency_analysis
    + impact_analysis
    = migration_planning_capability
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from kodiak.orchestration.evolution.capability import (
    Capability,
    CapabilityPerformance,
    CapabilityTracker,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CompositionResult:
    """Result of evaluating a capability composition."""

    result_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    composition_name: str = ""
    component_ids: tuple[str, ...] = ()
    component_names: tuple[str, ...] = ()
    composite_score: float = 0.0
    component_avg_score: float = 0.0
    improvement: float = 0.0
    is_improvement: bool = False
    evidence: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "composition_name": self.composition_name,
            "component_ids": list(self.component_ids),
            "component_names": list(self.component_names),
            "composite_score": round(self.composite_score, 4),
            "component_avg_score": round(self.component_avg_score, 4),
            "improvement": round(self.improvement, 4),
            "is_improvement": self.is_improvement,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
        }


class CapabilityComposer:
    """Combines capabilities and evaluates whether compositions help.

    Does NOT assume combining good capabilities automatically creates
    a better one — the composition must demonstrate improvement.
    """

    def __init__(self, tracker: CapabilityTracker) -> None:
        self._tracker = tracker
        self._compositions: dict[str, list[str]] = {}  # name → component IDs
        self._results: list[CompositionResult] = []
        self._log = logger.bind(component="capability_composer")

    def compose(
        self,
        capability_ids: tuple[str, ...],
        *,
        name: str = "",
    ) -> Capability | None:
        """Create a composite capability from existing capabilities.

        The composite inherits the best traits of its components.
        Does NOT automatically register — must be validated first.
        """
        components = []
        for cid in capability_ids:
            cap = self._tracker.get(cid)
            if cap is not None:
                components.append(cap)

        if len(components) < 2:
            self._log.warning(
                "compose_requires_two",
                provided=len(components),
            )
            return None

        # Build composite
        all_names = [c.name for c in components]
        composite_name = name or "_plus_".join(all_names[:3])

        # Combine capabilities
        all_capabilities: set[str] = set()
        for c in components:
            all_capabilities.update(c.performance.task_categories)

        # Combined failure modes
        all_failure_modes: list[str] = []
        for c in components:
            for fm in c.performance.common_failure_modes:
                if fm not in all_failure_modes:
                    all_failure_modes.append(fm)

        # Average performance
        total_attempts = sum(c.performance.total_attempts for c in components)
        total_success = sum(c.performance.successful_attempts for c in components)
        avg_duration = (
            sum(
                c.performance.avg_duration_seconds * c.performance.total_attempts
                for c in components
            )
            / total_attempts
            if total_attempts > 0
            else 0.0
        )

        # Evidence from components
        evidence_parts = []
        for c in components:
            if c.evidence:
                evidence_parts.extend(c.evidence[:2])
        # Add composition evidence
        evidence_parts.append(
            f"Composed from {len(components)} capabilities: {', '.join(all_names)}"
        )

        # Known limitations from all components
        all_limitations: list[str] = []
        for c in components:
            all_limitations.extend(c.known_limitations)
        # Add composition limitation
        all_limitations.append("Composition effectiveness not yet validated")

        composite = Capability(
            name=composite_name,
            category=components[0].category,  # Use first component's category
            description=f"Composite of: {', '.join(all_names)}",
            performance=CapabilityPerformance(
                total_attempts=total_attempts,
                successful_attempts=total_success,
                failed_attempts=total_attempts - total_success,
                avg_duration_seconds=avg_duration,
                avg_cost=sum(c.performance.avg_cost for c in components) / len(components),
                common_failure_modes=tuple(all_failure_modes[:5]),
                task_categories=tuple(sorted(all_capabilities)),
            ),
            evidence=tuple(evidence_parts[:10]),
            known_limitations=tuple(all_limitations[:5]),
            metadata={
                "composition": True,
                "component_ids": list(capability_ids),
                "component_names": all_names,
            },
        )

        self._compositions[composite_name] = list(capability_ids)
        self._log.info(
            "capability_composed",
            name=composite_name,
            components=len(components),
        )

        return composite

    def evaluate_composition(
        self,
        composite: Capability,
        *,
        test_results: dict[str, bool] | None = None,
    ) -> CompositionResult:
        """Evaluate whether a composition outperforms its components.

        If test_results is provided, uses actual test outcomes.
        Otherwise, uses theoretical projection from component performance.
        """
        component_ids = composite.metadata.get("component_ids", [])
        components = []
        for cid in component_ids:
            cap = self._tracker.get(cid)
            if cap is not None:
                components.append(cap)

        if not components:
            return CompositionResult(
                composition_name=composite.name,
                evidence="No component capabilities found for evaluation.",
            )

        # Component average score
        component_scores = [c.health_score for c in components]
        component_avg = sum(component_scores) / len(component_scores)

        # Composite score
        if test_results:
            # Use actual test results
            successes = sum(1 for v in test_results.values() if v)
            total = len(test_results)
            composite_score = successes / total if total > 0 else 0.0
            evidence = (
                f"Tested on {total} tasks: {successes} succeeded, {total - successes} failed."
            )
        else:
            # Theoretical projection
            composite_score = composite.health_score
            evidence = (
                f"Theoretical projection based on component performance. "
                f"Composite health score: {composite_score:.3f}."
            )

        improvement = composite_score - component_avg
        is_improvement = improvement > 0.05  # At least 5% improvement

        result = CompositionResult(
            composition_name=composite.name,
            component_ids=tuple(component_ids),
            component_names=tuple(c.name for c in components),
            composite_score=composite_score,
            component_avg_score=component_avg,
            improvement=improvement,
            is_improvement=is_improvement,
            evidence=evidence,
        )

        self._results.append(result)
        self._log.info(
            "composition_evaluated",
            name=composite.name,
            improvement=f"{improvement:+.3f}",
            is_improvement=is_improvement,
        )

        return result

    def suggest_compositions(
        self,
        *,
        min_health: float = 0.5,
        max_components: int = 4,
    ) -> list[tuple[str, ...]]:
        """Suggest potential compositions from strong capabilities.

        Returns tuples of capability IDs that might compose well.
        """
        strong = [
            c
            for c in self._tracker.all_capabilities()
            if c.health_score >= min_health and c.is_active
        ]

        suggestions: list[tuple[str, ...]] = []

        # Group by category
        by_category: dict[str, list[Capability]] = {}
        for cap in strong:
            by_category.setdefault(cap.category.value, []).append(cap)

        # Suggest within-category compositions (2-3 components)
        for caps in by_category.values():
            if len(caps) >= 2:
                for i in range(len(caps)):
                    for j in range(i + 1, min(len(caps), i + max_components)):
                        pair = tuple(sorted(c.capability_id for c in caps[i : j + 1]))
                        if pair not in [tuple(s) for s in suggestions]:
                            suggestions.append(pair)

        # Suggest cross-category compositions (pairs only)
        categories = list(by_category.keys())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                caps_a = by_category[categories[i]]
                caps_b = by_category[categories[j]]
                # Pick best from each
                best_a = max(caps_a, key=lambda c: c.health_score)
                best_b = max(caps_b, key=lambda c: c.health_score)
                pair = tuple(sorted([best_a.capability_id, best_b.capability_id]))
                if pair not in [tuple(s) for s in suggestions]:
                    suggestions.append(pair)

        return suggestions[:20]  # Limit suggestions

    def all_results(self) -> list[CompositionResult]:
        return sorted(
            self._results,
            key=lambda r: r.improvement,
            reverse=True,
        )

    def successful_compositions(self) -> list[CompositionResult]:
        return [r for r in self._results if r.is_improvement]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compositions": list(self._compositions.keys()),
            "results": [r.to_dict() for r in self._results],
            "total_evaluated": len(self._results),
            "successful": len(self.successful_compositions()),
        }


__all__ = [
    "CapabilityComposer",
    "CompositionResult",
]
