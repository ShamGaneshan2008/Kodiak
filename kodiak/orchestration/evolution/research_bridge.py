"""Research → Evolution bridge.

Connects Phase 5 research discoveries with Phase 6 evolution.
Research discoveries should not automatically become core
architecture — they must pass through the improvement queue.

Flow:
    research discovers strategy
    → benchmark strategy
    → strategy becomes candidate capability
    → capability integrated experimentally
    → system benchmark
    → accept / reject
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from kodiak.orchestration.evolution.capability import (
    Capability,
    CapabilityCategory,
    CapabilityTracker,
)
from kodiak.orchestration.evolution.improvement_queue import (
    ImprovementProposal,
    ImprovementQueue,
    ImprovementStatus,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ResearchDiscovery:
    """A discovery from the research subsystem."""

    discovery_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    description: str = ""
    strategy_name: str = ""
    strategy_approach: str = ""
    problem_class: str = ""
    confidence: float = 0.5
    evidence: tuple[str, ...] = ()
    benchmark_results: dict[str, float] = field(default_factory=dict)
    source_experiment_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_id": self.discovery_id,
            "title": self.title,
            "description": self.description,
            "strategy_name": self.strategy_name,
            "strategy_approach": self.strategy_approach,
            "problem_class": self.problem_class,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "benchmark_results": dict(self.benchmark_results),
            "source_experiment_id": self.source_experiment_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """Result of bridging a research discovery to the evolution system."""

    result_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    discovery_id: str = ""
    proposal_id: str = ""
    capability_id: str = ""
    action_taken: str = ""  # "proposal_created", "capability_registered", "skipped"
    reasoning: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "discovery_id": self.discovery_id,
            "proposal_id": self.proposal_id,
            "capability_id": self.capability_id,
            "action_taken": self.action_taken,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
        }


class ResearchEvolutionBridge:
    """Bridges research discoveries to the evolution system.

    Does NOT automatically implement research discoveries.
    Creates improvement proposals that go through the normal
    approval workflow.
    """

    def __init__(
        self,
        *,
        improvement_queue: ImprovementQueue,
        capability_tracker: CapabilityTracker,
        min_confidence: float = 0.6,
    ) -> None:
        self._queue = improvement_queue
        self._tracker = capability_tracker
        self._min_confidence = min_confidence
        self._bridge_results: list[BridgeResult] = []
        self._log = logger.bind(component="research_evolution_bridge")

    def bridge_discovery(self, discovery: ResearchDiscovery) -> BridgeResult:
        """Bridge a research discovery to the evolution system.

        Decides whether to:
        - Create an improvement proposal
        - Register a new capability
        - Skip (discovery doesn't meet quality threshold)
        """
        # Check confidence threshold
        if discovery.confidence < self._min_confidence:
            result = BridgeResult(
                discovery_id=discovery.discovery_id,
                action_taken="skipped",
                reasoning=(
                    f"Discovery confidence ({discovery.confidence:.2f}) "
                    f"below threshold ({self._min_confidence:.2f})."
                ),
            )
            self._bridge_results.append(result)
            self._log.info(
                "discovery_skipped",
                discovery_id=discovery.discovery_id,
                confidence=discovery.confidence,
            )
            return result

        # Check if this strategy already exists as a capability
        existing = self._tracker.get_by_name(discovery.strategy_name)
        if existing is not None:
            # Update existing capability with new evidence
            self._update_capability_from_discovery(existing, discovery)
            result = BridgeResult(
                discovery_id=discovery.discovery_id,
                capability_id=existing.capability_id,
                action_taken="capability_updated",
                reasoning=f"Updated existing capability '{existing.name}' with new evidence.",
            )
            self._bridge_results.append(result)
            self._log.info(
                "capability_updated_from_research",
                capability_id=existing.capability_id,
                discovery_id=discovery.discovery_id,
            )
            return result

        # Create improvement proposal
        proposal = ImprovementProposal(
            title=f"Integrate research discovery: {discovery.title}",
            problem=(
                f"Research discovered a potentially better approach for {discovery.problem_class}"
            ),
            evidence=discovery.evidence
            + (
                f"Benchmark results: {discovery.benchmark_results}",
                f"Confidence: {discovery.confidence:.2f}",
            ),
            expected_benefit=discovery.description,
            expected_impact_score=min(discovery.confidence, 0.8),
            implementation_cost=0.4,
            risk=0.3,
            benchmark_requirement=(
                f"Benchmark strategy '{discovery.strategy_name}' against current approach "
                f"for problem class '{discovery.problem_class}'."
            ),
            status=ImprovementStatus.PROPOSED,
            tags=(discovery.problem_class, "research_discovery"),
            metadata={
                "discovery_id": discovery.discovery_id,
                "strategy_name": discovery.strategy_name,
                "strategy_approach": discovery.strategy_approach,
                "source_experiment_id": discovery.source_experiment_id,
            },
        )

        self._queue.add(proposal)

        result = BridgeResult(
            discovery_id=discovery.discovery_id,
            proposal_id=proposal.proposal_id,
            action_taken="proposal_created",
            reasoning=(
                f"Created improvement proposal for research discovery. "
                f"Strategy: '{discovery.strategy_name}' for {discovery.problem_class}. "
                f"Awaiting approval before implementation."
            ),
        )
        self._bridge_results.append(result)
        self._log.info(
            "proposal_created_from_research",
            discovery_id=discovery.discovery_id,
            proposal_id=proposal.proposal_id,
        )

        return result

    def bridge_multiple(self, discoveries: list[ResearchDiscovery]) -> list[BridgeResult]:
        """Bridge multiple discoveries."""
        return [self.bridge_discovery(d) for d in discoveries]

    def register_discovered_capability(
        self,
        discovery: ResearchDiscovery,
        *,
        validated: bool = False,
    ) -> Capability | None:
        """Register a research discovery as a new capability.

        Only registers if validated=True (experiment confirmed improvement).
        """
        if not validated:
            self._log.warning(
                "capability_not_validated",
                discovery_id=discovery.discovery_id,
            )
            return None

        category = self._category_for_problem(discovery.problem_class)
        capability = Capability(
            name=discovery.strategy_name,
            category=category,
            description=discovery.description,
            evidence=discovery.evidence,
            metadata={
                "source": "research_discovery",
                "discovery_id": discovery.discovery_id,
                "validated": True,
            },
        )

        self._tracker.register(capability)
        self._log.info(
            "capability_registered_from_research",
            capability_id=capability.capability_id,
            name=capability.name,
        )
        return capability

    def all_results(self) -> list[BridgeResult]:
        return list(self._bridge_results)

    def stats(self) -> dict[str, Any]:
        actions = {}
        for r in self._bridge_results:
            actions[r.action_taken] = actions.get(r.action_taken, 0) + 1
        return {
            "total_bridged": len(self._bridge_results),
            "by_action": actions,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_capability_from_discovery(
        self, capability: Capability, discovery: ResearchDiscovery
    ) -> None:
        """Update an existing capability with new research evidence."""
        existing_evidence = list(capability.evidence)
        new_evidence = list(discovery.evidence)
        combined = existing_evidence + new_evidence

        # Update in-place (Capability is mutable)
        capability.evidence = tuple(combined[:15])  # Cap evidence list
        capability.metadata["last_research_update"] = datetime.now(UTC).isoformat()
        capability.metadata["research_discovery_id"] = discovery.discovery_id

    @staticmethod
    def _category_for_problem(problem_class: str) -> CapabilityCategory:
        """Map a problem class to a capability category."""
        mapping = {
            "test_failure": CapabilityCategory.TESTING,
            "syntax_error": CapabilityCategory.CODE_GENERATION,
            "type_error": CapabilityCategory.CODE_ANALYSIS,
            "lint_failure": CapabilityCategory.CODE_ANALYSIS,
            "missing_dependency": CapabilityCategory.TOOL_USE,
            "timeout": CapabilityCategory.RECOVERY,
            "incorrect_implementation": CapabilityCategory.CODE_GENERATION,
            "execution_failure": CapabilityCategory.RECOVERY,
            "architectural": CapabilityCategory.PLANNING,
            "performance": CapabilityCategory.REPOSITORY_ANALYSIS,
            "dependency_conflict": CapabilityCategory.TOOL_USE,
        }
        return mapping.get(problem_class, CapabilityCategory.RECOVERY)


__all__ = [
    "BridgeResult",
    "ResearchDiscovery",
    "ResearchEvolutionBridge",
]
