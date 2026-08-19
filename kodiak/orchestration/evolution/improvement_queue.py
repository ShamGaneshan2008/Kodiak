"""Improvement queue — prioritized backlog for system evolution.

Maintains a controlled evolution backlog with proposals, evidence,
expected impact, implementation cost, risk, and benchmark requirements.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ImprovementStatus(enum.StrEnum):
    """Lifecycle states for an improvement proposal."""

    OBSERVED = "observed"
    PROPOSED = "proposed"
    EXPERIMENTAL = "experimental"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass
class ImprovementProposal:
    """A structured proposal for system improvement.

    Includes problem statement, evidence, expected impact, cost,
    risk, and benchmark requirements.
    """

    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    problem: str = ""
    evidence: tuple[str, ...] = ()
    expected_benefit: str = ""
    expected_impact_score: float = 0.5  # 0.0-1.0
    implementation_cost: float = 0.5  # 0.0-1.0
    risk: float = 0.5  # 0.0-1.0
    benchmark_requirement: str = ""
    status: ImprovementStatus = ImprovementStatus.OBSERVED
    related_capability_ids: tuple[str, ...] = ()
    related_strategy_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    rejection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def priority_score(self) -> float:
        """Compute priority from impact, cost, and risk."""
        if self.implementation_cost == 0:
            roi = self.expected_impact_score
        else:
            roi = self.expected_impact_score / self.implementation_cost
        risk_adjusted = roi * (1.0 - self.risk * 0.3)
        return max(0.0, min(1.0, risk_adjusted))

    @property
    def is_active(self) -> bool:
        return self.status in {
            ImprovementStatus.OBSERVED,
            ImprovementStatus.PROPOSED,
            ImprovementStatus.EXPERIMENTAL,
            ImprovementStatus.VALIDATING,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "problem": self.problem,
            "evidence": list(self.evidence),
            "expected_benefit": self.expected_benefit,
            "expected_impact_score": self.expected_impact_score,
            "implementation_cost": self.implementation_cost,
            "risk": self.risk,
            "priority_score": round(self.priority_score, 4),
            "benchmark_requirement": self.benchmark_requirement,
            "status": self.status.value,
            "related_capability_ids": list(self.related_capability_ids),
            "related_strategy_ids": list(self.related_strategy_ids),
            "tags": list(self.tags),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "rejection_reason": self.rejection_reason,
            "metadata": dict(self.metadata),
        }


class ImprovementQueue:
    """Manages a prioritized queue of improvement proposals.

    Proposals are sorted by priority score (impact / cost, risk-adjusted).
    The queue supports lifecycle transitions and provides views into
    active, experimental, and completed proposals.
    """

    def __init__(self, max_proposals: int = 200) -> None:
        self._proposals: dict[str, ImprovementProposal] = {}
        self._max_proposals = max_proposals
        self._log = logger.bind(component="improvement_queue")

    def add(self, proposal: ImprovementProposal) -> None:
        if len(self._proposals) >= self._max_proposals:
            self._evict()
        self._proposals[proposal.proposal_id] = proposal
        self._log.info(
            "proposal_added",
            proposal_id=proposal.proposal_id,
            title=proposal.title,
            status=proposal.status.value,
        )

    def get(self, proposal_id: str) -> ImprovementProposal | None:
        return self._proposals.get(proposal_id)

    def update_status(
        self,
        proposal_id: str,
        status: ImprovementStatus,
        *,
        rejection_reason: str = "",
    ) -> ImprovementProposal | None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = status
        proposal.updated_at = datetime.now(UTC)
        if status == ImprovementStatus.REJECTED:
            proposal.rejection_reason = rejection_reason
        if status in {ImprovementStatus.ACCEPTED, ImprovementStatus.REJECTED}:
            proposal.resolved_at = datetime.now(UTC)
        self._log.info(
            "proposal_status_updated",
            proposal_id=proposal_id,
            new_status=status.value,
        )
        return proposal

    def active_proposals(self) -> list[ImprovementProposal]:
        """Return active proposals sorted by priority."""
        active = [p for p in self._proposals.values() if p.is_active]
        active.sort(key=lambda p: p.priority_score, reverse=True)
        return active

    def ranked_proposals(self, limit: int = 10) -> list[ImprovementProposal]:
        """Return top proposals across all statuses, sorted by priority."""
        all_proposals = sorted(
            self._proposals.values(),
            key=lambda p: p.priority_score,
            reverse=True,
        )
        return all_proposals[:limit]

    def by_status(self, status: ImprovementStatus) -> list[ImprovementProposal]:
        return sorted(
            [p for p in self._proposals.values() if p.status == status],
            key=lambda p: p.priority_score,
            reverse=True,
        )

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for p in self._proposals.values():
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
        return {
            "total": len(self._proposals),
            "active": sum(1 for p in self._proposals.values() if p.is_active),
            "by_status": by_status,
        }

    def _evict(self) -> None:
        """Remove the lowest-priority resolved proposal."""
        resolved = [
            p
            for p in self._proposals.values()
            if p.status in {ImprovementStatus.REJECTED, ImprovementStatus.DEFERRED}
        ]
        if resolved:
            worst = min(resolved, key=lambda p: p.priority_score)
            del self._proposals[worst.proposal_id]
            self._log.info("proposal_evicted", proposal_id=worst.proposal_id)
            return

        # If nothing resolved, remove lowest-priority active
        if self._proposals:
            lowest = min(self._proposals.values(), key=lambda p: p.priority_score)
            del self._proposals[lowest.proposal_id]
            self._log.info("proposal_evicted", proposal_id=lowest.proposal_id)

    def __len__(self) -> int:
        return len(self._proposals)

    def __contains__(self, proposal_id: str) -> bool:
        return proposal_id in self._proposals


__all__ = ["ImprovementProposal", "ImprovementQueue", "ImprovementStatus"]
