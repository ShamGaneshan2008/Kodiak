"""Research prioritization engine.

Ranks research problems according to multiple criteria to determine
which investigations provide the highest value.  Prevents wasting
resources on low-impact research when high-impact work is available.
"""

from __future__ import annotations

from typing import Any

import structlog

from kodiak.orchestration.research.models import (
    ResearchProblem,
    ResearchProblemPriority,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

# Weight for each criterion (sums to 1.0)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "potential_impact": 0.25,
    "uncertainty": 0.20,
    "frequency": 0.15,
    "cost_of_failure": 0.15,
    "reusability": 0.15,
    "measurement_feasibility": 0.10,
}


class ResearchPrioritizer:
    """Ranks research problems by multi-criteria scoring.

    Each criterion is scored 0.0–1.0, then weighted to produce a final
    priority score.  Problems with higher scores should be investigated
    first.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or dict(_DEFAULT_WEIGHTS)
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}
        self._log = logger.bind(component="research_prioritizer")

    def score_problem(
        self,
        problem: ResearchProblem,
        *,
        frequency: float = 0.5,
        cost_of_failure: float = 0.5,
        reusability: float = 0.5,
        measurement_feasibility: float = 0.5,
    ) -> float:
        """Score a research problem on multiple criteria.

        Args:
            problem: The research problem to score.
            frequency: How often this problem class occurs (0.0-1.0).
            cost_of_failure: How costly it is to fail to solve this (0.0-1.0).
            reusability: How reusable the solution would be (0.0-1.0).
            measurement_feasibility: How easily results can be measured (0.0-1.0).

        Returns:
            Composite priority score between 0.0 and 1.0.
        """
        potential_impact = self._estimate_impact(problem)
        uncertainty = self._estimate_uncertainty(problem)

        scores = {
            "potential_impact": potential_impact,
            "uncertainty": uncertainty,
            "frequency": frequency,
            "cost_of_failure": cost_of_failure,
            "reusability": reusability,
            "measurement_feasibility": measurement_feasibility,
        }

        total = sum(
            scores.get(criterion, 0.0) * weight for criterion, weight in self._weights.items()
        )

        return max(0.0, min(1.0, total))

    def rank_problems(
        self,
        problems: list[ResearchProblem],
        **kwargs: float,
    ) -> list[tuple[ResearchProblem, float]]:
        """Rank a list of research problems by priority score.

        Returns:
            List of (problem, score) tuples sorted by score descending.
        """
        scored = [(problem, self.score_problem(problem, **kwargs)) for problem in problems]
        scored.sort(key=lambda x: x[1], reverse=True)

        self._log.info(
            "problems_ranked",
            count=len(scored),
            top_score=scored[0][1] if scored else 0.0,
        )

        return scored

    def select_research_targets(
        self,
        problems: list[ResearchProblem],
        *,
        budget: int = 5,
        min_score: float = 0.3,
        **kwargs: float,
    ) -> list[tuple[ResearchProblem, float]]:
        """Select the top research targets within a budget.

        Only returns problems that score above ``min_score``.
        """
        ranked = self.rank_problems(problems, **kwargs)
        targets = [(problem, score) for problem, score in ranked if score >= min_score][:budget]

        self._log.info(
            "research_targets_selected",
            count=len(targets),
            budget=budget,
        )

        return targets

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_impact(problem: ResearchProblem) -> float:
        """Estimate potential impact from problem properties."""
        priority_scores = {
            ResearchProblemPriority.CRITICAL: 1.0,
            ResearchProblemPriority.HIGH: 0.8,
            ResearchProblemPriority.MEDIUM: 0.5,
            ResearchProblemPriority.LOW: 0.3,
            ResearchProblemPriority.DEFERRED: 0.1,
        }
        base = priority_scores.get(problem.priority, 0.5)

        # Decomposition quality bonus: more known/measurable = better
        decomp = problem.decomposition
        known_count = len(decomp.known_facts)
        measurable_count = len(decomp.measurable_quantities)
        decomposition_bonus = min((known_count + measurable_count) * 0.02, 0.15)

        return min(base + decomposition_bonus, 1.0)

    @staticmethod
    def _estimate_uncertainty(problem: ResearchProblem) -> float:
        """Estimate uncertainty from decomposition quality.

        High uncertainty = more unknowns = more research value.
        """
        decomp = problem.decomposition
        total_items = (
            len(decomp.known_facts)
            + len(decomp.inferences)
            + len(decomp.hypotheses)
            + len(decomp.unknowns)
        )
        if total_items == 0:
            return 0.5

        unknown_ratio = len(decomp.unknowns) / total_items
        inference_ratio = len(decomp.inferences) / total_items

        # High unknowns + many inferences = high uncertainty
        return min(unknown_ratio * 0.7 + inference_ratio * 0.3 + 0.2, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self._weights),
        }


__all__ = ["ResearchPrioritizer"]
