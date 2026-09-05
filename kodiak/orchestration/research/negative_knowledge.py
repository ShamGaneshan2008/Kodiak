"""Negative knowledge store.

Stores failed approaches and their contexts to prevent repeated
experimentation with known-bad strategies.  Negative evidence is
valuable because it reduces wasted effort.

Example:
    Strategy: Mass dependency upgrade
    Result: Increased compatibility failures.
    Conclusion: Poor strategy for localized dependency conflicts.
"""

from __future__ import annotations

import structlog

from kodiak.orchestration.research.models import NegativeKnowledge

logger = structlog.get_logger(__name__)


class NegativeKnowledgeStore:
    """Stores and retrieves negative knowledge (failed approaches).

    Helps Kodiak avoid repeating experiments that are known to fail
    under specific conditions.
    """

    def __init__(self, max_entries: int = 200) -> None:
        self._entries: dict[str, NegativeKnowledge] = {}
        self._max_entries = max_entries
        self._log = logger.bind(component="negative_knowledge_store")

    def store(self, knowledge: NegativeKnowledge) -> None:
        """Store a negative knowledge entry."""
        if len(self._entries) >= self._max_entries:
            self._evict()
        self._entries[knowledge.knowledge_id] = knowledge
        self._log.info(
            "negative_knowledge_stored",
            knowledge_id=knowledge.knowledge_id,
            problem_class=knowledge.problem_class,
            strategy=knowledge.strategy_description[:60],
        )

    def get(self, knowledge_id: str) -> NegativeKnowledge | None:
        return self._entries.get(knowledge_id)

    def retrieve(
        self,
        *,
        problem_class: str | None = None,
        strategy_description: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> list[NegativeKnowledge]:
        """Retrieve negative knowledge matching criteria.

        Useful for checking whether a proposed approach has already
        been tried and rejected.
        """
        candidates = list(self._entries.values())

        if problem_class:
            candidates = [n for n in candidates if n.problem_class == problem_class]

        if strategy_description:
            # Fuzzy match: check if the description appears in any entry
            search = strategy_description.lower()
            candidates = [
                n
                for n in candidates
                if search in n.strategy_description.lower() or search in n.conclusion.lower()
            ]

        candidates = [n for n in candidates if n.confidence >= min_confidence]
        return candidates[:limit]

    def check_approach(
        self,
        strategy_description: str,
        problem_class: str = "",
    ) -> NegativeKnowledge | None:
        """Check if a proposed approach has been tried and rejected.

        Returns the most relevant negative knowledge entry, or None
        if no matching rejection exists.
        """
        matches = self.retrieve(
            problem_class=problem_class or None,
            strategy_description=strategy_description,
        )
        if not matches:
            return None

        # Return highest confidence match
        matches.sort(key=lambda n: n.confidence, reverse=True)
        best = matches[0]

        self._log.info(
            "approach_already_rejected",
            strategy=strategy_description[:60],
            knowledge_id=best.knowledge_id,
            confidence=best.confidence,
        )

        return best

    def record_failure(
        self,
        *,
        strategy_description: str,
        problem_class: str,
        result: str,
        conclusion: str,
        conditions: tuple[str, ...] = (),
        confidence: float = 0.7,
        experiment_ids: tuple[str, ...] = (),
    ) -> NegativeKnowledge:
        """Convenience method to record a failed approach."""
        knowledge = NegativeKnowledge(
            strategy_description=strategy_description,
            problem_class=problem_class,
            result=result,
            conclusion=conclusion,
            conditions=conditions,
            confidence=confidence,
            experiment_ids=experiment_ids,
        )
        self.store(knowledge)
        return knowledge

    def all_entries(self) -> list[NegativeKnowledge]:
        return sorted(
            self._entries.values(),
            key=lambda n: n.confidence,
            reverse=True,
        )

    def stats(self) -> dict[str, int]:
        entries = list(self._entries.values())
        classes = {}
        for entry in entries:
            classes[entry.problem_class] = classes.get(entry.problem_class, 0) + 1
        return {
            "total": len(entries),
            "by_problem_class": classes,
        }

    def _evict(self) -> None:
        """Remove the lowest-confidence entry."""
        if not self._entries:
            return
        worst = min(self._entries.values(), key=lambda n: n.confidence)
        del self._entries[worst.knowledge_id]
        self._log.info("negative_knowledge_evicted", knowledge_id=worst.knowledge_id)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, knowledge_id: str) -> bool:
        return knowledge_id in self._entries


__all__ = ["NegativeKnowledgeStore"]
