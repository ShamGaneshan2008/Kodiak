"""Research memory — stores and retrieves research knowledge with provenance.

Research memory is separate from ordinary execution memories.  It allows
Kodiak to answer:

    "What have we actually tested about this problem?"

rather than just:

    "What have we previously done?"
"""

from __future__ import annotations

from typing import Any

import structlog

from kodiak.orchestration.research.models import (
    Conclusion,
    Evidence,
    EvidenceStrength,
    Hypothesis,
    HypothesisStatus,
    KnowledgeGap,
    Lesson,
    NegativeKnowledge,
    Observation,
    ResearchProblem,
    ResearchProblemPriority,
    StrategyVersion,
)

logger = structlog.get_logger(__name__)


class ResearchMemory:
    """Stores and retrieves research knowledge.

    Maintains separate collections for each entity type while supporting
    cross-references between them.  All mutations are tracked for
    provenance.
    """

    def __init__(self, max_problems: int = 200, max_hypotheses: int = 500) -> None:
        self._problems: dict[str, ResearchProblem] = {}
        self._knowledge_gaps: dict[str, KnowledgeGap] = {}
        self._hypotheses: dict[str, Hypothesis] = {}
        self._evidence: dict[str, Evidence] = {}
        self._observations: dict[str, Observation] = {}
        self._conclusions: dict[str, Conclusion] = {}
        self._lessons: dict[str, Lesson] = {}
        self._strategy_versions: dict[str, StrategyVersion] = {}
        self._negative_knowledge: dict[str, NegativeKnowledge] = {}
        self._max_problems = max_problems
        self._max_hypotheses = max_hypotheses
        self._log = logger.bind(component="research_memory")

    # ------------------------------------------------------------------
    # Problem operations
    # ------------------------------------------------------------------

    def store_problem(self, problem: ResearchProblem) -> None:
        """Store a research problem."""
        self._problems[problem.problem_id] = problem
        self._log.info(
            "problem_stored",
            problem_id=problem.problem_id,
            title=problem.title,
            priority=problem.priority.value,
        )

    def get_problem(self, problem_id: str) -> ResearchProblem | None:
        return self._problems.get(problem_id)

    def retrieve_problems(
        self,
        *,
        priority: ResearchProblemPriority | None = None,
        unresolved_only: bool = True,
        tags: tuple[str, ...] | None = None,
        limit: int = 10,
    ) -> list[ResearchProblem]:
        """Retrieve research problems, sorted by priority."""
        candidates = list(self._problems.values())

        if unresolved_only:
            candidates = [p for p in candidates if not p.is_resolved]

        if priority is not None:
            candidates = [p for p in candidates if p.priority == priority]

        if tags:
            tag_set = set(tags)
            candidates = [p for p in candidates if tag_set & set(p.tags)]

        priority_order = {
            ResearchProblemPriority.CRITICAL: 0,
            ResearchProblemPriority.HIGH: 1,
            ResearchProblemPriority.MEDIUM: 2,
            ResearchProblemPriority.LOW: 3,
            ResearchProblemPriority.DEFERRED: 4,
        }
        candidates.sort(key=lambda p: priority_order.get(p.priority, 5))
        return candidates[:limit]

    def resolve_problem(self, problem_id: str) -> bool:
        """Mark a problem as resolved."""
        problem = self._problems.get(problem_id)
        if problem is None:
            return False
        from datetime import UTC, datetime

        problem.resolved_at = datetime.now(UTC)
        self._log.info("problem_resolved", problem_id=problem_id)
        return True

    # ------------------------------------------------------------------
    # Knowledge gap operations
    # ------------------------------------------------------------------

    def store_knowledge_gap(self, gap: KnowledgeGap) -> None:
        self._knowledge_gaps[gap.gap_id] = gap
        self._log.info("knowledge_gap_stored", gap_id=gap.gap_id)

    def get_knowledge_gap(self, gap_id: str) -> KnowledgeGap | None:
        return self._knowledge_gaps.get(gap_id)

    def retrieve_knowledge_gaps(
        self,
        *,
        related_problem_id: str | None = None,
        limit: int = 10,
    ) -> list[KnowledgeGap]:
        candidates = list(self._knowledge_gaps.values())
        if related_problem_id:
            candidates = [g for g in candidates if related_problem_id in g.related_problem_ids]
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Hypothesis operations
    # ------------------------------------------------------------------

    def store_hypothesis(self, hypothesis: Hypothesis) -> None:
        if len(self._hypotheses) >= self._max_hypotheses:
            self._evict_hypotheses()
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._log.info(
            "hypothesis_stored",
            hypothesis_id=hypothesis.hypothesis_id,
            status=hypothesis.status.value,
        )

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return self._hypotheses.get(hypothesis_id)

    def retrieve_hypotheses(
        self,
        *,
        status: HypothesisStatus | None = None,
        problem_id: str | None = None,
        strategy_ids: tuple[str, ...] | None = None,
        limit: int = 20,
    ) -> list[Hypothesis]:
        candidates = list(self._hypotheses.values())

        if status is not None:
            candidates = [h for h in candidates if h.status == status]

        if problem_id:
            candidates = [h for h in candidates if h.related_problem_id == problem_id]

        if strategy_ids:
            strategy_set = set(strategy_ids)
            candidates = [h for h in candidates if strategy_set & set(h.related_strategy_ids)]

        # Sort by confidence (highest first)
        candidates.sort(key=lambda h: h.confidence, reverse=True)
        return candidates[:limit]

    def update_hypothesis_status(
        self, hypothesis_id: str, status: HypothesisStatus
    ) -> Hypothesis | None:
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return None
        hypothesis.status = status
        from datetime import UTC, datetime

        hypothesis.updated_at = datetime.now(UTC)
        self._log.info(
            "hypothesis_status_updated",
            hypothesis_id=hypothesis_id,
            new_status=status.value,
        )
        return hypothesis

    # ------------------------------------------------------------------
    # Evidence operations
    # ------------------------------------------------------------------

    def store_evidence(self, evidence: Evidence) -> None:
        if not evidence.has_provenance:
            self._log.warning(
                "evidence_without_provenance",
                evidence_id=evidence.evidence_id,
            )
        self._evidence[evidence.evidence_id] = evidence
        self._log.info(
            "evidence_stored",
            evidence_id=evidence.evidence_id,
            strength=evidence.strength.value,
            supports=evidence.supports_hypothesis,
        )

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return self._evidence.get(evidence_id)

    def retrieve_evidence_for_hypothesis(
        self,
        hypothesis_id: str,
        *,
        limit: int = 50,
    ) -> list[Evidence]:
        """Retrieve all evidence related to a hypothesis, sorted by strength."""
        strength_order = {
            EvidenceStrength.NONE: 0,
            EvidenceStrength.ANECDOTAL: 1,
            EvidenceStrength.WEAK: 2,
            EvidenceStrength.MODERATE: 3,
            EvidenceStrength.STRONG: 4,
            EvidenceStrength.CONCLUSIVE: 5,
        }
        candidates = [e for e in self._evidence.values() if e.hypothesis_id == hypothesis_id]
        candidates.sort(
            key=lambda e: (strength_order.get(e.strength, 0), e.confidence),
            reverse=True,
        )
        return candidates[:limit]

    def retrieve_evidence_for_experiment(self, experiment_id: str) -> list[Evidence]:
        return [e for e in self._evidence.values() if e.experiment_id == experiment_id]

    # ------------------------------------------------------------------
    # Observation operations
    # ------------------------------------------------------------------

    def store_observation(self, observation: Observation) -> None:
        self._observations[observation.observation_id] = observation
        self._log.info(
            "observation_stored",
            observation_id=observation.observation_id,
            category=observation.category,
        )

    def get_observation(self, observation_id: str) -> Observation | None:
        return self._observations.get(observation_id)

    def retrieve_observations(
        self,
        *,
        category: str | None = None,
        limit: int = 20,
    ) -> list[Observation]:
        candidates = list(self._observations.values())
        if category:
            candidates = [o for o in candidates if o.category == category]
        candidates.sort(key=lambda o: o.created_at, reverse=True)
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Conclusion operations
    # ------------------------------------------------------------------

    def store_conclusion(self, conclusion: Conclusion) -> None:
        if not conclusion.has_evidence:
            self._log.warning(
                "conclusion_without_evidence",
                conclusion_id=conclusion.conclusion_id,
            )
        self._conclusions[conclusion.conclusion_id] = conclusion
        self._log.info(
            "conclusion_stored",
            conclusion_id=conclusion.conclusion_id,
            classification=conclusion.classification.value,
            has_evidence=conclusion.has_evidence,
        )

    def get_conclusion(self, conclusion_id: str) -> Conclusion | None:
        return self._conclusions.get(conclusion_id)

    def retrieve_conclusions_for_hypothesis(self, hypothesis_id: str) -> list[Conclusion]:
        return [c for c in self._conclusions.values() if c.hypothesis_id == hypothesis_id]

    # ------------------------------------------------------------------
    # Lesson operations
    # ------------------------------------------------------------------

    def store_lesson(self, lesson: Lesson) -> None:
        self._lessons[lesson.lesson_id] = lesson
        self._log.info(
            "lesson_stored",
            lesson_id=lesson.lesson_id,
            domain=lesson.domain,
        )

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        return self._lessons.get(lesson_id)

    def retrieve_lessons(
        self,
        *,
        domain: str | None = None,
        limit: int = 20,
    ) -> list[Lesson]:
        candidates = list(self._lessons.values())
        if domain:
            candidates = [lesson for lesson in candidates if lesson.domain == domain]
        candidates.sort(key=lambda lesson: lesson.confidence, reverse=True)
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Strategy version operations
    # ------------------------------------------------------------------

    def store_strategy_version(self, version: StrategyVersion) -> None:
        self._strategy_versions[version.version_id] = version
        self._log.info(
            "strategy_version_stored",
            version_id=version.version_id,
            strategy_id=version.strategy_id,
            version_number=version.version_number,
        )

    def get_strategy_version(self, version_id: str) -> StrategyVersion | None:
        return self._strategy_versions.get(version_id)

    def retrieve_strategy_versions(self, strategy_id: str) -> list[StrategyVersion]:
        """Retrieve all versions for a strategy, sorted by version number."""
        versions = [v for v in self._strategy_versions.values() if v.strategy_id == strategy_id]
        versions.sort(key=lambda v: v.version_number)
        return versions

    def get_latest_strategy_version(self, strategy_id: str) -> StrategyVersion | None:
        versions = self.retrieve_strategy_versions(strategy_id)
        return versions[-1] if versions else None

    # ------------------------------------------------------------------
    # Negative knowledge operations
    # ------------------------------------------------------------------

    def store_negative_knowledge(self, knowledge: NegativeKnowledge) -> None:
        self._negative_knowledge[knowledge.knowledge_id] = knowledge
        self._log.info(
            "negative_knowledge_stored",
            knowledge_id=knowledge.knowledge_id,
            problem_class=knowledge.problem_class,
        )

    def get_negative_knowledge(self, knowledge_id: str) -> NegativeKnowledge | None:
        return self._negative_knowledge.get(knowledge_id)

    def retrieve_negative_knowledge(
        self,
        *,
        problem_class: str | None = None,
        limit: int = 20,
    ) -> list[NegativeKnowledge]:
        candidates = list(self._negative_knowledge.values())
        if problem_class:
            candidates = [n for n in candidates if n.problem_class == problem_class]
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def research_summary_for_problem(self, problem_id: str) -> dict[str, Any]:
        """Return a complete research summary for a problem.

        This answers: "What have we actually tested about this problem?"
        """
        problem = self.get_problem(problem_id)
        if problem is None:
            return {"error": "problem_not_found"}

        hypotheses = self.retrieve_hypotheses(problem_id=problem_id)
        all_evidence: list[Evidence] = []
        for h in hypotheses:
            all_evidence.extend(self.retrieve_evidence_for_hypothesis(h.hypothesis_id))

        conclusions: list[Conclusion] = []
        for h in hypotheses:
            conclusions.extend(self.retrieve_conclusions_for_hypothesis(h.hypothesis_id))

        strategy_ids = set()
        for h in hypotheses:
            strategy_ids.update(h.related_strategy_ids)

        negative = self.retrieve_negative_knowledge(problem_class=problem.problem_id)

        return {
            "problem": problem.to_dict(),
            "hypotheses": [h.to_dict() for h in hypotheses],
            "evidence": [e.to_dict() for e in all_evidence],
            "conclusions": [c.to_dict() for c in conclusions],
            "strategy_versions": {
                sid: [v.to_dict() for v in self.retrieve_strategy_versions(sid)]
                for sid in strategy_ids
            },
            "negative_knowledge": [n.to_dict() for n in negative],
        }

    def stats(self) -> dict[str, int]:
        """Return counts of all stored entities."""
        return {
            "problems": len(self._problems),
            "knowledge_gaps": len(self._knowledge_gaps),
            "hypotheses": len(self._hypotheses),
            "evidence": len(self._evidence),
            "observations": len(self._observations),
            "conclusions": len(self._conclusions),
            "lessons": len(self._lessons),
            "strategy_versions": len(self._strategy_versions),
            "negative_knowledge": len(self._negative_knowledge),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_hypotheses(self) -> None:
        """Remove rejected/inconclusive hypotheses to make room."""
        evictable = [
            h
            for h in self._hypotheses.values()
            if h.status in {HypothesisStatus.REJECTED, HypothesisStatus.INCONCLUSIVE}
        ]
        if evictable:
            worst = min(evictable, key=lambda h: h.confidence)
            del self._hypotheses[worst.hypothesis_id]
            self._log.info("hypothesis_evicted", hypothesis_id=worst.hypothesis_id)
            return

        # If nothing evictable, remove lowest-confidence proposed
        proposed = [h for h in self._hypotheses.values() if h.status == HypothesisStatus.PROPOSED]
        if proposed:
            lowest = min(proposed, key=lambda h: h.confidence)
            del self._hypotheses[lowest.hypothesis_id]
            self._log.info("hypothesis_evicted", hypothesis_id=lowest.hypothesis_id)


__all__ = ["ResearchMemory"]
