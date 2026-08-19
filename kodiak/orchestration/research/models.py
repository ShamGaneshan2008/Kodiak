"""Research object model — first-class entities for systematic strategy discovery.

Every research object has provenance.  No conclusions may exist without
supporting evidence.  No evidence may exist without an experiment or
observation source.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HypothesisStatus(enum.StrEnum):
    """Lifecycle state of a research hypothesis."""

    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    REFINED = "refined"


class EvidenceStrength(enum.StrEnum):
    """How strong the supporting evidence is."""

    NONE = "none"
    ANECDOTAL = "anecdotal"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONCLUSIVE = "conclusive"


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Result from running one strategy in an experiment.

    Contains the metrics, timing, and resource usage from a single
    strategy execution within a controlled experiment.
    """

    strategy_id: str = ""
    strategy_name: str = ""
    primary_metric: float = 0.0
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    task_success_rate: float = 0.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_duration_seconds: float = 0.0
    average_duration_seconds: float = 0.0
    tool_calls: int = 0
    resource_usage: dict[str, Any] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return (
            f"{self.strategy_name}: {self.successful_tasks}/{self.total_tasks} tasks, "
            f"primary_metric={self.primary_metric:.3f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "primary_metric": self.primary_metric,
            "secondary_metrics": dict(self.secondary_metrics),
            "task_success_rate": self.task_success_rate,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "total_duration_seconds": self.total_duration_seconds,
            "average_duration_seconds": self.average_duration_seconds,
            "tool_calls": self.tool_calls,
            "resource_usage": dict(self.resource_usage),
            "failures": list(self.failures),
            "metadata": dict(self.metadata),
        }


class KnowledgeClassification(enum.StrEnum):
    """How a piece of knowledge was established."""

    OBSERVED = "observed"  # Directly measured
    INFERRED = "inferred"  # Derived from evidence, not directly measured
    SUPPORTED = "supported"  # Backed by multiple experiments
    UNSUPPORTED = "unsupported"  # Contradicted by evidence
    UNKNOWN = "unknown"  # Not yet investigated


class ResearchProblemPriority(enum.StrEnum):
    """Priority levels for research problems."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"


# ---------------------------------------------------------------------------
# Problem Decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProblemDecomposition:
    """Structured breakdown of a research problem.

    Separates known facts from inferences, hypotheses, and unknowns.
    Never turn an inference into a fact simply because an LLM generated it.
    """

    known_facts: tuple[str, ...] = ()
    inferences: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    measurable_quantities: tuple[str, ...] = ()
    experimentally_changeable: tuple[str, ...] = ()
    improvement_criteria: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_facts": list(self.known_facts),
            "inferences": list(self.inferences),
            "hypotheses": list(self.hypotheses),
            "unknowns": list(self.unknowns),
            "assumptions": list(self.assumptions),
            "measurable_quantities": list(self.measurable_quantities),
            "experimentally_changeable": list(self.experimentally_changeable),
            "improvement_criteria": list(self.improvement_criteria),
        }


# ---------------------------------------------------------------------------
# ResearchProblem
# ---------------------------------------------------------------------------


@dataclass
class ResearchProblem:
    """A research question that Kodiak should investigate.

    Attributes:
        problem_id: Unique identifier.
        title: Short human-readable title.
        description: Detailed description of the problem.
        priority: Research priority.
        decomposition: Structured breakdown of what is known/unknown.
        related_strategy_ids: Strategies this problem relates to.
        tags: Searchable tags.
        created_at: When the problem was identified.
        resolved_at: When the problem was resolved, if ever.
        metadata: Arbitrary additional data.
    """

    problem_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    description: str = ""
    priority: ResearchProblemPriority = ResearchProblemPriority.MEDIUM
    decomposition: ProblemDecomposition = field(default_factory=ProblemDecomposition)
    related_strategy_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "decomposition": self.decomposition.to_dict(),
            "related_strategy_ids": list(self.related_strategy_ids),
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "is_resolved": self.is_resolved,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# KnowledgeGap
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KnowledgeGap:
    """Identified gap in existing knowledge.

    When existing knowledge is insufficient to make a decision,
    it becomes a research opportunity.
    """

    gap_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""
    known_context: str = ""
    unknown_quantity: str = ""
    potential_impact: str = ""
    related_problem_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "description": self.description,
            "known_context": self.known_context,
            "unknown_quantity": self.unknown_quantity,
            "potential_impact": self.potential_impact,
            "related_problem_ids": list(self.related_problem_ids),
        }


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A testable prediction about strategy behavior.

    Hypotheses must be:
    - Falsifiable (there must be a way to disprove them)
    - Specific (not vague claims)
    - Connected to evidence (through experiments)

    Attributes:
        hypothesis_id: Unique identifier.
        statement: The hypothesis statement.
        rationale: Why this hypothesis is plausible.
        status: Current lifecycle state.
        related_problem_id: The research problem this addresses.
        related_strategy_ids: Strategies involved.
        expected_benefit: What improvement is expected.
        expected_cost: Resource cost of testing.
        risk: Risk if the hypothesis is wrong.
        testability: How easily this can be tested.
        confidence: Current confidence in the hypothesis (0.0-1.0).
        experiment_ids: Experiments testing this hypothesis.
        created_at: When proposed.
        updated_at: Last modification.
        metadata: Arbitrary additional data.
    """

    hypothesis_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    statement: str = ""
    rationale: str = ""
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    related_problem_id: str = ""
    related_strategy_ids: tuple[str, ...] = ()
    expected_benefit: str = ""
    expected_cost: float = 0.5
    risk: float = 0.5
    testability: float = 0.5
    confidence: float = 0.5
    experiment_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "status": self.status.value,
            "related_problem_id": self.related_problem_id,
            "related_strategy_ids": list(self.related_strategy_ids),
            "expected_benefit": self.expected_benefit,
            "expected_cost": self.expected_cost,
            "risk": self.risk,
            "testability": self.testability,
            "confidence": self.confidence,
            "experiment_ids": list(self.experiment_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """A piece of evidence supporting or contradicting a hypothesis.

    Every piece of evidence must have provenance — an experiment_id or
    observation_id that produced it.

    Attributes:
        evidence_id: Unique identifier.
        hypothesis_id: The hypothesis this evidence relates to.
        experiment_id: The experiment that produced this evidence.
        observation_id: Alternative: an observation that produced this.
        strength: Strength classification.
        summary: Human-readable summary.
        measurements: Raw measurement data.
        supports_hypothesis: Whether this evidence supports the hypothesis.
        confidence: Confidence in this evidence (0.0-1.0).
        source: Where this evidence came from.
        created_at: When collected.
    """

    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    hypothesis_id: str = ""
    experiment_id: str = ""
    observation_id: str = ""
    strength: EvidenceStrength = EvidenceStrength.NONE
    summary: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)
    supports_hypothesis: bool | None = None
    confidence: float = 0.5
    source: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_provenance(self) -> bool:
        """Evidence without provenance is not valid."""
        return bool(self.experiment_id or self.observation_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "hypothesis_id": self.hypothesis_id,
            "experiment_id": self.experiment_id,
            "observation_id": self.observation_id,
            "strength": self.strength.value,
            "summary": self.summary,
            "measurements": dict(self.measurements),
            "supports_hypothesis": self.supports_hypothesis,
            "confidence": self.confidence,
            "source": self.source,
            "has_provenance": self.has_provenance,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """A recorded observation from execution or analysis.

    Observations are raw data — they don't require an experiment.
    They can seed hypotheses.
    """

    observation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = ""
    summary: str = ""
    source: str = ""
    category: str = ""  # e.g. "unexpected_success", "performance_bottleneck", "pattern"
    measurements: dict[str, Any] = field(default_factory=dict)
    related_problem_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "category": self.category,
            "measurements": dict(self.measurements),
            "related_problem_ids": list(self.related_problem_ids),
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------


@dataclass
class Conclusion:
    """A conclusion drawn from evidence.

    Conclusions MUST have supporting evidence.  A conclusion without
    evidence is not a conclusion — it is an opinion.

    Attributes:
        conclusion_id: Unique identifier.
        hypothesis_id: The hypothesis this concludes on.
        statement: The conclusion.
        classification: How the conclusion was established.
        supporting_evidence_ids: Evidence supporting this conclusion.
        contradicting_evidence_ids: Evidence contradicting this.
        confidence: Overall confidence (0.0-1.0).
        limitations: Known limitations of this conclusion.
        source_experiments: Experiments that contributed.
        created_at: When drawn.
    """

    conclusion_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    hypothesis_id: str = ""
    statement: str = ""
    classification: KnowledgeClassification = KnowledgeClassification.UNKNOWN
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.5
    limitations: tuple[str, ...] = ()
    source_experiments: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_evidence(self) -> bool:
        """A conclusion without evidence is invalid."""
        return bool(self.supporting_evidence_ids)

    @property
    def net_evidence_strength(self) -> int:
        """Positive if supporting > contradicting, negative if opposite."""
        return len(self.supporting_evidence_ids) - len(self.contradicting_evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "classification": self.classification.value,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "confidence": self.confidence,
            "limitations": list(self.limitations),
            "source_experiments": list(self.source_experiments),
            "has_evidence": self.has_evidence,
            "net_evidence_strength": self.net_evidence_strength,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Lesson
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lesson:
    """A reusable lesson learned from research.

    Lessons generalize across specific experiments and conclusions.
    They capture transferable knowledge.
    """

    lesson_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    statement: str = ""
    domain: str = ""  # e.g. "test_fixing", "dependency_resolution"
    scope: str = ""  # e.g. "small_repos", "high_coupling"
    supporting_conclusion_ids: tuple[str, ...] = ()
    confidence: float = 0.5
    limitations: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "statement": self.statement,
            "domain": self.domain,
            "scope": self.scope,
            "supporting_conclusion_ids": list(self.supporting_conclusion_ids),
            "confidence": self.confidence,
            "limitations": list(self.limitations),
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# StrategyVersion
# ---------------------------------------------------------------------------


@dataclass
class StrategyVersion:
    """A versioned snapshot of a strategy.

    Tracks how a strategy evolves through modifications, experiments,
    and benchmark results.
    """

    version_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    strategy_id: str = ""
    version_number: int = 1
    name: str = ""
    approach: str = ""
    parent_version_id: str | None = None
    change_description: str = ""
    experiment_ids: tuple[str, ...] = ()
    benchmark_task_ids: tuple[str, ...] = ()
    success_rate: float = 0.5
    effectiveness_score: float = 0.5
    use_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "strategy_id": self.strategy_id,
            "version_number": self.version_number,
            "name": self.name,
            "approach": self.approach,
            "parent_version_id": self.parent_version_id,
            "change_description": self.change_description,
            "experiment_ids": list(self.experiment_ids),
            "benchmark_task_ids": list(self.benchmark_task_ids),
            "success_rate": self.success_rate,
            "effectiveness_score": self.effectiveness_score,
            "use_count": self.use_count,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# NegativeKnowledge
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NegativeKnowledge:
    """A recorded failed approach.

    Negative evidence is valuable because it reduces repeated
    experimentation with known-bad strategies.
    """

    knowledge_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    strategy_description: str = ""
    problem_class: str = ""
    result: str = ""
    conclusion: str = ""
    conditions: tuple[str, ...] = ()
    confidence: float = 0.5
    experiment_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "strategy_description": self.strategy_description,
            "problem_class": self.problem_class,
            "result": self.result,
            "conclusion": self.conclusion,
            "conditions": list(self.conditions),
            "confidence": self.confidence,
            "experiment_ids": list(self.experiment_ids),
            "created_at": self.created_at.isoformat(),
        }


__all__ = [
    "Conclusion",
    "Evidence",
    "EvidenceStrength",
    "ExperimentResult",
    "Hypothesis",
    "HypothesisStatus",
    "KnowledgeClassification",
    "KnowledgeGap",
    "Lesson",
    "NegativeKnowledge",
    "Observation",
    "ProblemDecomposition",
    "ResearchProblem",
    "ResearchProblemPriority",
    "StrategyVersion",
]
