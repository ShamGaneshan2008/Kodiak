"""Research, experimentation, and strategy discovery subsystem.

Phase 5 introduces systematic discovery, testing, evaluation, and
retention of new approaches to engineering problems.  The central
principle is:

    Kodiak must distinguish between an idea and a demonstrated improvement.

A generated idea is only a candidate.  A discovered improvement requires:

    candidate + experiment + evidence + comparison + verification

Only then may Kodiak treat it as a learned strategy.
"""

from kodiak.orchestration.research.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkTaskCategory,
)
from kodiak.orchestration.research.composer import StrategyComposer
from kodiak.orchestration.research.experiment import (
    Experiment,
    ExperimentDesignEngine,
    ExperimentPhase,
    ExperimentResult,
)
from kodiak.orchestration.research.memory import ResearchMemory
from kodiak.orchestration.research.models import (
    Conclusion,
    Evidence,
    EvidenceStrength,
    ExperimentResult,
    Hypothesis,
    HypothesisStatus,
    KnowledgeClassification,
    KnowledgeGap,
    Lesson,
    NegativeKnowledge,
    Observation,
    ProblemDecomposition,
    ResearchProblem,
    ResearchProblemPriority,
    StrategyVersion,
)
from kodiak.orchestration.research.negative_knowledge import NegativeKnowledgeStore
from kodiak.orchestration.research.prioritizer import ResearchPrioritizer

__all__ = [
    # Models
    "Conclusion",
    "Evidence",
    "EvidenceStrength",
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
    # Research Memory
    "ResearchMemory",
    # Experiments
    "Experiment",
    "ExperimentDesignEngine",
    "ExperimentPhase",
    "ExperimentResult",
    # Benchmarks
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "BenchmarkTaskCategory",
    # Composition
    "StrategyComposer",
    # Negative Knowledge
    "NegativeKnowledgeStore",
    # Prioritization
    "ResearchPrioritizer",
]
