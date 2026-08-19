"""Intelligence evolution and meta-learning subsystem.

Phase 6 introduces controlled, measurable, evidence-driven evolution.
Kodiak evaluates and improves the mechanisms it uses for solving
problems — not just its answers, but its problem-solving machinery.

The central principle:

    No self-improvement without evaluation.
    No evaluation without evidence.
    No evidence without provenance.
"""

from kodiak.orchestration.evolution.capability import (
    Capability,
    CapabilityEvaluation,
    CapabilityPerformance,
    CapabilityTracker,
)
from kodiak.orchestration.evolution.capability_composer import (
    CapabilityComposer,
    CompositionResult,
)
from kodiak.orchestration.evolution.failure_patterns import (
    FailurePattern,
    FailurePatternMiner,
    FailurePatternSeverity,
)
from kodiak.orchestration.evolution.health import (
    HealthDimension,
    SystemHealth,
    SystemHealthDashboard,
)
from kodiak.orchestration.evolution.improvement_queue import (
    ImprovementProposal,
    ImprovementQueue,
    ImprovementStatus,
)
from kodiak.orchestration.evolution.memory_quality import (
    Contradiction,
    MemoryEntry,
    MemoryQualityController,
    QualityReport,
)
from kodiak.orchestration.evolution.meta_strategies import (
    MetaStrategyDecision,
    MetaStrategyProfile,
    MetaStrategySelector,
    RiskLevel,
    StrategySelectionMethod,
    TaskComplexity,
)
from kodiak.orchestration.evolution.models import (
    EvaluationDimension,
    EvaluationVerdict,
    SystemEvaluation,
    TaskEvaluation,
)
from kodiak.orchestration.evolution.research_bridge import (
    BridgeResult,
    ResearchDiscovery,
    ResearchEvolutionBridge,
)
from kodiak.orchestration.evolution.resource_aware import (
    ReasoningDepth,
    ResourceAwareEngine,
    ResourceProfile,
    TaskComplexityAssessment,
    VerificationLevel,
)
from kodiak.orchestration.evolution.self_evaluation import SelfEvaluationEngine

__all__ = [
    # Self-evaluation models
    "EvaluationDimension",
    "EvaluationVerdict",
    "SystemEvaluation",
    "TaskEvaluation",
    # Self-evaluation engine
    "SelfEvaluationEngine",
    # Capability model
    "Capability",
    "CapabilityEvaluation",
    "CapabilityTracker",
    "CapabilityPerformance",
    # Capability composition
    "CapabilityComposer",
    "CompositionResult",
    # Improvement queue
    "ImprovementProposal",
    "ImprovementQueue",
    "ImprovementStatus",
    # Failure pattern mining
    "FailurePattern",
    "FailurePatternMiner",
    "FailurePatternSeverity",
    # System health
    "HealthDimension",
    "SystemHealth",
    "SystemHealthDashboard",
    # Meta-strategies
    "MetaStrategySelector",
    "MetaStrategyProfile",
    "MetaStrategyDecision",
    "StrategySelectionMethod",
    "TaskComplexity",
    "RiskLevel",
    # Memory quality
    "MemoryEntry",
    "MemoryQualityController",
    "QualityReport",
    "Contradiction",
    # Research-evolution bridge
    "ResearchDiscovery",
    "ResearchEvolutionBridge",
    "BridgeResult",
    # Resource-aware intelligence
    "ResourceAwareEngine",
    "ResourceProfile",
    "TaskComplexityAssessment",
    "ReasoningDepth",
    "VerificationLevel",
]
