"""Resource-aware intelligence — adaptive reasoning depth.

Kodiak should choose reasoning depth according to task complexity:

    simple task → cheap reasoning
    moderate task → normal planning
    difficult task → deeper analysis
    high uncertainty → experimentation
    high-risk task → stronger verification

Do not spend expensive reasoning on trivial work.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ReasoningDepth(enum.StrEnum):
    """Levels of reasoning depth."""

    MINIMAL = "minimal"
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class VerificationLevel(enum.StrEnum):
    """Levels of verification rigor."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    THOROUGH = "thorough"
    EXHAUSTIVE = "exhaustive"


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Resource allocation profile for a task.

    Specifies how much computational effort to invest in each phase.
    """

    profile_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    reasoning_depth: ReasoningDepth = ReasoningDepth.STANDARD
    verification_level: VerificationLevel = VerificationLevel.STANDARD
    max_planning_time_seconds: float = 30.0
    max_retries: int = 3
    max_replans: int = 2
    enable_experimentation: bool = False
    enable_research: bool = False
    memory_recall_limit: int = 5
    parallel_verification: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "reasoning_depth": self.reasoning_depth.value,
            "verification_level": self.verification_level.value,
            "max_planning_time_seconds": self.max_planning_time_seconds,
            "max_retries": self.max_retries,
            "max_replans": self.max_replans,
            "enable_experimentation": self.enable_experimentation,
            "enable_research": self.enable_research,
            "memory_recall_limit": self.memory_recall_limit,
            "parallel_verification": self.parallel_verification,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TaskComplexityAssessment:
    """Assessment of task complexity for resource allocation."""

    assessment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    goal: str = ""
    complexity_score: float = 0.5  # 0.0 = trivial, 1.0 = extremely complex
    uncertainty_score: float = 0.5
    risk_score: float = 0.5
    reasoning_depth: ReasoningDepth = ReasoningDepth.STANDARD
    verification_level: VerificationLevel = VerificationLevel.STANDARD
    factors: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "task_id": self.task_id,
            "goal": self.goal,
            "complexity_score": round(self.complexity_score, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "risk_score": round(self.risk_score, 4),
            "reasoning_depth": self.reasoning_depth.value,
            "verification_level": self.verification_level.value,
            "factors": list(self.factors),
            "created_at": self.created_at.isoformat(),
        }


# Default profiles for each reasoning depth
_DEFAULT_PROFILES: dict[ReasoningDepth, ResourceProfile] = {
    ReasoningDepth.MINIMAL: ResourceProfile(
        name="minimal",
        reasoning_depth=ReasoningDepth.MINIMAL,
        verification_level=VerificationLevel.NONE,
        max_planning_time_seconds=5.0,
        max_retries=0,
        max_replans=0,
        memory_recall_limit=1,
    ),
    ReasoningDepth.LIGHT: ResourceProfile(
        name="light",
        reasoning_depth=ReasoningDepth.LIGHT,
        verification_level=VerificationLevel.BASIC,
        max_planning_time_seconds=15.0,
        max_retries=1,
        max_replans=0,
        memory_recall_limit=3,
    ),
    ReasoningDepth.STANDARD: ResourceProfile(
        name="standard",
        reasoning_depth=ReasoningDepth.STANDARD,
        verification_level=VerificationLevel.STANDARD,
        max_planning_time_seconds=30.0,
        max_retries=3,
        max_replans=2,
        memory_recall_limit=5,
    ),
    ReasoningDepth.DEEP: ResourceProfile(
        name="deep",
        reasoning_depth=ReasoningDepth.DEEP,
        verification_level=VerificationLevel.THOROUGH,
        max_planning_time_seconds=60.0,
        max_retries=5,
        max_replans=3,
        enable_experimentation=True,
        memory_recall_limit=10,
    ),
    ReasoningDepth.EXHAUSTIVE: ResourceProfile(
        name="exhaustive",
        reasoning_depth=ReasoningDepth.EXHAUSTIVE,
        verification_level=VerificationLevel.EXHAUSTIVE,
        max_planning_time_seconds=120.0,
        max_retries=7,
        max_replans=5,
        enable_experimentation=True,
        enable_research=True,
        memory_recall_limit=15,
        parallel_verification=True,
    ),
}


class ResourceAwareEngine:
    """Adapts reasoning depth and resource allocation to task complexity.

    Assesses task complexity and selects an appropriate resource profile.
    Tracks resource usage to improve future assessments.
    """

    def __init__(self) -> None:
        self._profiles: dict[ReasoningDepth, ResourceProfile] = dict(_DEFAULT_PROFILES)
        self._assessments: list[TaskComplexityAssessment] = []
        self._usage_history: list[dict[str, Any]] = []
        self._log = logger.bind(component="resource_aware_engine")

    def assess_task(
        self,
        *,
        task_id: str,
        goal: str,
        has_known_strategies: bool = False,
        strategy_confidence: float = 0.5,
        previous_failures: int = 0,
        requires_research: bool = False,
        is_high_risk: bool = False,
    ) -> TaskComplexityAssessment:
        """Assess task complexity and determine resource allocation."""
        factors: list[str] = []
        complexity = 0.3  # Base complexity
        uncertainty = 0.3
        risk = 0.2

        # Goal length and specificity
        if len(goal) > 200:
            complexity += 0.1
            factors.append("Long, detailed goal")
        if len(goal) < 20:
            complexity -= 0.1
            factors.append("Short, simple goal")

        # Known strategies
        if has_known_strategies:
            complexity -= 0.1
            uncertainty -= 0.1
            factors.append("Known strategies available")
        else:
            complexity += 0.15
            uncertainty += 0.2
            factors.append("No known strategies")
            factors.append("High uncertainty")

        # Strategy confidence
        if strategy_confidence > 0.7:
            uncertainty -= 0.1
        elif strategy_confidence < 0.3:
            uncertainty += 0.15
            factors.append("Low strategy confidence")

        # Previous failures
        if previous_failures > 0:
            complexity += previous_failures * 0.05
            risk += previous_failures * 0.05
            factors.append(f"{previous_failures} previous failure(s)")

        # Research needed
        if requires_research:
            complexity += 0.1
            uncertainty += 0.1
            factors.append("Research required")

        # High risk
        if is_high_risk:
            risk += 0.3
            factors.append("High-risk task")

        # Clamp scores
        complexity = max(0.0, min(1.0, complexity))
        uncertainty = max(0.0, min(1.0, uncertainty))
        risk = max(0.0, min(1.0, risk))

        # Determine reasoning depth
        combined = (complexity + uncertainty + risk) / 3.0
        depth = self._depth_for_score(combined)
        verification = self._verification_for_depth(depth, risk)

        assessment = TaskComplexityAssessment(
            task_id=task_id,
            goal=goal,
            complexity_score=complexity,
            uncertainty_score=uncertainty,
            risk_score=risk,
            reasoning_depth=depth,
            verification_level=verification,
            factors=tuple(factors),
        )

        self._assessments.append(assessment)
        self._log.info(
            "task_assessed",
            task_id=task_id,
            complexity=round(complexity, 3),
            depth=depth.value,
            factors=len(factors),
        )

        return assessment

    def get_profile(self, depth: ReasoningDepth) -> ResourceProfile:
        """Get the resource profile for a reasoning depth."""
        return self._profiles.get(depth, _DEFAULT_PROFILES[ReasoningDepth.STANDARD])

    def profile_for_assessment(
        self, assessment: TaskComplexityAssessment
    ) -> ResourceProfile:
        """Get the resource profile for an assessment."""
        return self.get_profile(assessment.reasoning_depth)

    def register_profile(self, profile: ResourceProfile) -> None:
        """Register a custom resource profile."""
        self._profiles[profile.reasoning_depth] = profile
        self._log.info(
            "profile_registered",
            depth=profile.reasoning_depth.value,
            name=profile.name,
        )

    def record_usage(
        self,
        *,
        task_id: str,
        depth: ReasoningDepth,
        duration_seconds: float,
        success: bool,
        retries_used: int = 0,
        replans_used: int = 0,
    ) -> None:
        """Record resource usage for a completed task."""
        usage = {
            "task_id": task_id,
            "depth": depth.value,
            "duration_seconds": duration_seconds,
            "success": success,
            "retries_used": retries_used,
            "replans_used": replans_used,
        }
        self._usage_history.append(usage)
        # Keep last 200
        if len(self._usage_history) > 200:
            self._usage_history = self._usage_history[-200:]

    def usage_report(self) -> dict[str, Any]:
        """Report on resource usage patterns."""
        if not self._usage_history:
            return {"total": 0}

        by_depth: dict[str, list[dict[str, Any]]] = {}
        for u in self._usage_history:
            by_depth.setdefault(u["depth"], []).append(u)

        report: dict[str, Any] = {"total": len(self._usage_history), "by_depth": {}}
        for depth, usages in by_depth.items():
            durations = [u["duration_seconds"] for u in usages]
            successes = sum(1 for u in usages if u["success"])
            report["by_depth"][depth] = {
                "count": len(usages),
                "success_rate": successes / len(usages) if usages else 0,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "total_retries": sum(u["retries_used"] for u in usages),
                "total_replans": sum(u["replans_used"] for u in usages),
            }

        return report

    def suggest_depth_adjustment(
        self, current_depth: ReasoningDepth, usage_report: dict[str, Any]
    ) -> ReasoningDepth | None:
        """Suggest whether to adjust reasoning depth based on usage patterns.

        Returns a suggested depth, or None if no adjustment needed.
        """
        depth_data = usage_report.get("by_depth", {}).get(current_depth.value)
        if not depth_data or depth_data["count"] < 5:
            return None  # Insufficient data

        # If success rate is very high and avg duration is low, can go lighter
        if depth_data["success_rate"] > 0.9 and depth_data["avg_duration"] < 10.0:
            depths = list(ReasoningDepth)
            idx = depths.index(current_depth)
            if idx > 0:
                return depths[idx - 1]

        # If success rate is low, should go deeper
        if depth_data["success_rate"] < 0.5:
            depths = list(ReasoningDepth)
            idx = depths.index(current_depth)
            if idx < len(depths) - 1:
                return depths[idx + 1]

        return None

    def recent_assessments(self, limit: int = 10) -> list[TaskComplexityAssessment]:
        return self._assessments[-limit:]

    def all_profiles(self) -> list[ResourceProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _depth_for_score(score: float) -> ReasoningDepth:
        """Map a combined complexity score to reasoning depth."""
        if score < 0.2:
            return ReasoningDepth.MINIMAL
        if score < 0.4:
            return ReasoningDepth.LIGHT
        if score < 0.6:
            return ReasoningDepth.STANDARD
        if score < 0.8:
            return ReasoningDepth.DEEP
        return ReasoningDepth.EXHAUSTIVE

    @staticmethod
    def _verification_for_depth(
        depth: ReasoningDepth, risk: float
    ) -> VerificationLevel:
        """Determine verification level from depth and risk."""
        base = {
            ReasoningDepth.MINIMAL: VerificationLevel.NONE,
            ReasoningDepth.LIGHT: VerificationLevel.BASIC,
            ReasoningDepth.STANDARD: VerificationLevel.STANDARD,
            ReasoningDepth.DEEP: VerificationLevel.THOROUGH,
            ReasoningDepth.EXHAUSTIVE: VerificationLevel.EXHAUSTIVE,
        }[depth]

        # Upgrade verification for high-risk tasks
        if risk > 0.7:
            levels = list(VerificationLevel)
            idx = levels.index(base)
            if idx < len(levels) - 1:
                return levels[idx + 1]

        return base


__all__ = [
    "ReasoningDepth",
    "ResourceAwareEngine",
    "ResourceProfile",
    "TaskComplexityAssessment",
    "VerificationLevel",
]
