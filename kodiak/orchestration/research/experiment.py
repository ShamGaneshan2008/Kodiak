"""Experiment framework for controlled strategy comparison.

Designs, executes, and evaluates controlled experiments to test
hypotheses about strategy effectiveness.  Implements the baseline-first
rule: never claim improvement without a baseline.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from kodiak.orchestration.research.models import (
    Evidence,
    EvidenceStrength,
    ExperimentResult,
    Hypothesis,
    Observation,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExperimentPhase(enum.StrEnum):
    """Lifecycle phase of an experiment."""

    DESIGNED = "designed"
    BASELINE_RUNNING = "baseline_running"
    BASELINE_COMPLETE = "baseline_complete"
    CANDIDATE_RUNNING = "candidate_running"
    CANDIDATE_COMPLETE = "candidate_complete"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Experiment model
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """A controlled experiment to test a hypothesis.

    An experiment compares a baseline strategy against a candidate
    strategy under comparable conditions.
    """

    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    hypothesis_id: str = ""
    title: str = ""
    objective: str = ""
    description: str = ""

    # Strategies under comparison
    baseline_strategy_id: str = ""
    baseline_strategy_name: str = ""
    baseline_approach: str = ""

    candidate_strategy_id: str = ""
    candidate_strategy_name: str = ""
    candidate_approach: str = ""

    # Experiment design
    variables: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    procedure: tuple[str, ...] = ()
    measurements: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    failure_criteria: tuple[str, ...] = ()
    resource_budget: dict[str, Any] = field(default_factory=dict)

    # Execution state
    phase: ExperimentPhase = ExperimentPhase.DESIGNED
    baseline_result: ExperimentResult | None = None
    candidate_result: ExperimentResult | None = None

    # Analysis
    evidence_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    conclusion: str = ""

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.phase == ExperimentPhase.COMPLETED

    @property
    def improvement(self) -> float | None:
        """Calculate improvement of candidate over baseline (if both exist)."""
        if self.baseline_result is None or self.candidate_result is None:
            return None
        baseline_score = self.baseline_result.primary_metric
        candidate_score = self.candidate_result.primary_metric
        if baseline_score == 0:
            return None
        return (candidate_score - baseline_score) / abs(baseline_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "objective": self.objective,
            "description": self.description,
            "baseline_strategy_id": self.baseline_strategy_id,
            "baseline_strategy_name": self.baseline_strategy_name,
            "baseline_approach": self.baseline_approach,
            "candidate_strategy_id": self.candidate_strategy_id,
            "candidate_strategy_name": self.candidate_strategy_name,
            "candidate_approach": self.candidate_approach,
            "variables": list(self.variables),
            "controls": list(self.controls),
            "procedure": list(self.procedure),
            "measurements": list(self.measurements),
            "success_criteria": list(self.success_criteria),
            "failure_criteria": list(self.failure_criteria),
            "resource_budget": dict(self.resource_budget),
            "phase": self.phase.value,
            "baseline_result": self.baseline_result.to_dict() if self.baseline_result else None,
            "candidate_result": self.candidate_result.to_dict() if self.candidate_result else None,
            "evidence_ids": list(self.evidence_ids),
            "observation_ids": list(self.observation_ids),
            "conclusion": self.conclusion,
            "improvement": self.improvement,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# ExperimentResult model (re-exported in __init__ but defined here for clarity)
# ---------------------------------------------------------------------------

# ExperimentResult is actually defined in models.py. We keep the design
# engine using it from there.


# ---------------------------------------------------------------------------
# ExperimentDesignEngine
# ---------------------------------------------------------------------------


class ExperimentDesignEngine:
    """Designs controlled experiments to test hypotheses.

    The engine creates experiment specifications that compare baseline
    and candidate strategies under controlled conditions.  It does not
    execute experiments — that is the caller's responsibility.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="experiment_design_engine")

    def design_experiment(
        self,
        hypothesis: Hypothesis,
        *,
        baseline_strategy_id: str = "",
        baseline_strategy_name: str = "current_strategy",
        baseline_approach: str = "",
        candidate_strategy_id: str = "",
        candidate_strategy_name: str = "candidate_strategy",
        candidate_approach: str = "",
        task_categories: tuple[str, ...] = (),
        resource_budget: dict[str, Any] | None = None,
    ) -> Experiment:
        """Design a controlled experiment for a hypothesis.

        Creates an experiment that fairly compares the baseline and
        candidate strategies, controlling important variables.
        """
        objective = f"Test hypothesis: {hypothesis.statement}"

        variables = self._identify_variables(baseline_approach, candidate_approach)
        controls = self._default_controls(task_categories)
        procedure = self._default_procedure()
        measurements = self._default_measurements()
        success_criteria = self._default_success_criteria(hypothesis)
        failure_criteria = self._default_failure_criteria(hypothesis)

        experiment = Experiment(
            hypothesis_id=hypothesis.hypothesis_id,
            title=f"Experiment: {hypothesis.statement[:80]}",
            objective=objective,
            description=self._design_description(hypothesis),
            baseline_strategy_id=baseline_strategy_id,
            baseline_strategy_name=baseline_strategy_name,
            baseline_approach=baseline_approach,
            candidate_strategy_id=candidate_strategy_id,
            candidate_strategy_name=candidate_strategy_name,
            candidate_approach=candidate_approach,
            variables=variables,
            controls=controls,
            procedure=procedure,
            measurements=measurements,
            success_criteria=success_criteria,
            failure_criteria=failure_criteria,
            resource_budget=resource_budget or {"max_runs": 10, "timeout_seconds": 300},
        )

        self._log.info(
            "experiment_designed",
            experiment_id=experiment.experiment_id,
            hypothesis_id=hypothesis.hypothesis_id,
            variables=len(variables),
        )

        return experiment

    def record_baseline_result(
        self, experiment: Experiment, result: ExperimentResult
    ) -> Experiment:
        """Record the baseline result of an experiment."""
        experiment.baseline_result = result
        experiment.phase = ExperimentPhase.BASELINE_COMPLETE
        self._log.info(
            "baseline_result_recorded",
            experiment_id=experiment.experiment_id,
            primary_metric=result.primary_metric,
        )
        return experiment

    def record_candidate_result(
        self, experiment: Experiment, result: ExperimentResult
    ) -> Experiment:
        """Record the candidate result of an experiment."""
        experiment.candidate_result = result
        experiment.phase = ExperimentPhase.CANDIDATE_COMPLETE
        self._log.info(
            "candidate_result_recorded",
            experiment_id=experiment.experiment_id,
            primary_metric=result.primary_metric,
        )
        return experiment

    def analyze_results(self, experiment: Experiment) -> Experiment:
        """Analyze completed experiment results and generate evidence.

        Generates evidence supporting or contradicting the hypothesis.
        Records observations about unexpected behavior.
        """
        if experiment.baseline_result is None or experiment.candidate_result is None:
            experiment.phase = ExperimentPhase.FAILED
            experiment.conclusion = "Cannot analyze: missing baseline or candidate result."
            return experiment

        experiment.phase = ExperimentPhase.ANALYZING

        improvement = experiment.improvement
        baseline = experiment.baseline_result
        candidate = experiment.candidate_result

        # Build conclusion
        if improvement is not None and improvement > 0:
            experiment.conclusion = (
                f"Candidate improved over baseline by {improvement:.1%}. "
                f"Baseline: {baseline.primary_metric:.3f}, "
                f"Candidate: {candidate.primary_metric:.3f}."
            )
        elif improvement is not None and improvement < 0:
            experiment.conclusion = (
                f"Candidate degraded vs baseline by {abs(improvement):.1%}. "
                f"Baseline: {baseline.primary_metric:.3f}, "
                f"Candidate: {candidate.primary_metric:.3f}."
            )
        else:
            experiment.conclusion = (
                f"Results inconclusive. "
                f"Baseline: {baseline.primary_metric:.3f}, "
                f"Candidate: {candidate.primary_metric:.3f}."
            )

        experiment.phase = ExperimentPhase.COMPLETED
        experiment.completed_at = datetime.now(UTC)

        self._log.info(
            "experiment_analyzed",
            experiment_id=experiment.experiment_id,
            improvement=f"{improvement:.1%}" if improvement is not None else "N/A",
        )

        return experiment

    def generate_evidence(self, experiment: Experiment, hypothesis_id: str) -> Evidence:
        """Generate evidence from a completed experiment.

        Evidence must have provenance (experiment_id).
        """
        improvement = experiment.improvement
        if improvement is None:
            supports = None
            strength = EvidenceStrength.WEAK
            summary = "Experiment produced inconclusive results."
            measurements: dict[str, Any] = {}
        elif improvement > 0:
            supports = True
            strength = EvidenceStrength.STRONG if improvement > 0.1 else EvidenceStrength.MODERATE
            summary = f"Candidate strategy improved by {improvement:.1%} over baseline."
            measurements = {
                "improvement": improvement,
                "baseline_metric": experiment.baseline_result.primary_metric
                if experiment.baseline_result
                else None,
                "candidate_metric": experiment.candidate_result.primary_metric
                if experiment.candidate_result
                else None,
            }
        else:
            supports = False
            strength = EvidenceStrength.MODERATE
            summary = f"Candidate strategy degraded by {abs(improvement):.1%} vs baseline."
            measurements = {
                "degradation": abs(improvement),
                "baseline_metric": experiment.baseline_result.primary_metric
                if experiment.baseline_result
                else None,
                "candidate_metric": experiment.candidate_result.primary_metric
                if experiment.candidate_result
                else None,
            }

        evidence = Evidence(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment.experiment_id,
            strength=strength,
            summary=summary,
            measurements=measurements,
            supports_hypothesis=supports,
            confidence=0.8
            if strength in {EvidenceStrength.STRONG, EvidenceStrength.MODERATE}
            else 0.4,
            source=f"experiment:{experiment.experiment_id}",
        )

        self._log.info(
            "evidence_generated",
            evidence_id=evidence.evidence_id,
            supports=supports,
            strength=strength.value,
        )

        return evidence

    def generate_observation(self, experiment: Experiment) -> Observation:
        """Generate an observation from experiment behavior."""
        improvement = experiment.improvement
        if improvement is not None and improvement > 0.2:
            category = "unexpected_success"
            title = f"Strong improvement detected: {improvement:.1%}"
        elif improvement is not None and improvement < -0.2:
            category = "unexpected_degradation"
            title = f"Strong degradation detected: {abs(improvement):.1%}"
        else:
            category = "experiment_outcome"
            title = "Experiment completed with expected-range results"

        observation = Observation(
            title=title,
            summary=experiment.conclusion,
            source=f"experiment:{experiment.experiment_id}",
            category=category,
            measurements={
                "improvement": improvement,
                "baseline": experiment.baseline_result.to_dict()
                if experiment.baseline_result
                else None,
                "candidate": experiment.candidate_result.to_dict()
                if experiment.candidate_result
                else None,
            },
        )

        self._log.info(
            "observation_generated",
            observation_id=observation.observation_id,
            category=category,
        )

        return observation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_variables(baseline_approach: str, candidate_approach: str) -> tuple[str, ...]:
        """Identify the key variables that differ between strategies."""
        variables: list[str] = []
        if baseline_approach != candidate_approach:
            variables.append("strategy_approach")
        return tuple(variables)

    @staticmethod
    def _default_controls(task_categories: tuple[str, ...] = ()) -> tuple[str, ...]:
        controls = [
            "same_task_set",
            "same_repository_state",
            "same_constraints",
            "same_tool_availability",
            "same_evaluation_criteria",
        ]
        if task_categories:
            controls.append(f"task_categories:{','.join(task_categories)}")
        return tuple(controls)

    @staticmethod
    def _default_procedure() -> tuple[str, ...]:
        return (
            "1. Establish baseline by running strategy A on task set.",
            "2. Record baseline metrics.",
            "3. Reset repository to initial state.",
            "4. Run candidate strategy B on same task set.",
            "5. Record candidate metrics.",
            "6. Compare results.",
            "7. Generate evidence and observations.",
        )

    @staticmethod
    def _default_measurements() -> tuple[str, ...]:
        return (
            "task_success_rate",
            "time_to_completion",
            "tool_calls",
            "regression_rate",
            "recovery_rate",
        )

    @staticmethod
    def _default_success_criteria(hypothesis: Hypothesis) -> tuple[str, ...]:
        criteria = [
            "Candidate achieves higher task success rate than baseline.",
            "Candidate does not cause more regressions than baseline.",
            "Improvement is measurable and reproducible.",
        ]
        if hypothesis.expected_benefit:
            criteria.append(f"Expected benefit: {hypothesis.expected_benefit}")
        return tuple(criteria)

    @staticmethod
    def _default_failure_criteria(hypothesis: Hypothesis) -> tuple[str, ...]:
        return (
            "Candidate performs worse than baseline.",
            "Candidate causes regressions.",
            "Results are not reproducible.",
            "Resource budget exceeded.",
        )

    @staticmethod
    def _design_description(hypothesis: Hypothesis) -> str:
        parts = [
            f"This experiment tests the following hypothesis:\n{hypothesis.statement}",
            "",
            f"Rationale: {hypothesis.rationale}",
            "",
            "The experiment compares a baseline strategy against a candidate strategy "
            "under controlled conditions to determine whether the candidate provides "
            "a measurable improvement.",
        ]
        if hypothesis.expected_benefit:
            parts.append(f"\nExpected benefit: {hypothesis.expected_benefit}")
        return "\n".join(parts)


__all__ = [
    "Experiment",
    "ExperimentDesignEngine",
    "ExperimentPhase",
]
