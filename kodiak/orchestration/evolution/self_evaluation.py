"""Self-evaluation engine — structured post-execution analysis.

After significant tasks, Kodiak evaluates what worked, what failed,
where effort was wasted, and which assumptions were wrong.  Reduces
complex evaluations to structured evidence, never to a single score.
"""

from __future__ import annotations

from typing import Any

import structlog

from kodiak.orchestration.evolution.models import (
    DimensionScore,
    EvaluationDimension,
    EvaluationVerdict,
    SystemEvaluation,
    TaskEvaluation,
)

logger = structlog.get_logger(__name__)

# Default dimensions to evaluate
_DEFAULT_DIMENSIONS: tuple[EvaluationDimension, ...] = (
    EvaluationDimension.PLANNING_QUALITY,
    EvaluationDimension.STRATEGY_SELECTION,
    EvaluationDimension.AGENT_SELECTION,
    EvaluationDimension.TOOL_SELECTION,
    EvaluationDimension.MEMORY_RETRIEVAL,
    EvaluationDimension.VERIFICATION_QUALITY,
    EvaluationDimension.RECOVERY_BEHAVIOR,
    EvaluationDimension.CODE_GENERATION,
    EvaluationDimension.RESOURCE_EFFICIENCY,
    EvaluationDimension.EXECUTION_RELIABILITY,
)


def _verdict_for_score(score: float) -> EvaluationVerdict:
    """Map a 0.0-1.0 score to a verdict."""
    if score >= 0.8:
        return EvaluationVerdict.STRONG
    if score >= 0.6:
        return EvaluationVerdict.ADEQUATE
    if score >= 0.4:
        return EvaluationVerdict.WEAK
    if score > 0.0:
        return EvaluationVerdict.FAILING
    return EvaluationVerdict.UNKNOWN


class SelfEvaluationEngine:
    """Evaluates task execution across multiple dimensions.

    The engine produces structured evaluations that identify what
    worked, what failed, and where the system should improve.  It
    never reduces everything to a single number — each dimension
    carries its own evidence.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="self_evaluation_engine")
        self._evaluations: list[TaskEvaluation] = []

    def evaluate_task(
        self,
        *,
        task_id: str,
        goal: str,
        success: bool,
        attempts: int,
        replans: int,
        duration_seconds: float,
        reflection_results: list[dict[str, Any]] | None = None,
        verification_status: str = "unknown",
        memory_recalled: bool = False,
        selected_agent: str | None = None,
        error_message: str | None = None,
    ) -> TaskEvaluation:
        """Evaluate a completed task across multiple dimensions.

        Produces a structured TaskEvaluation with per-dimension scores,
        qualitative observations, and an overall assessment.
        """
        reflection_results = reflection_results or []

        # Score each dimension
        dimension_scores = self._score_dimensions(
            success=success,
            attempts=attempts,
            replans=replans,
            duration_seconds=duration_seconds,
            reflection_results=reflection_results,
            verification_status=verification_status,
            memory_recalled=memory_recalled,
            error_message=error_message,
        )

        # Qualitative analysis
        what_worked = self._identify_what_worked(
            success=success,
            verification_status=verification_status,
            memory_recalled=memory_recalled,
        )
        what_failed = self._identify_what_failed(
            success=success,
            reflection_results=reflection_results,
            error_message=error_message,
        )
        wasted_effort = self._identify_wasted_effort(
            attempts=attempts,
            replans=replans,
            reflection_results=reflection_results,
        )
        wrong_assumptions = self._identify_wrong_assumptions(
            reflection_results=reflection_results,
        )

        # Overall
        scores = [d.score for d in dimension_scores]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        overall_verdict = _verdict_for_score(overall_score)

        summary = self._build_summary(
            goal=goal,
            success=success,
            overall_score=overall_score,
            overall_verdict=overall_verdict,
            what_worked=what_worked,
            what_failed=what_failed,
        )

        evaluation = TaskEvaluation(
            task_id=task_id,
            goal=goal,
            dimension_scores=tuple(dimension_scores),
            what_worked=tuple(what_worked),
            what_failed=tuple(what_failed),
            wasted_effort=tuple(wasted_effort),
            wrong_assumptions=tuple(wrong_assumptions),
            failure_component=self._identify_failure_component(reflection_results),
            memory_helped=memory_recalled and success,
            planning_helped=replans == 0 and success,
            verification_caught_failure=verification_status == "failed",
            agent_appropriate=selected_agent is not None,
            overall_score=overall_score,
            overall_verdict=overall_verdict,
            summary=summary,
        )

        self._evaluations.append(evaluation)
        self._log.info(
            "task_evaluated",
            task_id=task_id,
            overall_score=round(overall_score, 3),
            overall_verdict=overall_verdict.value,
            weak_dimensions=[d.value for d in evaluation.weak_dimensions],
        )

        return evaluation

    def aggregate_system_evaluation(self) -> SystemEvaluation:
        """Aggregate all task evaluations into a system-level view."""
        if not self._evaluations:
            return SystemEvaluation()

        # Compute dimension averages
        dimension_totals: dict[EvaluationDimension, list[float]] = {}
        for eval_ in self._evaluations:
            for ds in eval_.dimension_scores:
                dimension_totals.setdefault(ds.dimension, []).append(ds.score)

        dimension_averages: list[DimensionScore] = []
        for dim, scores in dimension_totals.items():
            avg = sum(scores) / len(scores)
            dimension_averages.append(
                DimensionScore(
                    dimension=dim,
                    score=avg,
                    verdict=_verdict_for_score(avg),
                    evidence=f"Average across {len(scores)} evaluations",
                    confidence=min(len(scores) / 10.0, 1.0),
                    measurements={"count": len(scores), "min": min(scores), "max": max(scores)},
                )
            )

        dimension_averages.sort(key=lambda d: d.score)

        overall_scores = [e.overall_score for e in self._evaluations]
        overall_score = sum(overall_scores) / len(overall_scores)

        weakest = tuple(d.dimension for d in dimension_averages[:3] if d.score < 0.6)
        strongest = tuple(d.dimension for d in reversed(dimension_averages[:3]) if d.score >= 0.7)

        # Collect recurring failures
        all_failures: list[str] = []
        for e in self._evaluations:
            all_failures.extend(e.what_failed)

        failure_counts: dict[str, int] = {}
        for f in all_failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1
        recurring = tuple(
            f for f, count in sorted(failure_counts.items(), key=lambda x: -x[1]) if count >= 2
        )

        # Improvement opportunities from weak dimensions
        opportunities = tuple(
            f"Improve {d.dimension.value} (current score: {d.score:.2f})"
            for d in dimension_averages
            if d.score < 0.6
        )

        return SystemEvaluation(
            task_evaluations_count=len(self._evaluations),
            dimension_averages=tuple(dimension_averages),
            overall_score=overall_score,
            overall_verdict=_verdict_for_score(overall_score),
            weakest_dimensions=weakest,
            strongest_dimensions=strongest,
            recurring_failures=recurring,
            improvement_opportunities=opportunities,
        )

    def recent_evaluations(self, limit: int = 10) -> list[TaskEvaluation]:
        return self._evaluations[-limit:]

    def clear(self) -> None:
        self._evaluations.clear()

    # ------------------------------------------------------------------
    # Dimension scoring
    # ------------------------------------------------------------------

    def _score_dimensions(
        self,
        *,
        success: bool,
        attempts: int,
        replans: int,
        duration_seconds: float,
        reflection_results: list[dict[str, Any]],
        verification_status: str,
        memory_recalled: bool,
        error_message: str | None,
    ) -> list[DimensionScore]:
        scores: list[DimensionScore] = []

        # Planning quality: high if no replans needed
        planning_score = max(0.0, 1.0 - replans * 0.25)
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.PLANNING_QUALITY,
                score=planning_score,
                verdict=_verdict_for_score(planning_score),
                evidence=f"{replans} replan(s) needed" if replans else "No replanning required",
                measurements={"replans": replans, "attempts": attempts},
            )
        )

        # Strategy selection: inferred from attempt count and success
        strategy_score = 1.0 if success and attempts == 1 else max(0.3, 1.0 - (attempts - 1) * 0.2)
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.STRATEGY_SELECTION,
                score=strategy_score,
                verdict=_verdict_for_score(strategy_score),
                evidence=f"Succeeded in {attempts} attempt(s)"
                if success
                else f"Failed after {attempts} attempt(s)",
                measurements={"attempts": attempts, "success": success},
            )
        )

        # Agent selection: based on success
        agent_score = 1.0 if success else 0.4
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.AGENT_SELECTION,
                score=agent_score,
                verdict=_verdict_for_score(agent_score),
                evidence="Agent completed task successfully"
                if success
                else "Agent did not complete task",
            )
        )

        # Tool selection: based on success and no tool errors
        tool_score = 1.0 if success else 0.5
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.TOOL_SELECTION,
                score=tool_score,
                verdict=_verdict_for_score(tool_score),
                evidence="No tool selection errors detected",
            )
        )

        # Memory retrieval
        memory_score = 0.7 if memory_recalled else 0.4
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.MEMORY_RETRIEVAL,
                score=memory_score,
                verdict=_verdict_for_score(memory_score),
                evidence="Memory recalled and helped"
                if memory_recalled
                else "No memory recall or recall did not help",
            )
        )

        # Verification quality
        verification_score = (
            0.9
            if verification_status == "verified"
            else 0.5
            if verification_status == "inconclusive"
            else 0.3
            if verification_status == "failed"
            else 0.4
        )
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.VERIFICATION_QUALITY,
                score=verification_score,
                verdict=_verdict_for_score(verification_score),
                evidence=f"Verification status: {verification_status}",
            )
        )

        # Recovery behavior
        if success and attempts > 1:
            recovery_score = 0.8  # Recovered from initial failure
        elif not success and attempts > 1:
            recovery_score = 0.3  # Tried but failed to recover
        else:
            recovery_score = 0.6  # No recovery needed or attempted
        if success and attempts > 1:
            recovery_evidence = "Recovery succeeded"
        elif attempts == 1:
            recovery_evidence = "Recovery not needed"
        else:
            recovery_evidence = "Recovery failed"
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.RECOVERY_BEHAVIOR,
                score=recovery_score,
                verdict=_verdict_for_score(recovery_score),
                evidence=recovery_evidence,
            )
        )

        # Code generation: inferred from success
        code_score = 0.9 if success else 0.3
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.CODE_GENERATION,
                score=code_score,
                verdict=_verdict_for_score(code_score),
                evidence="Generated code satisfied requirements"
                if success
                else "Generated code did not satisfy requirements",
            )
        )

        # Resource efficiency
        efficiency_score = max(0.2, 1.0 - (attempts - 1) * 0.15 - replans * 0.1)
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.RESOURCE_EFFICIENCY,
                score=efficiency_score,
                verdict=_verdict_for_score(efficiency_score),
                evidence=f"Duration: {duration_seconds:.1f}s, {attempts} attempt(s)",
                measurements={"duration_seconds": duration_seconds, "attempts": attempts},
            )
        )

        # Execution reliability
        reliability_score = 1.0 if success else 0.2
        scores.append(
            DimensionScore(
                dimension=EvaluationDimension.EXECUTION_RELIABILITY,
                score=reliability_score,
                verdict=_verdict_for_score(reliability_score),
                evidence="Execution succeeded"
                if success
                else f"Execution failed: {error_message or 'unknown'}",
            )
        )

        return scores

    # ------------------------------------------------------------------
    # Qualitative analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_what_worked(
        *,
        success: bool,
        verification_status: str,
        memory_recalled: bool,
    ) -> list[str]:
        worked: list[str] = []
        if success:
            worked.append("Task completed successfully")
        if verification_status == "verified":
            worked.append("Verification confirmed the solution")
        if memory_recalled:
            worked.append("Memory retrieval provided useful context")
        return worked

    @staticmethod
    def _identify_what_failed(
        *,
        success: bool,
        reflection_results: list[dict[str, Any]],
        error_message: str | None,
    ) -> list[str]:
        failed: list[str] = []
        if not success:
            failed.append("Task did not complete successfully")
        for r in reflection_results:
            root_cause = r.get("root_cause", "")
            if root_cause:
                failed.append(f"Reflection identified: {root_cause}")
        if error_message:
            failed.append(f"Error: {error_message}")
        return failed

    @staticmethod
    def _identify_wasted_effort(
        *,
        attempts: int,
        replans: int,
        reflection_results: list[dict[str, Any]],
    ) -> list[str]:
        wasted: list[str] = []
        if attempts > 2:
            wasted.append(f"Required {attempts} attempts (excessive)")
        if replans > 0:
            wasted.append(f"Required {replans} replan(s)")
        for r in reflection_results:
            if r.get("strategy") == "retry":
                wasted.append("Retried without changing approach")
        return wasted

    @staticmethod
    def _identify_wrong_assumptions(
        reflection_results: list[dict[str, Any]],
    ) -> list[str]:
        assumptions: list[str] = []
        for r in reflection_results:
            root_cause = r.get("root_cause", "")
            if root_cause and "incorrect" in root_cause.lower():
                assumptions.append(f"Assumption was wrong: {root_cause}")
            category = r.get("category", "")
            if category == "incorrect_implementation":
                assumptions.append("Initial implementation approach was incorrect")
        return assumptions

    @staticmethod
    def _identify_failure_component(
        reflection_results: list[dict[str, Any]],
    ) -> str:
        for r in reversed(reflection_results):
            category = r.get("category", "")
            if category and category != "unknown":
                return category
        return ""

    @staticmethod
    def _build_summary(
        *,
        goal: str,
        success: bool,
        overall_score: float,
        overall_verdict: EvaluationVerdict,
        what_worked: list[str],
        what_failed: list[str],
    ) -> str:
        status = "succeeded" if success else "failed"
        parts = [
            f"Task '{goal[:60]}' {status}.",
            f"Overall score: {overall_score:.2f} ({overall_verdict.value}).",
        ]
        if what_worked:
            parts.append(f"What worked: {'; '.join(what_worked[:3])}.")
        if what_failed:
            parts.append(f"What failed: {'; '.join(what_failed[:3])}.")
        return " ".join(parts)


__all__ = ["SelfEvaluationEngine"]
