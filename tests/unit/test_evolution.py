"""Tests for Phase 6 intelligence evolution and meta-learning components."""

from __future__ import annotations

import pytest

from kodiak.orchestration.evolution.capability import (
    Capability,
    CapabilityCategory,
    CapabilityEvaluation,
    CapabilityPerformance,
    CapabilityTracker,
)
from kodiak.orchestration.evolution.failure_patterns import (
    FailurePattern,
    FailurePatternMiner,
    FailurePatternSeverity,
)
from kodiak.orchestration.evolution.health import (
    HealthDimension,
    HealthMetric,
    HealthStatus,
    SystemHealth,
    SystemHealthDashboard,
)
from kodiak.orchestration.evolution.improvement_queue import (
    ImprovementProposal,
    ImprovementQueue,
    ImprovementStatus,
)
from kodiak.orchestration.evolution.models import (
    DimensionScore,
    EvaluationDimension,
    EvaluationVerdict,
    SystemEvaluation,
    TaskEvaluation,
)
from kodiak.orchestration.evolution.self_evaluation import SelfEvaluationEngine


# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------


class TestDimensionScore:
    def test_creation(self) -> None:
        ds = DimensionScore(
            dimension=EvaluationDimension.PLANNING_QUALITY,
            score=0.85,
            verdict=EvaluationVerdict.STRONG,
            evidence="Good planning",
        )
        assert ds.dimension == EvaluationDimension.PLANNING_QUALITY
        assert ds.score == 0.85

    def test_to_dict(self) -> None:
        ds = DimensionScore(
            dimension=EvaluationDimension.CODE_GENERATION,
            score=0.7,
            verdict=EvaluationVerdict.ADEQUATE,
            measurements={"attempts": 1},
        )
        d = ds.to_dict()
        assert d["dimension"] == "code_generation"
        assert d["score"] == 0.7


class TestTaskEvaluation:
    def test_creation(self) -> None:
        te = TaskEvaluation(
            task_id="t1",
            goal="Fix tests",
            overall_score=0.8,
            overall_verdict=EvaluationVerdict.STRONG,
        )
        assert te.task_id == "t1"
        assert te.overall_score == 0.8

    def test_avg_score(self) -> None:
        te = TaskEvaluation(
            dimension_scores=(
                DimensionScore(dimension=EvaluationDimension.PLANNING_QUALITY, score=0.8, verdict=EvaluationVerdict.STRONG),
                DimensionScore(dimension=EvaluationDimension.CODE_GENERATION, score=0.6, verdict=EvaluationVerdict.ADEQUATE),
            ),
        )
        assert te.avg_score == pytest.approx(0.7)

    def test_weak_dimensions(self) -> None:
        te = TaskEvaluation(
            dimension_scores=(
                DimensionScore(dimension=EvaluationDimension.PLANNING_QUALITY, score=0.8, verdict=EvaluationVerdict.STRONG),
                DimensionScore(dimension=EvaluationDimension.MEMORY_RETRIEVAL, score=0.3, verdict=EvaluationVerdict.WEAK),
            ),
        )
        assert EvaluationDimension.MEMORY_RETRIEVAL in te.weak_dimensions
        assert EvaluationDimension.PLANNING_QUALITY not in te.weak_dimensions

    def test_to_dict(self) -> None:
        te = TaskEvaluation(task_id="t1", goal="test", overall_score=0.7)
        d = te.to_dict()
        assert d["task_id"] == "t1"


class TestSystemEvaluation:
    def test_creation(self) -> None:
        se = SystemEvaluation(
            task_evaluations_count=10,
            overall_score=0.75,
            overall_verdict=EvaluationVerdict.ADEQUATE,
        )
        assert se.task_evaluations_count == 10

    def test_to_dict(self) -> None:
        se = SystemEvaluation(overall_score=0.6, weakest_dimensions=(EvaluationDimension.PLANNING_QUALITY,))
        d = se.to_dict()
        assert d["overall_score"] == 0.6


# ---------------------------------------------------------------------------
# SelfEvaluationEngine tests
# ---------------------------------------------------------------------------


class TestSelfEvaluationEngine:
    def test_evaluate_successful_task(self) -> None:
        engine = SelfEvaluationEngine()
        eval_ = engine.evaluate_task(
            task_id="t1",
            goal="Fix broken test",
            success=True,
            attempts=1,
            replans=0,
            duration_seconds=5.0,
            verification_status="verified",
            memory_recalled=True,
        )

        assert eval_.overall_score > 0.5
        assert eval_.overall_verdict in {EvaluationVerdict.STRONG, EvaluationVerdict.ADEQUATE}
        assert len(eval_.dimension_scores) > 0
        assert eval_.memory_helped is True
        assert eval_.planning_helped is True

    def test_evaluate_failed_task(self) -> None:
        engine = SelfEvaluationEngine()
        eval_ = engine.evaluate_task(
            task_id="t2",
            goal="Implement feature",
            success=False,
            attempts=3,
            replans=2,
            duration_seconds=30.0,
            reflection_results=[
                {"root_cause": "Tests failed", "category": "test_failure", "strategy": "retry"},
            ],
            verification_status="failed",
        )

        assert eval_.overall_score < 0.6
        assert len(eval_.what_failed) > 0
        assert len(eval_.wasted_effort) > 0

    def test_evaluate_with_replans(self) -> None:
        engine = SelfEvaluationEngine()
        eval_ = engine.evaluate_task(
            task_id="t3",
            goal="Complex task",
            success=True,
            attempts=3,
            replans=2,
            duration_seconds=20.0,
        )

        # Planning quality should be lower due to replans
        planning_scores = [
            d for d in eval_.dimension_scores
            if d.dimension == EvaluationDimension.PLANNING_QUALITY
        ]
        assert len(planning_scores) == 1
        assert planning_scores[0].score < 0.8

    def test_aggregate_system_evaluation(self) -> None:
        engine = SelfEvaluationEngine()

        for i in range(5):
            engine.evaluate_task(
                task_id=f"t{i}",
                goal=f"Task {i}",
                success=i < 3,
                attempts=1 if i < 3 else 2,
                replans=0,
                duration_seconds=5.0,
            )

        system = engine.aggregate_system_evaluation()
        assert system.task_evaluations_count == 5
        assert system.overall_score > 0
        assert len(system.dimension_averages) > 0

    def test_aggregate_empty(self) -> None:
        engine = SelfEvaluationEngine()
        system = engine.aggregate_system_evaluation()
        assert system.task_evaluations_count == 0

    def test_recent_evaluations(self) -> None:
        engine = SelfEvaluationEngine()
        for i in range(5):
            engine.evaluate_task(
                task_id=f"t{i}", goal=f"T{i}", success=True, attempts=1, replans=0, duration_seconds=1.0
            )
        recent = engine.recent_evaluations(limit=3)
        assert len(recent) == 3

    def test_clear(self) -> None:
        engine = SelfEvaluationEngine()
        engine.evaluate_task(task_id="t1", goal="T", success=True, attempts=1, replans=0, duration_seconds=1.0)
        assert len(engine.recent_evaluations()) == 1
        engine.clear()
        assert len(engine.recent_evaluations()) == 0

    def test_dimension_scores_are_comprehensive(self) -> None:
        engine = SelfEvaluationEngine()
        eval_ = engine.evaluate_task(
            task_id="t1", goal="test", success=True, attempts=1, replans=0, duration_seconds=5.0
        )
        dimensions = {d.dimension for d in eval_.dimension_scores}
        assert EvaluationDimension.PLANNING_QUALITY in dimensions
        assert EvaluationDimension.CODE_GENERATION in dimensions
        assert EvaluationDimension.EXECUTION_RELIABILITY in dimensions


# ---------------------------------------------------------------------------
# Capability tests
# ---------------------------------------------------------------------------


class TestCapability:
    def test_creation(self) -> None:
        cap = Capability(
            name="test_generation",
            category=CapabilityCategory.TESTING,
            description="Generate tests",
        )
        assert cap.name == "test_generation"
        assert cap.is_active

    def test_health_score(self) -> None:
        cap = Capability(
            name="test",
            performance=CapabilityPerformance(
                total_attempts=10, successful_attempts=8
            ),
            evidence=("Evidence 1", "Evidence 2"),
        )
        assert cap.health_score > 0.5

    def test_health_score_with_limitations(self) -> None:
        cap = Capability(
            name="test",
            performance=CapabilityPerformance(
                total_attempts=10, successful_attempts=8
            ),
            known_limitations=("Limitation 1", "Limitation 2", "Limitation 3"),
        )
        # Limitations reduce health
        cap_no_lim = Capability(
            name="test",
            performance=CapabilityPerformance(
                total_attempts=10, successful_attempts=8
            ),
        )
        assert cap.health_score < cap_no_lim.health_score

    def test_to_dict(self) -> None:
        cap = Capability(name="test", category=CapabilityCategory.PLANNING)
        d = cap.to_dict()
        assert d["name"] == "test"
        assert d["category"] == "planning"


class TestCapabilityPerformance:
    def test_success_rate(self) -> None:
        p = CapabilityPerformance(total_attempts=10, successful_attempts=7)
        assert p.success_rate == pytest.approx(0.7)

    def test_success_rate_zero(self) -> None:
        p = CapabilityPerformance()
        assert p.success_rate == 0.0

    def test_to_dict(self) -> None:
        p = CapabilityPerformance(total_attempts=5, successful_attempts=3)
        d = p.to_dict()
        assert d["total_attempts"] == 5


class TestCapabilityTracker:
    def test_register_and_get(self) -> None:
        tracker = CapabilityTracker()
        cap = Capability(name="planning", category=CapabilityCategory.PLANNING)
        tracker.register(cap)
        assert tracker.get(cap.capability_id) is not None

    def test_get_by_name(self) -> None:
        tracker = CapabilityTracker()
        cap = Capability(name="testing", category=CapabilityCategory.TESTING)
        tracker.register(cap)
        assert tracker.get_by_name("testing") is not None
        assert tracker.get_by_name("nonexistent") is None

    def test_record_outcome(self) -> None:
        tracker = CapabilityTracker()
        cap = Capability(name="test", category=CapabilityCategory.TESTING)
        tracker.register(cap)

        tracker.record_outcome(cap.capability_id, success=True, duration_seconds=2.0)
        assert cap.performance.total_attempts == 1
        assert cap.performance.successful_attempts == 1

        tracker.record_outcome(cap.capability_id, success=False, failure_mode="timeout")
        assert cap.performance.total_attempts == 2
        assert cap.performance.failed_attempts == 1
        assert "timeout" in cap.performance.common_failure_modes

    def test_evaluate(self) -> None:
        tracker = CapabilityTracker()
        cap = Capability(name="test", category=CapabilityCategory.TESTING)
        tracker.register(cap)
        for _ in range(5):
            tracker.record_outcome(cap.capability_id, success=True)

        eval_ = tracker.evaluate(cap.capability_id)
        assert eval_ is not None
        assert eval_.score > 0.5

    def test_strong_and_weak(self) -> None:
        tracker = CapabilityTracker()
        strong = Capability(
            name="strong",
            performance=CapabilityPerformance(total_attempts=10, successful_attempts=9),
        )
        weak = Capability(
            name="weak",
            performance=CapabilityPerformance(total_attempts=10, successful_attempts=2),
        )
        tracker.register(strong)
        tracker.register(weak)

        assert len(tracker.strong_capabilities()) >= 1
        assert len(tracker.weak_capabilities()) >= 1

    def test_missing_capabilities(self) -> None:
        tracker = CapabilityTracker()
        cap = Capability(name="planning", category=CapabilityCategory.PLANNING)
        tracker.register(cap)

        missing = tracker.missing_capabilities(frozenset({"planning", "testing", "debugging"}))
        assert "testing" in missing
        assert "debugging" in missing
        assert "planning" not in missing

    def test_to_dict(self) -> None:
        tracker = CapabilityTracker()
        tracker.register(Capability(name="a", category=CapabilityCategory.PLANNING))
        d = tracker.to_dict()
        assert d["total"] == 1


# ---------------------------------------------------------------------------
# ImprovementQueue tests
# ---------------------------------------------------------------------------


class TestImprovementProposal:
    def test_creation(self) -> None:
        p = ImprovementProposal(
            title="Improve planner",
            problem="Planner creates unnecessary steps",
            expected_benefit="Reduce replanning",
            expected_impact_score=0.8,
            implementation_cost=0.3,
            risk=0.2,
        )
        assert p.title == "Improve planner"
        assert p.is_active

    def test_priority_score(self) -> None:
        p = ImprovementProposal(
            expected_impact_score=0.9,
            implementation_cost=0.1,
            risk=0.1,
        )
        assert p.priority_score > 0.5

    def test_priority_score_high_cost(self) -> None:
        p = ImprovementProposal(
            expected_impact_score=0.5,
            implementation_cost=0.9,
            risk=0.5,
        )
        assert p.priority_score < 0.5

    def test_to_dict(self) -> None:
        p = ImprovementProposal(title="test", status=ImprovementStatus.PROPOSED)
        d = p.to_dict()
        assert d["title"] == "test"
        assert d["status"] == "proposed"


class TestImprovementQueue:
    def test_add_and_get(self) -> None:
        queue = ImprovementQueue()
        p = ImprovementProposal(title="Test")
        queue.add(p)
        assert queue.get(p.proposal_id) is not None

    def test_update_status(self) -> None:
        queue = ImprovementQueue()
        p = ImprovementProposal(title="Test")
        queue.add(p)

        updated = queue.update_status(p.proposal_id, ImprovementStatus.PROPOSED)
        assert updated is not None
        assert updated.status == ImprovementStatus.PROPOSED

        updated = queue.update_status(p.proposal_id, ImprovementStatus.REJECTED, rejection_reason="Too risky")
        assert updated.status == ImprovementStatus.REJECTED
        assert updated.rejection_reason == "Too risky"
        assert updated.resolved_at is not None

    def test_active_proposals(self) -> None:
        queue = ImprovementQueue()
        queue.add(ImprovementProposal(title="Active", status=ImprovementStatus.PROPOSED))
        queue.add(ImprovementProposal(title="Rejected", status=ImprovementStatus.REJECTED))

        active = queue.active_proposals()
        assert len(active) == 1
        assert active[0].title == "Active"

    def test_ranked_proposals(self) -> None:
        queue = ImprovementQueue()
        queue.add(ImprovementProposal(title="High", expected_impact_score=0.9, implementation_cost=0.1))
        queue.add(ImprovementProposal(title="Low", expected_impact_score=0.2, implementation_cost=0.8))

        ranked = queue.ranked_proposals(limit=2)
        assert ranked[0].title == "High"

    def test_stats(self) -> None:
        queue = ImprovementQueue()
        queue.add(ImprovementProposal(title="A", status=ImprovementStatus.PROPOSED))
        queue.add(ImprovementProposal(title="B", status=ImprovementStatus.ACCEPTED))

        stats = queue.stats()
        assert stats["total"] == 2

    def test_eviction(self) -> None:
        queue = ImprovementQueue(max_proposals=2)
        queue.add(ImprovementProposal(title="A", expected_impact_score=0.9))
        queue.add(ImprovementProposal(title="B", expected_impact_score=0.1))
        assert len(queue) == 2

        queue.add(ImprovementProposal(title="C", expected_impact_score=0.8))
        assert len(queue) == 2


# ---------------------------------------------------------------------------
# FailurePatternMiner tests
# ---------------------------------------------------------------------------


class TestFailurePattern:
    def test_creation(self) -> None:
        p = FailurePattern(
            description="Recurring test failure",
            category="test_failure",
            occurrence_count=3,
        )
        assert p.is_recurring

    def test_not_recurring(self) -> None:
        p = FailurePattern(occurrence_count=1)
        assert not p.is_recurring

    def test_to_dict(self) -> None:
        p = FailurePattern(description="test", category="test_cat", severity=FailurePatternSeverity.HIGH)
        d = p.to_dict()
        assert d["description"] == "test"
        assert d["severity"] == "high"


class TestFailurePatternMiner:
    def test_analyze_recurring_failures(self) -> None:
        miner = FailurePatternMiner(min_occurrences=2)
        evaluations = [
            {"task_id": "t1", "failure_component": "planner", "what_failed": ["Plan was too broad"], "wasted_effort": [], "dimension_scores": []},
            {"task_id": "t2", "failure_component": "planner", "what_failed": ["Plan was too broad"], "wasted_effort": [], "dimension_scores": []},
            {"task_id": "t3", "failure_component": "planner", "what_failed": ["Plan was too broad"], "wasted_effort": [], "dimension_scores": []},
        ]

        patterns = miner.analyze_evaluations(evaluations)
        assert len(patterns) > 0

        # Should detect recurring planner failure
        planner_patterns = [p for p in patterns if "planner" in p.description.lower() or "planner" in str(p.affected_components)]
        assert len(planner_patterns) > 0

    def test_analyze_empty(self) -> None:
        miner = FailurePatternMiner()
        patterns = miner.analyze_evaluations([])
        assert patterns == []

    def test_analyze_waste_patterns(self) -> None:
        miner = FailurePatternMiner(min_occurrences=2)
        evaluations = [
            {"task_id": "t1", "failure_component": "", "what_failed": [], "wasted_effort": ["Retried without changing approach"], "dimension_scores": []},
            {"task_id": "t2", "failure_component": "", "what_failed": [], "wasted_effort": ["Retried without changing approach"], "dimension_scores": []},
        ]

        patterns = miner.analyze_evaluations(evaluations)
        waste_patterns = [p for p in patterns if p.category == "wasted_effort"]
        assert len(waste_patterns) > 0

    def test_recurring_patterns(self) -> None:
        miner = FailurePatternMiner(min_occurrences=2)
        evaluations = [
            {"task_id": "t1", "failure_component": "x", "what_failed": [], "wasted_effort": [], "dimension_scores": []},
            {"task_id": "t2", "failure_component": "x", "what_failed": [], "wasted_effort": [], "dimension_scores": []},
        ]
        miner.analyze_evaluations(evaluations)
        assert len(miner.recurring_patterns()) > 0

    def test_critical_patterns(self) -> None:
        miner = FailurePatternMiner(min_occurrences=1)
        evaluations = [
            {"task_id": f"t{i}", "failure_component": "critical_comp", "what_failed": [], "wasted_effort": [], "dimension_scores": []}
            for i in range(6)
        ]
        miner.analyze_evaluations(evaluations)
        critical = miner.critical_patterns()
        assert len(critical) > 0

    def test_to_dict(self) -> None:
        miner = FailurePatternMiner()
        d = miner.to_dict()
        assert "patterns" in d


# ---------------------------------------------------------------------------
# SystemHealthDashboard tests
# ---------------------------------------------------------------------------


class TestSystemHealthDashboard:
    def test_compute_health_healthy(self) -> None:
        dashboard = SystemHealthDashboard()
        health = dashboard.compute_health(
            total_tasks=10,
            successful_tasks=9,
            failed_tasks=1,
            total_replans=1,
            verification_pass_rate=0.9,
            memory_usefulness=0.8,
        )

        assert health.overall_score > 0.5
        assert health.overall_status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}
        assert len(health.metrics) > 0

    def test_compute_health_unhealthy(self) -> None:
        dashboard = SystemHealthDashboard()
        health = dashboard.compute_health(
            total_tasks=10,
            successful_tasks=2,
            failed_tasks=8,
            total_replans=5,
            human_interventions=5,
            verification_pass_rate=0.2,
        )

        assert health.overall_score < 0.5
        assert len(health.alerts) > 0

    def test_weakest_and_strongest(self) -> None:
        dashboard = SystemHealthDashboard()
        health = dashboard.compute_health(
            total_tasks=10,
            successful_tasks=8,
            verification_pass_rate=0.9,
            memory_usefulness=0.3,
        )
        assert health.weakest_dimension is not None
        assert health.strongest_dimension is not None

    def test_health_history(self) -> None:
        dashboard = SystemHealthDashboard()
        dashboard.compute_health(total_tasks=5, successful_tasks=4)
        dashboard.compute_health(total_tasks=5, successful_tasks=5)

        history = dashboard.health_history()
        assert len(history) == 2

    def test_trend(self) -> None:
        dashboard = SystemHealthDashboard()
        dashboard.compute_health(total_tasks=5, successful_tasks=3, dimension_scores={"reliability": 0.6})
        dashboard.compute_health(total_tasks=5, successful_tasks=5, dimension_scores={"reliability": 0.9})

        trend = dashboard.trend(HealthDimension.RELIABILITY)
        assert trend == "improving"

    def test_trend_insufficient_data(self) -> None:
        dashboard = SystemHealthDashboard()
        assert dashboard.trend(HealthDimension.RELIABILITY) == "insufficient_data"

    def test_to_dict(self) -> None:
        dashboard = SystemHealthDashboard()
        dashboard.compute_health(total_tasks=5, successful_tasks=4)
        d = dashboard.to_dict()
        assert d["latest"] is not None


class TestHealthMetric:
    def test_creation(self) -> None:
        m = HealthMetric(
            dimension=HealthDimension.RELIABILITY,
            score=0.85,
            status=HealthStatus.HEALTHY,
            evidence="9/10 tasks succeeded",
        )
        assert m.dimension == HealthDimension.RELIABILITY
        assert m.status == HealthStatus.HEALTHY

    def test_to_dict(self) -> None:
        m = HealthMetric(
            dimension=HealthDimension.RELIABILITY,
            score=0.8,
            status=HealthStatus.HEALTHY,
            trend="improving",
        )
        d = m.to_dict()
        assert d["trend"] == "improving"
