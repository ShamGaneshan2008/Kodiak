"""Tests for Phase 5 research subsystem components."""

from __future__ import annotations

import pytest

from kodiak.orchestration.research.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkTaskCategory,
)
from kodiak.orchestration.research.composer import StrategyComposer
from kodiak.orchestration.research.experiment import (
    ExperimentDesignEngine,
    ExperimentPhase,
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
from kodiak.orchestration.strategy import (
    EngineeringStrategy,
    ProblemClass,
    StrategyOutcome,
)

# ---------------------------------------------------------------------------
# Research Object Model tests
# ---------------------------------------------------------------------------


class TestResearchProblem:
    def test_creation(self) -> None:
        problem = ResearchProblem(
            title="Test problem",
            description="Description",
            priority=ResearchProblemPriority.HIGH,
        )
        assert problem.title == "Test problem"
        assert problem.priority == ResearchProblemPriority.HIGH
        assert not problem.is_resolved
        assert problem.problem_id

    def test_resolve(self) -> None:
        problem = ResearchProblem(title="Test")
        assert not problem.is_resolved
        from datetime import UTC, datetime

        problem.resolved_at = datetime.now(UTC)
        assert problem.is_resolved

    def test_to_dict(self) -> None:
        problem = ResearchProblem(title="Test", tags=("a", "b"))
        d = problem.to_dict()
        assert d["title"] == "Test"
        assert d["tags"] == ["a", "b"]
        assert "problem_id" in d


class TestProblemDecomposition:
    def test_creation(self) -> None:
        decomp = ProblemDecomposition(
            known_facts=("fact1",),
            unknowns=("unknown1",),
            hypotheses=("hypothesis1",),
        )
        assert len(decomp.known_facts) == 1
        assert len(decomp.unknowns) == 1

    def test_to_dict(self) -> None:
        decomp = ProblemDecomposition(known_facts=("a",), unknowns=("b",))
        d = decomp.to_dict()
        assert d["known_facts"] == ["a"]
        assert d["unknowns"] == ["b"]


class TestKnowledgeGap:
    def test_creation(self) -> None:
        gap = KnowledgeGap(
            description="We don't know X",
            known_context="We know Y",
            unknown_quantity="Whether Z works",
        )
        assert gap.description == "We don't know X"
        assert gap.gap_id

    def test_to_dict(self) -> None:
        gap = KnowledgeGap(description="test")
        d = gap.to_dict()
        assert d["description"] == "test"


class TestHypothesis:
    def test_creation(self) -> None:
        h = Hypothesis(
            statement="Strategy B is better than A",
            rationale="Evidence suggests B handles edge cases.",
            status=HypothesisStatus.PROPOSED,
        )
        assert h.statement == "Strategy B is better than A"
        assert h.status == HypothesisStatus.PROPOSED
        assert h.hypothesis_id

    def test_to_dict(self) -> None:
        h = Hypothesis(
            statement="test",
            related_strategy_ids=("s1", "s2"),
        )
        d = h.to_dict()
        assert d["statement"] == "test"
        assert d["related_strategy_ids"] == ["s1", "s2"]


class TestEvidence:
    def test_creation(self) -> None:
        e = Evidence(
            experiment_id="exp1",
            strength=EvidenceStrength.STRONG,
            summary="Strong support",
            supports_hypothesis=True,
        )
        assert e.experiment_id == "exp1"
        assert e.has_provenance

    def test_no_provenance(self) -> None:
        e = Evidence(summary="No source")
        assert not e.has_provenance

    def test_to_dict(self) -> None:
        e = Evidence(
            experiment_id="exp1",
            strength=EvidenceStrength.MODERATE,
            measurements={"improvement": 0.15},
        )
        d = e.to_dict()
        assert d["strength"] == "moderate"
        assert d["measurements"]["improvement"] == 0.15


class TestObservation:
    def test_creation(self) -> None:
        o = Observation(
            title="Unexpected success",
            category="unexpected_success",
            summary="Strategy worked better than expected.",
        )
        assert o.title == "Unexpected success"
        assert o.category == "unexpected_success"

    def test_to_dict(self) -> None:
        o = Observation(title="test", category="test_cat")
        d = o.to_dict()
        assert d["title"] == "test"


class TestConclusion:
    def test_creation_with_evidence(self) -> None:
        c = Conclusion(
            statement="Hypothesis supported",
            classification=KnowledgeClassification.SUPPORTED,
            supporting_evidence_ids=("e1", "e2"),
        )
        assert c.has_evidence
        assert c.net_evidence_strength == 2

    def test_creation_without_evidence(self) -> None:
        c = Conclusion(statement="No evidence")
        assert not c.has_evidence
        assert c.net_evidence_strength == 0

    def test_to_dict(self) -> None:
        c = Conclusion(
            statement="test",
            classification=KnowledgeClassification.OBSERVED,
            supporting_evidence_ids=("e1",),
        )
        d = c.to_dict()
        assert d["classification"] == "observed"
        assert d["has_evidence"] is True


class TestLesson:
    def test_creation(self) -> None:
        lesson = Lesson(
            statement="Dependency-aware planning improves success rate.",
            domain="dependency_resolution",
            scope="medium_repos",
            confidence=0.8,
        )
        assert lesson.statement == "Dependency-aware planning improves success rate."
        assert lesson.domain == "dependency_resolution"

    def test_to_dict(self) -> None:
        lesson = Lesson(statement="test", domain="test_domain")
        d = lesson.to_dict()
        assert d["statement"] == "test"


class TestStrategyVersion:
    def test_creation(self) -> None:
        v = StrategyVersion(
            strategy_id="s1",
            version_number=1,
            name="test_v1",
            approach="Initial approach",
        )
        assert v.strategy_id == "s1"
        assert v.version_number == 1

    def test_to_dict(self) -> None:
        v = StrategyVersion(strategy_id="s1", version_number=2)
        d = v.to_dict()
        assert d["strategy_id"] == "s1"
        assert d["version_number"] == 2


class TestNegativeKnowledge:
    def test_creation(self) -> None:
        nk = NegativeKnowledge(
            strategy_description="Mass dependency upgrade",
            problem_class="dependency_conflict",
            result="Increased compatibility failures.",
            conclusion="Poor strategy for localized dependency conflicts.",
        )
        assert nk.strategy_description == "Mass dependency upgrade"
        assert nk.knowledge_id

    def test_to_dict(self) -> None:
        nk = NegativeKnowledge(strategy_description="test", conclusion="bad")
        d = nk.to_dict()
        assert d["strategy_description"] == "test"


class TestExperimentResult:
    def test_creation(self) -> None:
        r = ExperimentResult(
            strategy_id="s1",
            strategy_name="test",
            primary_metric=0.85,
            total_tasks=10,
            successful_tasks=8,
        )
        assert r.primary_metric == 0.85
        assert "8/10" in r.summary

    def test_to_dict(self) -> None:
        r = ExperimentResult(primary_metric=0.75, total_tasks=5)
        d = r.to_dict()
        assert d["primary_metric"] == 0.75


# ---------------------------------------------------------------------------
# ResearchMemory tests
# ---------------------------------------------------------------------------


class TestResearchMemory:
    def test_store_and_retrieve_problem(self) -> None:
        memory = ResearchMemory()
        problem = ResearchProblem(title="Test problem", priority=ResearchProblemPriority.HIGH)
        memory.store_problem(problem)

        retrieved = memory.get_problem(problem.problem_id)
        assert retrieved is not None
        assert retrieved.title == "Test problem"

    def test_retrieve_problems_by_priority(self) -> None:
        memory = ResearchMemory()
        memory.store_problem(ResearchProblem(title="Low", priority=ResearchProblemPriority.LOW))
        memory.store_problem(ResearchProblem(title="High", priority=ResearchProblemPriority.HIGH))
        memory.store_problem(
            ResearchProblem(title="Critical", priority=ResearchProblemPriority.CRITICAL)
        )

        problems = memory.retrieve_problems()
        assert len(problems) == 3
        assert problems[0].priority == ResearchProblemPriority.CRITICAL
        assert problems[1].priority == ResearchProblemPriority.HIGH

    def test_resolve_problem(self) -> None:
        memory = ResearchMemory()
        problem = ResearchProblem(title="Test")
        memory.store_problem(problem)
        assert memory.resolve_problem(problem.problem_id)
        assert memory.get_problem(problem.problem_id).is_resolved

    def test_store_and_retrieve_hypothesis(self) -> None:
        memory = ResearchMemory()
        h = Hypothesis(
            statement="A > B",
            status=HypothesisStatus.PROPOSED,
            related_problem_id="p1",
        )
        memory.store_hypothesis(h)

        retrieved = memory.get_hypothesis(h.hypothesis_id)
        assert retrieved is not None
        assert retrieved.statement == "A > B"

    def test_update_hypothesis_status(self) -> None:
        memory = ResearchMemory()
        h = Hypothesis(statement="test")
        memory.store_hypothesis(h)

        updated = memory.update_hypothesis_status(h.hypothesis_id, HypothesisStatus.SUPPORTED)
        assert updated is not None
        assert updated.status == HypothesisStatus.SUPPORTED

    def test_retrieve_hypotheses_by_problem(self) -> None:
        memory = ResearchMemory()
        h1 = Hypothesis(statement="H1", related_problem_id="p1")
        h2 = Hypothesis(statement="H2", related_problem_id="p2")
        memory.store_hypothesis(h1)
        memory.store_hypothesis(h2)

        results = memory.retrieve_hypotheses(problem_id="p1")
        assert len(results) == 1
        assert results[0].statement == "H1"

    def test_store_and_retrieve_evidence(self) -> None:
        memory = ResearchMemory()
        e = Evidence(
            experiment_id="exp1",
            hypothesis_id="h1",
            strength=EvidenceStrength.STRONG,
            summary="Strong support",
            supports_hypothesis=True,
        )
        memory.store_evidence(e)

        results = memory.retrieve_evidence_for_hypothesis("h1")
        assert len(results) == 1
        assert results[0].strength == EvidenceStrength.STRONG

    def test_retrieve_evidence_for_experiment(self) -> None:
        memory = ResearchMemory()
        e1 = Evidence(experiment_id="exp1", summary="e1")
        e2 = Evidence(experiment_id="exp2", summary="e2")
        memory.store_evidence(e1)
        memory.store_evidence(e2)

        results = memory.retrieve_evidence_for_experiment("exp1")
        assert len(results) == 1

    def test_store_and_retrieve_observation(self) -> None:
        memory = ResearchMemory()
        o = Observation(title="Unexpected", category="unexpected_success")
        memory.store_observation(o)

        results = memory.retrieve_observations(category="unexpected_success")
        assert len(results) == 1

    def test_store_and_retrieve_conclusion(self) -> None:
        memory = ResearchMemory()
        c = Conclusion(
            hypothesis_id="h1",
            statement="Supported",
            classification=KnowledgeClassification.SUPPORTED,
            supporting_evidence_ids=("e1",),
        )
        memory.store_conclusion(c)

        results = memory.retrieve_conclusions_for_hypothesis("h1")
        assert len(results) == 1
        assert results[0].statement == "Supported"

    def test_store_and_retrieve_lesson(self) -> None:
        memory = ResearchMemory()
        lesson = Lesson(
            statement="Lesson learned",
            domain="test_fixing",
            confidence=0.9,
        )
        memory.store_lesson(lesson)

        results = memory.retrieve_lessons(domain="test_fixing")
        assert len(results) == 1

    def test_store_and_retrieve_strategy_version(self) -> None:
        memory = ResearchMemory()
        v1 = StrategyVersion(strategy_id="s1", version_number=1, name="v1", approach="A")
        v2 = StrategyVersion(strategy_id="s1", version_number=2, name="v2", approach="B")
        memory.store_strategy_version(v1)
        memory.store_strategy_version(v2)

        versions = memory.retrieve_strategy_versions("s1")
        assert len(versions) == 2
        assert versions[0].version_number == 1
        assert versions[1].version_number == 2

        latest = memory.get_latest_strategy_version("s1")
        assert latest is not None
        assert latest.version_number == 2

    def test_store_and_retrieve_negative_knowledge(self) -> None:
        memory = ResearchMemory()
        nk = NegativeKnowledge(
            strategy_description="Bad approach",
            problem_class="test_failure",
            conclusion="Doesn't work",
        )
        memory.store_negative_knowledge(nk)

        results = memory.retrieve_negative_knowledge(problem_class="test_failure")
        assert len(results) == 1

    def test_research_summary_for_problem(self) -> None:
        memory = ResearchMemory()
        problem = ResearchProblem(title="Test", problem_id="p1")
        memory.store_problem(problem)
        h = Hypothesis(statement="H1", related_problem_id="p1")
        memory.store_hypothesis(h)

        summary = memory.research_summary_for_problem("p1")
        assert "problem" in summary
        assert len(summary["hypotheses"]) == 1

    def test_research_summary_nonexistent_problem(self) -> None:
        memory = ResearchMemory()
        summary = memory.research_summary_for_problem("nonexistent")
        assert summary.get("error") == "problem_not_found"

    def test_stats(self) -> None:
        memory = ResearchMemory()
        memory.store_problem(ResearchProblem(title="p"))
        memory.store_hypothesis(Hypothesis(statement="h"))
        memory.store_evidence(Evidence(summary="e"))

        stats = memory.stats()
        assert stats["problems"] == 1
        assert stats["hypotheses"] == 1
        assert stats["evidence"] == 1


# ---------------------------------------------------------------------------
# ExperimentDesignEngine tests
# ---------------------------------------------------------------------------


class TestExperimentDesignEngine:
    def test_design_experiment(self) -> None:
        engine = ExperimentDesignEngine()
        hypothesis = Hypothesis(
            statement="Strategy B is faster than A",
            rationale="B uses caching.",
            expected_benefit="20% faster execution",
        )

        experiment = engine.design_experiment(
            hypothesis,
            baseline_strategy_name="strategy_a",
            candidate_strategy_name="strategy_b",
            baseline_approach="Direct execution",
            candidate_approach="Cached execution",
        )

        assert experiment.hypothesis_id == hypothesis.hypothesis_id
        assert experiment.phase == ExperimentPhase.DESIGNED
        assert experiment.baseline_strategy_name == "strategy_a"
        assert experiment.candidate_strategy_name == "strategy_b"
        assert len(experiment.controls) > 0
        assert len(experiment.procedure) > 0

    def test_record_results_and_analyze(self) -> None:
        engine = ExperimentDesignEngine()
        hypothesis = Hypothesis(statement="B > A")
        experiment = engine.design_experiment(hypothesis)

        baseline = ExperimentResult(
            strategy_id="a", strategy_name="A", primary_metric=0.7, total_tasks=10
        )
        candidate = ExperimentResult(
            strategy_id="b", strategy_name="B", primary_metric=0.85, total_tasks=10
        )

        engine.record_baseline_result(experiment, baseline)
        assert experiment.phase == ExperimentPhase.BASELINE_COMPLETE

        engine.record_candidate_result(experiment, candidate)
        assert experiment.phase == ExperimentPhase.CANDIDATE_COMPLETE

        engine.analyze_results(experiment)
        assert experiment.phase == ExperimentPhase.COMPLETED
        assert experiment.improvement is not None
        assert experiment.improvement > 0
        assert experiment.completed_at is not None

    def test_generate_evidence(self) -> None:
        engine = ExperimentDesignEngine()
        hypothesis = Hypothesis(statement="B > A")
        experiment = engine.design_experiment(hypothesis)

        engine.record_baseline_result(
            experiment,
            ExperimentResult(strategy_id="a", primary_metric=0.7, total_tasks=10),
        )
        engine.record_candidate_result(
            experiment,
            ExperimentResult(strategy_id="b", primary_metric=0.85, total_tasks=10),
        )
        engine.analyze_results(experiment)

        evidence = engine.generate_evidence(experiment, hypothesis.hypothesis_id)
        assert evidence.has_provenance
        assert evidence.experiment_id == experiment.experiment_id
        assert evidence.supports_hypothesis is True
        assert evidence.strength in {EvidenceStrength.STRONG, EvidenceStrength.MODERATE}

    def test_generate_observation(self) -> None:
        engine = ExperimentDesignEngine()
        hypothesis = Hypothesis(statement="B > A")
        experiment = engine.design_experiment(hypothesis)

        engine.record_baseline_result(
            experiment,
            ExperimentResult(strategy_id="a", primary_metric=0.7, total_tasks=10),
        )
        engine.record_candidate_result(
            experiment,
            ExperimentResult(strategy_id="b", primary_metric=0.9, total_tasks=10),
        )
        engine.analyze_results(experiment)

        observation = engine.generate_observation(experiment)
        assert observation.observation_id
        assert observation.category == "unexpected_success"

    def test_analyze_incomplete_experiment(self) -> None:
        engine = ExperimentDesignEngine()
        hypothesis = Hypothesis(statement="test")
        experiment = engine.design_experiment(hypothesis)

        # No results recorded
        engine.analyze_results(experiment)
        assert experiment.phase == ExperimentPhase.FAILED

    def test_improvement_calculation(self) -> None:
        engine = ExperimentDesignEngine()
        hypothesis = Hypothesis(statement="test")
        experiment = engine.design_experiment(hypothesis)

        engine.record_baseline_result(
            experiment,
            ExperimentResult(strategy_id="a", primary_metric=10.0),
        )
        engine.record_candidate_result(
            experiment,
            ExperimentResult(strategy_id="b", primary_metric=12.0),
        )

        assert experiment.improvement == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestBenchmarkTask:
    def test_creation(self) -> None:
        task = BenchmarkTask(
            title="Fix broken test",
            category=BenchmarkTaskCategory.TEST_REPAIR,
            objective="Make tests pass",
            difficulty=3,
        )
        assert task.title == "Fix broken test"
        assert task.category == BenchmarkTaskCategory.TEST_REPAIR

    def test_to_dict(self) -> None:
        task = BenchmarkTask(title="test", category=BenchmarkTaskCategory.BUG_FIXING)
        d = task.to_dict()
        assert d["category"] == "bug_fixing"


class TestBenchmarkSuite:
    def test_creation(self) -> None:
        tasks = (
            BenchmarkTask(title="T1", category=BenchmarkTaskCategory.BUG_FIXING),
            BenchmarkTask(title="T2", category=BenchmarkTaskCategory.TEST_REPAIR),
        )
        suite = BenchmarkSuite(name="Test Suite", tasks=tasks)
        assert suite.task_count == 2
        assert "bug_fixing" in suite.categories
        assert "test_repair" in suite.categories

    def test_tasks_by_category(self) -> None:
        tasks = (
            BenchmarkTask(title="T1", category=BenchmarkTaskCategory.BUG_FIXING),
            BenchmarkTask(title="T2", category=BenchmarkTaskCategory.BUG_FIXING),
            BenchmarkTask(title="T3", category=BenchmarkTaskCategory.TEST_REPAIR),
        )
        suite = BenchmarkSuite(name="Test", tasks=tasks)
        bug_fixes = suite.tasks_by_category(BenchmarkTaskCategory.BUG_FIXING)
        assert len(bug_fixes) == 2


class TestBenchmarkRunner:
    def test_aggregate_results(self) -> None:
        runner = BenchmarkRunner()
        results = [
            BenchmarkResult(task_id="t1", strategy_id="s1", strategy_name="S", success=True),
            BenchmarkResult(task_id="t2", strategy_id="s1", strategy_name="S", success=True),
            BenchmarkResult(task_id="t3", strategy_id="s1", strategy_name="S", success=False),
        ]
        agg = runner.aggregate_results(results)
        assert agg.total_tasks == 3
        assert agg.successful_tasks == 2
        assert agg.primary_metric == pytest.approx(2 / 3)

    def test_compare_suite_results(self) -> None:
        runner = BenchmarkRunner()
        baseline = [
            BenchmarkResult(task_id="t1", strategy_name="A", success=True),
            BenchmarkResult(task_id="t2", strategy_name="A", success=False),
        ]
        candidate = [
            BenchmarkResult(task_id="t1", strategy_name="B", success=True),
            BenchmarkResult(task_id="t2", strategy_name="B", success=True),
        ]
        comparison = runner.compare_suite_results(baseline, candidate)
        assert comparison["improvement"] > 0
        assert "improved" in comparison["conclusion"].lower()


# ---------------------------------------------------------------------------
# ResearchPrioritizer tests
# ---------------------------------------------------------------------------


class TestResearchPrioritizer:
    def test_score_problem_high_priority(self) -> None:
        prioritizer = ResearchPrioritizer()
        problem = ResearchProblem(
            title="Critical issue",
            priority=ResearchProblemPriority.CRITICAL,
            decomposition=ProblemDecomposition(
                known_facts=("fact1",),
                unknowns=("unknown1", "unknown2"),
                measurable_quantities=("metric1",),
            ),
        )
        score = prioritizer.score_problem(problem)
        assert score > 0.5

    def test_score_problem_low_priority(self) -> None:
        prioritizer = ResearchPrioritizer()
        problem = ResearchProblem(
            title="Minor issue",
            priority=ResearchProblemPriority.LOW,
        )
        score = prioritizer.score_problem(problem)
        assert score < 0.5

    def test_rank_problems(self) -> None:
        prioritizer = ResearchPrioritizer()
        problems = [
            ResearchProblem(title="Low", priority=ResearchProblemPriority.LOW),
            ResearchProblem(title="High", priority=ResearchProblemPriority.HIGH),
            ResearchProblem(title="Critical", priority=ResearchProblemPriority.CRITICAL),
        ]
        ranked = prioritizer.rank_problems(problems)
        assert len(ranked) == 3
        assert ranked[0][0].priority == ResearchProblemPriority.CRITICAL
        assert ranked[1][0].priority == ResearchProblemPriority.HIGH

    def test_select_research_targets(self) -> None:
        prioritizer = ResearchPrioritizer()
        problems = [
            ResearchProblem(title=f"P{i}", priority=ResearchProblemPriority.MEDIUM)
            for i in range(10)
        ]
        targets = prioritizer.select_research_targets(problems, budget=3)
        assert len(targets) <= 3

    def test_to_dict(self) -> None:
        prioritizer = ResearchPrioritizer()
        d = prioritizer.to_dict()
        assert "weights" in d
        assert sum(d["weights"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# StrategyComposer tests
# ---------------------------------------------------------------------------


class TestStrategyComposer:
    def test_compose_two_strategies(self) -> None:
        composer = StrategyComposer()
        s1 = EngineeringStrategy(
            name="dependency_planning",
            approach="Plan with dependency awareness.",
            tags=("dependency",),
        )
        s2 = EngineeringStrategy(
            name="failure_replanning",
            approach="Replan on failure.",
            tags=("recovery",),
        )

        plan, composite = composer.compose([s1, s2])

        assert plan.plan_id
        assert len(plan.component_strategy_ids) == 2
        assert composite.name
        assert composite.parent_strategy_id is None
        assert "dependency_planning" in composite.metadata["component_names"]

    def test_compose_requires_two(self) -> None:
        composer = StrategyComposer()
        s1 = EngineeringStrategy(name="alone")
        with pytest.raises(ValueError, match="At least 2"):
            composer.compose([s1])

    def test_suggest_compositions(self) -> None:
        composer = StrategyComposer()
        strategies = [
            EngineeringStrategy(
                name=f"strategy_{i}",
                approach=f"Approach {i}",
                problem_class=ProblemClass.TEST_FAILURE,
            )
            for i in range(3)
        ]
        # Give them success history
        for s in strategies:
            for _ in range(5):
                s.record_use(StrategyOutcome.SUCCESS)

        plans = composer.suggest_compositions(strategies)
        assert len(plans) > 0

    def test_suggest_compositions_filters_deprecated(self) -> None:
        composer = StrategyComposer()
        s1 = EngineeringStrategy(name="good")
        for _ in range(5):
            s1.record_use(StrategyOutcome.SUCCESS)

        s2 = EngineeringStrategy(name="deprecated")
        for _ in range(5):
            s2.record_use(StrategyOutcome.FAILURE)

        plans = composer.suggest_compositions([s1, s2])
        assert len(plans) == 0


# ---------------------------------------------------------------------------
# NegativeKnowledgeStore tests
# ---------------------------------------------------------------------------


class TestNegativeKnowledgeStore:
    def test_store_and_retrieve(self) -> None:
        store = NegativeKnowledgeStore()
        nk = NegativeKnowledge(
            strategy_description="Mass upgrade",
            problem_class="dependency_conflict",
            result="Failures",
            conclusion="Bad approach",
        )
        store.store(nk)
        assert len(store) == 1
        assert nk.knowledge_id in store

    def test_check_approach_found(self) -> None:
        store = NegativeKnowledgeStore()
        store.record_failure(
            strategy_description="Mass dependency upgrade",
            problem_class="dependency_conflict",
            result="Increased failures",
            conclusion="Bad for localized conflicts",
        )

        result = store.check_approach("Mass dependency upgrade", "dependency_conflict")
        assert result is not None
        assert "Bad" in result.conclusion

    def test_check_approach_not_found(self) -> None:
        store = NegativeKnowledgeStore()
        result = store.check_approach("Unknown approach")
        assert result is None

    def test_record_failure(self) -> None:
        store = NegativeKnowledgeStore()
        nk = store.record_failure(
            strategy_description="Test",
            problem_class="test",
            result="Failed",
            conclusion="Bad",
        )
        assert nk.knowledge_id in store

    def test_stats(self) -> None:
        store = NegativeKnowledgeStore()
        store.record_failure(
            strategy_description="A", problem_class="dep", result="F", conclusion="C"
        )
        store.record_failure(
            strategy_description="B", problem_class="dep", result="F", conclusion="C"
        )
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["by_problem_class"]["dep"] == 2

    def test_eviction(self) -> None:
        store = NegativeKnowledgeStore(max_entries=2)
        store.record_failure(
            strategy_description="A",
            problem_class="x",
            result="F",
            conclusion="C",
            confidence=0.3,
        )
        store.record_failure(
            strategy_description="B",
            problem_class="x",
            result="F",
            conclusion="C",
            confidence=0.9,
        )
        assert len(store) == 2

        store.record_failure(
            strategy_description="C",
            problem_class="x",
            result="F",
            conclusion="C",
            confidence=0.5,
        )
        assert len(store) == 2
        # The lowest confidence (0.3) should have been evicted
        remaining = store.all_entries()
        assert all(n.confidence >= 0.3 for n in remaining)


# ---------------------------------------------------------------------------
# StrategyVersion integration tests
# ---------------------------------------------------------------------------


class TestStrategyVersioning:
    def test_strategy_has_version_fields(self) -> None:
        s = EngineeringStrategy(name="test", version_number=1)
        assert s.version_number == 1
        assert s.parent_strategy_id is None
        assert s.version_history == ()

    def test_version_in_to_dict(self) -> None:
        s = EngineeringStrategy(
            name="test",
            version_number=2,
            parent_strategy_id="parent1",
            version_history=("v1", "v2"),
        )
        d = s.to_dict()
        assert d["version_number"] == 2
        assert d["parent_strategy_id"] == "parent1"
        assert d["version_history"] == ["v1", "v2"]
