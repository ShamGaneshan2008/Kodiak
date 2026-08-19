"""Tests for Phase 6 advanced components: meta-strategies, memory quality,
capability composition, research-evolution bridge, resource-aware intelligence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kodiak.orchestration.evolution.capability import (
    Capability,
    CapabilityCategory,
    CapabilityPerformance,
    CapabilityTracker,
)
from kodiak.orchestration.evolution.capability_composer import (
    CapabilityComposer,
    CompositionResult,
)
from kodiak.orchestration.evolution.improvement_queue import (
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


# ======================================================================
# MetaStrategySelector tests
# ======================================================================


class TestMetaStrategySelector:
    def test_default_profiles_loaded(self) -> None:
        selector = MetaStrategySelector()
        profiles = selector.all_profiles()
        assert len(profiles) >= 5

    def test_select_simple_known(self) -> None:
        selector = MetaStrategySelector()
        decision = selector.select_method(
            task_id="t1",
            complexity=TaskComplexity.SIMPLE,
            risk_level=RiskLevel.LOW,
            has_known_strategies=True,
            strategy_confidence=0.8,
        )
        assert decision.selection_method in {
            StrategySelectionMethod.HISTORICAL_RETRIEVAL,
            StrategySelectionMethod.DIRECT_APPLICATION,
        }
        assert decision.confidence > 0.3

    def test_select_novel_no_strategies(self) -> None:
        selector = MetaStrategySelector()
        decision = selector.select_method(
            task_id="t2",
            complexity=TaskComplexity.NOVEL,
            risk_level=RiskLevel.HIGH,
            has_known_strategies=False,
            strategy_confidence=0.2,
        )
        assert decision.selection_method == StrategySelectionMethod.RESEARCH_DRIVEN

    def test_select_complex_high_risk(self) -> None:
        selector = MetaStrategySelector()
        decision = selector.select_method(
            task_id="t3",
            complexity=TaskComplexity.COMPLEX,
            risk_level=RiskLevel.CRITICAL,
            has_known_strategies=True,
            strategy_confidence=0.4,
        )
        assert decision.selection_method in {
            StrategySelectionMethod.MULTI_STRATEGY_EXPERIMENT,
            StrategySelectionMethod.BENCHMARK_RANKING,
        }

    def test_select_trivial(self) -> None:
        selector = MetaStrategySelector()
        decision = selector.select_method(
            task_id="t4",
            complexity=TaskComplexity.TRIVIAL,
            risk_level=RiskLevel.LOW,
            has_known_strategies=True,
            strategy_confidence=0.9,
        )
        assert decision.selection_method == StrategySelectionMethod.DIRECT_APPLICATION

    def test_record_outcome(self) -> None:
        selector = MetaStrategySelector()
        selector.record_outcome(StrategySelectionMethod.HISTORICAL_RETRIEVAL, True)
        selector.record_outcome(StrategySelectionMethod.HISTORICAL_RETRIEVAL, True)
        selector.record_outcome(StrategySelectionMethod.HISTORICAL_RETRIEVAL, False)

        rate = selector.method_success_rate(StrategySelectionMethod.HISTORICAL_RETRIEVAL)
        assert rate == pytest.approx(2 / 3)

    def test_method_report(self) -> None:
        selector = MetaStrategySelector()
        selector.record_outcome(StrategySelectionMethod.BENCHMARK_RANKING, True)
        report = selector.method_report()
        assert "benchmark_ranking" in report
        assert report["benchmark_ranking"]["total_uses"] == 1

    def test_register_custom_profile(self) -> None:
        selector = MetaStrategySelector()
        custom = MetaStrategyProfile(
            name="custom_profile",
            task_complexity=TaskComplexity.COMPLEX,
            selection_method=StrategySelectionMethod.COMPOSITION_BASED,
        )
        selector.register_profile(custom)
        assert any(p.name == "custom_profile" for p in selector.all_profiles())

    def test_recent_decisions(self) -> None:
        selector = MetaStrategySelector()
        for i in range(5):
            selector.select_method(
                task_id=f"t{i}",
                complexity=TaskComplexity.SIMPLE,
                risk_level=RiskLevel.LOW,
            )
        recent = selector.recent_decisions(limit=3)
        assert len(recent) == 3

    def test_decision_has_reasoning(self) -> None:
        selector = MetaStrategySelector()
        decision = selector.select_method(
            task_id="t1",
            complexity=TaskComplexity.MODERATE,
            risk_level=RiskLevel.MEDIUM,
        )
        assert len(decision.reasoning) > 0

    def test_alternatives(self) -> None:
        selector = MetaStrategySelector()
        decision = selector.select_method(
            task_id="t1",
            complexity=TaskComplexity.COMPLEX,
            risk_level=RiskLevel.HIGH,
            has_known_strategies=True,
            strategy_confidence=0.6,
        )
        # May or may not have alternatives depending on profile matching
        assert isinstance(decision.alternatives, tuple)


class TestMetaStrategyProfile:
    def test_creation(self) -> None:
        profile = MetaStrategyProfile(
            name="test",
            task_complexity=TaskComplexity.SIMPLE,
            selection_method=StrategySelectionMethod.HISTORICAL_RETRIEVAL,
        )
        assert profile.name == "test"

    def test_to_dict(self) -> None:
        profile = MetaStrategyProfile(name="test")
        d = profile.to_dict()
        assert d["name"] == "test"
        assert "profile_id" in d


class TestMetaStrategyDecision:
    def test_to_dict(self) -> None:
        decision = MetaStrategyDecision(
            task_id="t1",
            selection_method=StrategySelectionMethod.BENCHMARK_RANKING,
            reasoning="Test reasoning",
            confidence=0.7,
        )
        d = decision.to_dict()
        assert d["task_id"] == "t1"
        assert d["selection_method"] == "benchmark_ranking"


# ======================================================================
# MemoryQualityController tests
# ======================================================================


class TestMemoryEntry:
    def test_creation(self) -> None:
        entry = MemoryEntry(
            content="Strategy A works well",
            memory_type="strategy",
            confidence=0.8,
            importance=0.7,
        )
        assert entry.confidence == 0.8

    def test_quality_score(self) -> None:
        entry = MemoryEntry(
            confidence=0.9,
            importance=0.8,
            evidence_strength=0.7,
            usage_count=5,
        )
        score = entry.quality_score
        assert score > 0.5

    def test_freshness(self) -> None:
        entry = MemoryEntry()
        assert entry.freshness > 0.9  # Very fresh

    def test_to_dict(self) -> None:
        entry = MemoryEntry(content="test", confidence=0.5)
        d = entry.to_dict()
        assert d["content"] == "test"
        assert "quality_score" in d


class TestMemoryQualityController:
    def test_add_and_get(self) -> None:
        controller = MemoryQualityController()
        entry = MemoryEntry(content="test", confidence=0.5)
        controller.add_entry(entry)
        assert controller.get_entry(entry.entry_id) is not None

    def test_access_entry(self) -> None:
        controller = MemoryQualityController()
        entry = MemoryEntry(content="test", usage_count=0)
        controller.add_entry(entry)
        accessed = controller.access_entry(entry.entry_id)
        assert accessed is not None
        assert accessed.usage_count == 1

    def test_detect_contradictions(self) -> None:
        controller = MemoryQualityController()
        a = MemoryEntry(
            content="Strategy A is successful and effective",
            memory_type="strategy",
            strategy_id="s1",
            tags=("test",),
        )
        b = MemoryEntry(
            content="Strategy A is broken and unreliable",
            memory_type="strategy",
            strategy_id="s1",
            tags=("test",),
        )
        controller.add_entry(a)
        controller.add_entry(b)

        contradictions = controller.detect_contradictions()
        assert len(contradictions) > 0
        assert contradictions[0].entry_a_id == a.entry_id

    def test_find_duplicates(self) -> None:
        controller = MemoryQualityController()
        content = "The test failure strategy involves inspecting output and fixing"
        a = MemoryEntry(content=content, memory_type="strategy")
        b = MemoryEntry(content=content, memory_type="strategy")
        controller.add_entry(a)
        controller.add_entry(b)

        duplicates = controller.find_duplicates()
        assert len(duplicates) == 1

    def test_find_stale(self) -> None:
        controller = MemoryQualityController(freshness_threshold=0.5)
        entry = MemoryEntry(
            content="old",
            last_accessed=datetime(2020, 1, 1, tzinfo=UTC),
        )
        controller.add_entry(entry)
        stale = controller.find_stale_entries()
        assert len(stale) == 1

    def test_find_low_quality(self) -> None:
        controller = MemoryQualityController(confidence_threshold=0.5)
        entry = MemoryEntry(confidence=0.1, importance=0.1, evidence_strength=0.1)
        controller.add_entry(entry)
        low = controller.find_low_quality()
        assert len(low) == 1

    def test_quality_report(self) -> None:
        controller = MemoryQualityController()
        controller.add_entry(MemoryEntry(confidence=0.8, importance=0.7))
        controller.add_entry(MemoryEntry(confidence=0.3, importance=0.2))

        report = controller.compute_quality_report()
        assert report.total_entries == 2
        assert report.avg_confidence > 0

    def test_prune_low_quality(self) -> None:
        controller = MemoryQualityController(confidence_threshold=0.5)
        for i in range(5):
            controller.add_entry(MemoryEntry(confidence=0.1, importance=0.1))
        controller.add_entry(MemoryEntry(confidence=0.9, importance=0.9))

        pruned = controller.prune_low_quality(max_prune=3)
        assert len(pruned) == 3
        assert len(controller.all_entries()) == 3

    def test_quality_report_with_recommendations(self) -> None:
        controller = MemoryQualityController(confidence_threshold=0.5)
        # Add many low-quality entries
        for i in range(10):
            controller.add_entry(MemoryEntry(confidence=0.1, importance=0.1))

        report = controller.compute_quality_report()
        assert len(report.recommendations) > 0

    def test_no_contradictions_when_similar_sentiment(self) -> None:
        controller = MemoryQualityController()
        a = MemoryEntry(
            content="Strategy A is successful and effective",
            strategy_id="s1",
        )
        b = MemoryEntry(
            content="Strategy A works well and is good",
            strategy_id="s1",
        )
        controller.add_entry(a)
        controller.add_entry(b)

        contradictions = controller.detect_contradictions()
        assert len(contradictions) == 0


class TestQualityReport:
    def test_to_dict(self) -> None:
        report = QualityReport(
            total_entries=10,
            avg_confidence=0.7,
            contradiction_count=2,
            recommendations=("Clean up",),
        )
        d = report.to_dict()
        assert d["total_entries"] == 10
        assert d["contradiction_count"] == 2


class TestContradiction:
    def test_to_dict(self) -> None:
        c = Contradiction(
            entry_a_id="a",
            entry_b_id="b",
            explanation="Opposite sentiments",
            possible_reasons=("Different contexts",),
        )
        d = c.to_dict()
        assert d["entry_a_id"] == "a"


# ======================================================================
# CapabilityComposer tests
# ======================================================================


class TestCapabilityComposer:
    def _make_tracker(self) -> CapabilityTracker:
        tracker = CapabilityTracker()
        tracker.register(Capability(
            name="planning",
            category=CapabilityCategory.PLANNING,
            performance=CapabilityPerformance(total_attempts=10, successful_attempts=8),
            evidence=("Evidence 1",),
        ))
        tracker.register(Capability(
            name="testing",
            category=CapabilityCategory.TESTING,
            performance=CapabilityPerformance(total_attempts=10, successful_attempts=9),
            evidence=("Evidence 2", "Evidence 3"),
        ))
        tracker.register(Capability(
            name="debugging",
            category=CapabilityCategory.DEBUGGING,
            performance=CapabilityPerformance(total_attempts=5, successful_attempts=3),
        ))
        return tracker

    def test_compose_two_capabilities(self) -> None:
        tracker = self._make_tracker()
        composer = CapabilityComposer(tracker)

        caps = tracker.all_capabilities()
        composite = composer.compose(
            (caps[0].capability_id, caps[1].capability_id),
            name="plan_and_test",
        )
        assert composite is not None
        assert composite.name == "plan_and_test"
        assert composite.metadata.get("composition") is True

    def test_compose_requires_two(self) -> None:
        tracker = self._make_tracker()
        composer = CapabilityComposer(tracker)
        caps = tracker.all_capabilities()
        result = composer.compose((caps[0].capability_id,))
        assert result is None

    def test_evaluate_composition(self) -> None:
        tracker = self._make_tracker()
        composer = CapabilityComposer(tracker)
        caps = tracker.all_capabilities()

        composite = composer.compose(
            (caps[0].capability_id, caps[1].capability_id),
            name="test_composition",
        )
        assert composite is not None

        result = composer.evaluate_composition(
            composite,
            test_results={"task1": True, "task2": True, "task3": False},
        )
        assert result.is_improvement is not None
        assert result.composite_score == pytest.approx(2 / 3)

    def test_suggest_compositions(self) -> None:
        tracker = self._make_tracker()
        composer = CapabilityComposer(tracker)
        suggestions = composer.suggest_compositions()
        assert len(suggestions) > 0

    def test_all_results(self) -> None:
        tracker = self._make_tracker()
        composer = CapabilityComposer(tracker)
        caps = tracker.all_capabilities()

        composite = composer.compose(
            (caps[0].capability_id, caps[1].capability_id),
        )
        composer.evaluate_composition(composite, test_results={"t": True})
        results = composer.all_results()
        assert len(results) == 1

    def test_to_dict(self) -> None:
        tracker = self._make_tracker()
        composer = CapabilityComposer(tracker)
        d = composer.to_dict()
        assert "compositions" in d


class TestCompositionResult:
    def test_to_dict(self) -> None:
        r = CompositionResult(
            composition_name="test",
            composite_score=0.8,
            component_avg_score=0.6,
            improvement=0.2,
            is_improvement=True,
        )
        d = r.to_dict()
        assert d["is_improvement"] is True
        assert d["improvement"] == 0.2


# ======================================================================
# ResearchEvolutionBridge tests
# ======================================================================


class TestResearchEvolutionBridge:
    def _make_bridge(self) -> ResearchEvolutionBridge:
        queue = ImprovementQueue()
        tracker = CapabilityTracker()
        return ResearchEvolutionBridge(
            improvement_queue=queue,
            capability_tracker=tracker,
            min_confidence=0.5,
        )

    def test_bridge_creates_proposal(self) -> None:
        bridge = self._make_bridge()
        discovery = ResearchDiscovery(
            title="Better test fix strategy",
            description="A more effective approach to test failures",
            strategy_name="enhanced_test_fix",
            problem_class="test_failure",
            confidence=0.8,
            evidence=("Experiment showed 20% improvement",),
        )

        result = bridge.bridge_discovery(discovery)
        assert result.action_taken == "proposal_created"
        assert result.proposal_id != ""

    def test_bridge_skips_low_confidence(self) -> None:
        bridge = self._make_bridge()
        discovery = ResearchDiscovery(
            title="Uncertain discovery",
            confidence=0.3,
        )
        result = bridge.bridge_discovery(discovery)
        assert result.action_taken == "skipped"

    def test_bridge_updates_existing_capability(self) -> None:
        bridge = self._make_bridge()
        # Register existing capability
        cap = Capability(
            name="existing_strategy",
            category=CapabilityCategory.TESTING,
            evidence=("Original evidence",),
        )
        bridge._tracker.register(cap)

        discovery = ResearchDiscovery(
            title="Update for existing",
            strategy_name="existing_strategy",
            confidence=0.8,
            evidence=("New evidence",),
        )
        result = bridge.bridge_discovery(discovery)
        assert result.action_taken == "capability_updated"

    def test_bridge_multiple(self) -> None:
        bridge = self._make_bridge()
        discoveries = [
            ResearchDiscovery(title=f"Discovery {i}", confidence=0.7)
            for i in range(3)
        ]
        results = bridge.bridge_multiple(discoveries)
        assert len(results) == 3

    def test_register_validated_capability(self) -> None:
        bridge = self._make_bridge()
        discovery = ResearchDiscovery(
            title="Validated strategy",
            strategy_name="validated_strategy",
            description="A validated approach",
            problem_class="test_failure",
            confidence=0.9,
            evidence=("Validated by experiment",),
        )
        cap = bridge.register_discovered_capability(discovery, validated=True)
        assert cap is not None
        assert cap.name == "validated_strategy"

    def test_register_unvalidated_returns_none(self) -> None:
        bridge = self._make_bridge()
        discovery = ResearchDiscovery(
            title="Unvalidated",
            strategy_name="unvalidated_strategy",
        )
        cap = bridge.register_discovered_capability(discovery, validated=False)
        assert cap is None

    def test_stats(self) -> None:
        bridge = self._make_bridge()
        bridge.bridge_discovery(ResearchDiscovery(title="D1", confidence=0.8))
        bridge.bridge_discovery(ResearchDiscovery(title="D2", confidence=0.2))
        stats = bridge.stats()
        assert stats["total_bridged"] == 2

    def test_all_results(self) -> None:
        bridge = self._make_bridge()
        bridge.bridge_discovery(ResearchDiscovery(title="D1", confidence=0.8))
        results = bridge.all_results()
        assert len(results) == 1


class TestResearchDiscovery:
    def test_to_dict(self) -> None:
        d = ResearchDiscovery(
            title="Test discovery",
            strategy_name="test_strategy",
            confidence=0.7,
        ).to_dict()
        assert d["title"] == "Test discovery"
        assert d["confidence"] == 0.7


# ======================================================================
# ResourceAwareEngine tests
# ======================================================================


class TestResourceAwareEngine:
    def test_assess_simple_task(self) -> None:
        engine = ResourceAwareEngine()
        assessment = engine.assess_task(
            task_id="t1",
            goal="Fix typo in docstring",
            has_known_strategies=True,
            strategy_confidence=0.9,
        )
        assert assessment.reasoning_depth in {
            ReasoningDepth.MINIMAL,
            ReasoningDepth.LIGHT,
        }

    def test_assess_complex_task(self) -> None:
        engine = ResourceAwareEngine()
        assessment = engine.assess_task(
            task_id="t2",
            goal="Implement a complex multi-module architecture refactor with dependency injection, event-driven communication, and comprehensive test coverage",
            has_known_strategies=False,
            strategy_confidence=0.2,
            previous_failures=3,
            requires_research=True,
            is_high_risk=True,
        )
        assert assessment.reasoning_depth in {
            ReasoningDepth.DEEP,
            ReasoningDepth.EXHAUSTIVE,
        }
        assert assessment.risk_score > 0.5

    def test_assess_recorded(self) -> None:
        engine = ResourceAwareEngine()
        engine.assess_task(task_id="t1", goal="Simple task")
        recent = engine.recent_assessments()
        assert len(recent) == 1

    def test_get_profile(self) -> None:
        engine = ResourceAwareEngine()
        profile = engine.get_profile(ReasoningDepth.DEEP)
        assert profile.reasoning_depth == ReasoningDepth.DEEP
        assert profile.enable_experimentation is True

    def test_profile_for_assessment(self) -> None:
        engine = ResourceAwareEngine()
        assessment = engine.assess_task(
            task_id="t1",
            goal="Complex multi-file refactor with no known strategies",
            has_known_strategies=False,
            strategy_confidence=0.2,
            is_high_risk=True,
        )
        profile = engine.profile_for_assessment(assessment)
        assert profile is not None

    def test_record_usage(self) -> None:
        engine = ResourceAwareEngine()
        engine.record_usage(
            task_id="t1",
            depth=ReasoningDepth.STANDARD,
            duration_seconds=15.0,
            success=True,
        )
        report = engine.usage_report()
        assert report["total"] == 1

    def test_usage_report(self) -> None:
        engine = ResourceAwareEngine()
        for i in range(5):
            engine.record_usage(
                task_id=f"t{i}",
                depth=ReasoningDepth.STANDARD,
                duration_seconds=10.0 + i,
                success=i < 4,
            )
        report = engine.usage_report()
        assert "standard" in report["by_depth"]
        assert report["by_depth"]["standard"]["count"] == 5

    def test_suggest_depth_adjustment_down(self) -> None:
        engine = ResourceAwareEngine()
        # Simulate many successful, fast tasks
        for i in range(10):
            engine.record_usage(
                task_id=f"t{i}",
                depth=ReasoningDepth.STANDARD,
                duration_seconds=2.0,
                success=True,
            )
        report = engine.usage_report()
        suggestion = engine.suggest_depth_adjustment(ReasoningDepth.STANDARD, report)
        # Should suggest going lighter
        assert suggestion in {ReasoningDepth.LIGHT, None}

    def test_register_custom_profile(self) -> None:
        engine = ResourceAwareEngine()
        custom = ResourceProfile(
            name="custom",
            reasoning_depth=ReasoningDepth.DEEP,
            max_retries=10,
        )
        engine.register_profile(custom)
        profile = engine.get_profile(ReasoningDepth.DEEP)
        assert profile.max_retries == 10

    def test_all_profiles(self) -> None:
        engine = ResourceAwareEngine()
        profiles = engine.all_profiles()
        assert len(profiles) >= 5

    def test_assessment_factors(self) -> None:
        engine = ResourceAwareEngine()
        assessment = engine.assess_task(
            task_id="t1",
            goal="Short",
            has_known_strategies=False,
            previous_failures=2,
            is_high_risk=True,
        )
        assert len(assessment.factors) > 0

    def test_assessment_to_dict(self) -> None:
        engine = ResourceAwareEngine()
        assessment = engine.assess_task(task_id="t1", goal="Test task")
        d = assessment.to_dict()
        assert d["task_id"] == "t1"
        assert "complexity_score" in d


class TestResourceProfile:
    def test_to_dict(self) -> None:
        profile = ResourceProfile(
            name="test",
            reasoning_depth=ReasoningDepth.DEEP,
            max_retries=5,
        )
        d = profile.to_dict()
        assert d["name"] == "test"
        assert d["max_retries"] == 5


class TestTaskComplexityAssessment:
    def test_to_dict(self) -> None:
        a = TaskComplexityAssessment(
            task_id="t1",
            goal="Test",
            complexity_score=0.7,
            reasoning_depth=ReasoningDepth.DEEP,
        )
        d = a.to_dict()
        assert d["task_id"] == "t1"
