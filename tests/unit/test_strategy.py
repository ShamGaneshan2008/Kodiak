"""Tests for the Engineering Strategy model and StrategyMemory."""

from __future__ import annotations

from kodiak.orchestration.strategy import (
    EngineeringStrategy,
    ProblemClass,
    StrategyComparison,
    StrategyMemory,
    StrategyOutcome,
    default_strategies,
)

# ---------------------------------------------------------------------------
# EngineeringStrategy model tests
# ---------------------------------------------------------------------------


class TestEngineeringStrategy:
    def test_strategy_creation(self) -> None:
        strategy = EngineeringStrategy(
            name="test_strategy",
            problem_class=ProblemClass.TEST_FAILURE,
            approach="Run tests, inspect failures, fix.",
        )
        assert strategy.name == "test_strategy"
        assert strategy.problem_class == ProblemClass.TEST_FAILURE
        assert strategy.use_count == 0
        assert strategy.success_count == 0
        assert strategy.failure_count == 0

    def test_strategy_id_is_unique(self) -> None:
        s1 = EngineeringStrategy(name="a")
        s2 = EngineeringStrategy(name="b")
        assert s1.strategy_id != s2.strategy_id

    def test_success_rate_with_no_uses(self) -> None:
        strategy = EngineeringStrategy(name="test")
        assert strategy.success_rate == 0.5  # default when no history

    def test_success_rate_after_uses(self) -> None:
        strategy = EngineeringStrategy(name="test")
        strategy.record_use(StrategyOutcome.SUCCESS)
        strategy.record_use(StrategyOutcome.SUCCESS)
        strategy.record_use(StrategyOutcome.FAILURE)
        assert strategy.success_count == 2
        assert strategy.failure_count == 1
        assert abs(strategy.success_rate - 2 / 3) < 0.01

    def test_record_use_success(self) -> None:
        strategy = EngineeringStrategy(name="test")
        strategy.record_use(StrategyOutcome.SUCCESS)
        assert strategy.use_count == 1
        assert strategy.success_count == 1
        assert strategy.failure_count == 0
        assert strategy.last_used_at is not None

    def test_record_use_failure(self) -> None:
        strategy = EngineeringStrategy(name="test")
        strategy.record_use(StrategyOutcome.FAILURE)
        assert strategy.use_count == 1
        assert strategy.success_count == 0
        assert strategy.failure_count == 1

    def test_record_use_partial(self) -> None:
        strategy = EngineeringStrategy(name="test")
        strategy.record_use(StrategyOutcome.PARTIAL)
        assert strategy.use_count == 1
        assert strategy.success_count == 0
        assert strategy.failure_count == 0  # partial doesn't count as failure

    def test_is_reliable(self) -> None:
        strategy = EngineeringStrategy(name="test")
        # Not reliable with fewer than 2 uses
        strategy.record_use(StrategyOutcome.SUCCESS)
        assert not strategy.is_reliable

        strategy.record_use(StrategyOutcome.SUCCESS)
        assert strategy.is_reliable  # 2/2 = 1.0 >= 0.6

    def test_is_reliable_with_mixed_results(self) -> None:
        strategy = EngineeringStrategy(name="test")
        for _ in range(4):
            strategy.record_use(StrategyOutcome.SUCCESS)
        strategy.record_use(StrategyOutcome.FAILURE)
        assert strategy.is_reliable  # 4/5 = 0.8 >= 0.6

    def test_is_not_reliable_when_poor(self) -> None:
        strategy = EngineeringStrategy(name="test")
        for _ in range(2):
            strategy.record_use(StrategyOutcome.FAILURE)
        assert not strategy.is_reliable  # 0/2 = 0.0 < 0.6

    def test_is_deprecated(self) -> None:
        strategy = EngineeringStrategy(name="test")
        for _ in range(3):
            strategy.record_use(StrategyOutcome.FAILURE)
        assert strategy.is_deprecated  # 0/3 = 0.0 < 0.4, total >= 3

    def test_is_not_deprecated_with_few_uses(self) -> None:
        strategy = EngineeringStrategy(name="test")
        strategy.record_use(StrategyOutcome.FAILURE)
        strategy.record_use(StrategyOutcome.FAILURE)
        assert not strategy.is_deprecated  # only 2 uses, threshold is 3

    def test_effectiveness_score_no_history(self) -> None:
        strategy = EngineeringStrategy(name="test", expected_success_probability=0.8)
        assert abs(strategy.effectiveness_score() - 0.8) < 0.01

    def test_effectiveness_score_with_history(self) -> None:
        strategy = EngineeringStrategy(name="test", expected_success_probability=0.5)
        for _ in range(10):
            strategy.record_use(StrategyOutcome.SUCCESS)
        # After 10 uses, weight=1.0, so score = success_rate = 1.0
        assert strategy.effectiveness_score() >= 0.9

    def test_to_dict(self) -> None:
        strategy = EngineeringStrategy(
            name="test",
            problem_class=ProblemClass.TEST_FAILURE,
            approach="Run tests.",
            tags=("test", "fix"),
        )
        d = strategy.to_dict()
        assert d["name"] == "test"
        assert d["problem_class"] == "test_failure"
        assert d["tags"] == ["test", "fix"]
        assert "strategy_id" in d
        assert "effectiveness_score" in d


# ---------------------------------------------------------------------------
# StrategyMemory tests
# ---------------------------------------------------------------------------


class TestStrategyMemory:
    def test_store_and_retrieve(self) -> None:
        memory = StrategyMemory()
        strategy = EngineeringStrategy(
            name="test",
            problem_class=ProblemClass.TEST_FAILURE,
            approach="Run tests.",
        )
        memory.store(strategy)
        assert strategy.strategy_id in memory
        assert len(memory) == 1

        retrieved = memory.get(strategy.strategy_id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_retrieve_nonexistent(self) -> None:
        memory = StrategyMemory()
        assert memory.get("nonexistent") is None

    def test_remove(self) -> None:
        memory = StrategyMemory()
        strategy = EngineeringStrategy(name="test", problem_class=ProblemClass.TIMEOUT)
        memory.store(strategy)
        assert memory.remove(strategy.strategy_id)
        assert strategy.strategy_id not in memory
        assert not memory.remove("nonexistent")

    def test_retrieve_for_problem(self) -> None:
        memory = StrategyMemory()
        s1 = EngineeringStrategy(name="fix_a", problem_class=ProblemClass.TEST_FAILURE)
        s2 = EngineeringStrategy(name="fix_b", problem_class=ProblemClass.TEST_FAILURE)
        s3 = EngineeringStrategy(name="fix_c", problem_class=ProblemClass.TIMEOUT)
        memory.store(s1)
        memory.store(s2)
        memory.store(s3)

        results = memory.retrieve_for_problem(ProblemClass.TEST_FAILURE)
        assert len(results) == 2
        assert all(s.problem_class == ProblemClass.TEST_FAILURE for s in results)

    def test_retrieve_for_problem_with_tags(self) -> None:
        memory = StrategyMemory()
        s1 = EngineeringStrategy(
            name="fix_a",
            problem_class=ProblemClass.TEST_FAILURE,
            tags=("regression",),
        )
        s2 = EngineeringStrategy(
            name="fix_b",
            problem_class=ProblemClass.TEST_FAILURE,
            tags=("unit_test",),
        )
        memory.store(s1)
        memory.store(s2)

        results = memory.retrieve_for_problem(ProblemClass.TEST_FAILURE, tags=("regression",))
        assert len(results) == 1
        assert results[0].name == "fix_a"

    def test_retrieve_for_problem_respects_limit(self) -> None:
        memory = StrategyMemory()
        for i in range(10):
            memory.store(
                EngineeringStrategy(
                    name=f"fix_{i}",
                    problem_class=ProblemClass.TEST_FAILURE,
                )
            )
        results = memory.retrieve_for_problem(ProblemClass.TEST_FAILURE, limit=3)
        assert len(results) == 3

    def test_retrieve_for_problem_prefers_reliable(self) -> None:
        memory = StrategyMemory()
        poor = EngineeringStrategy(name="poor", problem_class=ProblemClass.TEST_FAILURE)
        for _ in range(3):
            poor.record_use(StrategyOutcome.FAILURE)

        good = EngineeringStrategy(name="good", problem_class=ProblemClass.TEST_FAILURE)
        for _ in range(5):
            good.record_use(StrategyOutcome.SUCCESS)

        memory.store(poor)
        memory.store(good)

        results = memory.retrieve_for_problem(ProblemClass.TEST_FAILURE)
        assert len(results) == 1
        assert results[0].name == "good"  # reliable preferred

    def test_retrieve_for_problem_falls_back_to_deprecated(self) -> None:
        memory = StrategyMemory()
        deprecated = EngineeringStrategy(name="deprecated", problem_class=ProblemClass.TEST_FAILURE)
        for _ in range(5):
            deprecated.record_use(StrategyOutcome.FAILURE)
        memory.store(deprecated)

        results = memory.retrieve_for_problem(ProblemClass.TEST_FAILURE)
        assert len(results) == 1  # only deprecated available

    def test_compare_strategies(self) -> None:
        memory = StrategyMemory()
        s1 = EngineeringStrategy(name="A", problem_class=ProblemClass.TEST_FAILURE)
        s1.record_use(StrategyOutcome.SUCCESS)
        s1.record_use(StrategyOutcome.SUCCESS)

        s2 = EngineeringStrategy(name="B", problem_class=ProblemClass.TEST_FAILURE)
        s2.record_use(StrategyOutcome.SUCCESS)
        s2.record_use(StrategyOutcome.FAILURE)

        comparison = memory.compare_strategies([s1, s2])
        assert comparison is not None
        assert comparison.recommended.name == "A"
        assert comparison.recommended_index == 0
        assert len(comparison.scores) == 2

    def test_compare_empty_returns_none(self) -> None:
        memory = StrategyMemory()
        assert memory.compare_strategies([]) is None

    def test_record_outcome(self) -> None:
        memory = StrategyMemory()
        strategy = EngineeringStrategy(name="test", problem_class=ProblemClass.TIMEOUT)
        memory.store(strategy)

        memory.record_outcome(strategy.strategy_id, StrategyOutcome.SUCCESS)
        assert strategy.success_count == 1
        assert strategy.use_count == 1

        memory.record_outcome(strategy.strategy_id, StrategyOutcome.FAILURE)
        assert strategy.failure_count == 1
        assert strategy.use_count == 2

    def test_record_outcome_nonexistent(self) -> None:
        memory = StrategyMemory()
        result = memory.record_outcome("nonexistent", StrategyOutcome.SUCCESS)
        assert result is None

    def test_deprecate(self) -> None:
        memory = StrategyMemory()
        strategy = EngineeringStrategy(name="test", problem_class=ProblemClass.TIMEOUT)
        memory.store(strategy)
        assert memory.deprecate(strategy.strategy_id)
        assert strategy.is_deprecated

    def test_deprecate_nonexistent(self) -> None:
        memory = StrategyMemory()
        assert not memory.deprecate("nonexistent")

    def test_max_strategies_evicts_deprecated(self) -> None:
        memory = StrategyMemory(max_strategies=3)
        for i in range(3):
            s = EngineeringStrategy(name=f"s{i}", problem_class=ProblemClass.TEST_FAILURE)
            memory.store(s)

        # Make one deprecated
        deprecated = memory.get(list(memory._strategies.keys())[0])
        assert deprecated is not None
        for _ in range(5):
            deprecated.record_use(StrategyOutcome.FAILURE)

        # Adding a 4th should evict the deprecated one
        new = EngineeringStrategy(name="new", problem_class=ProblemClass.TIMEOUT)
        memory.store(new)
        assert len(memory) == 3
        assert deprecated.strategy_id not in memory

    def test_retrieve_by_tags(self) -> None:
        memory = StrategyMemory()
        s1 = EngineeringStrategy(name="a", problem_class=ProblemClass.TEST_FAILURE, tags=("fast",))
        s2 = EngineeringStrategy(
            name="b", problem_class=ProblemClass.TIMEOUT, tags=("safe", "fast")
        )
        memory.store(s1)
        memory.store(s2)

        results = memory.retrieve_by_tags(("fast",))
        assert len(results) == 2

        results = memory.retrieve_by_tags(("safe",))
        assert len(results) == 1
        assert results[0].name == "b"

    def test_all_strategies_sorted(self) -> None:
        memory = StrategyMemory()
        s1 = EngineeringStrategy(name="poor", problem_class=ProblemClass.TIMEOUT)
        s1.record_use(StrategyOutcome.FAILURE)
        s2 = EngineeringStrategy(name="good", problem_class=ProblemClass.TIMEOUT)
        s2.record_use(StrategyOutcome.SUCCESS)
        memory.store(s1)
        memory.store(s2)

        all_s = memory.all_strategies()
        assert all_s[0].name == "good"
        assert all_s[1].name == "poor"


# ---------------------------------------------------------------------------
# default_strategies tests
# ---------------------------------------------------------------------------


class TestDefaultStrategies:
    def test_default_strategies_not_empty(self) -> None:
        strategies = default_strategies()
        assert len(strategies) > 0

    def test_all_have_required_fields(self) -> None:
        for strategy in default_strategies():
            assert strategy.name
            assert strategy.problem_class != ProblemClass.UNKNOWN
            assert strategy.approach
            assert strategy.provenance == "builtin"

    def test_all_are_storable(self) -> None:
        memory = StrategyMemory()
        for strategy in default_strategies():
            memory.store(strategy)
        assert len(memory) == len(default_strategies())


# ---------------------------------------------------------------------------
# StrategyComparison tests
# ---------------------------------------------------------------------------


class TestStrategyComparison:
    def test_comparison_recommended(self) -> None:
        s1 = EngineeringStrategy(name="A", problem_class=ProblemClass.TEST_FAILURE)
        s2 = EngineeringStrategy(name="B", problem_class=ProblemClass.TEST_FAILURE)
        comparison = StrategyComparison(
            strategies=(s1, s2),
            scores=(0.8, 0.4),
            reasoning="A is better",
            recommended_index=0,
        )
        assert comparison.recommended.name == "A"

    def test_comparison_to_dict(self) -> None:
        s1 = EngineeringStrategy(name="A", problem_class=ProblemClass.TEST_FAILURE)
        comparison = StrategyComparison(
            strategies=(s1,),
            scores=(0.8,),
            reasoning="Only one option",
            recommended_index=0,
        )
        d = comparison.to_dict()
        assert len(d["strategies"]) == 1
        assert d["recommended_name"] == "A"
