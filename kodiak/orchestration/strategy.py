"""Engineering strategy representation and strategy memory.

This module introduces strategies as first-class objects in Kodiak's
autonomous loop.  A strategy captures *how* to solve a problem class,
not just *what* to do.  The ``StrategyMemory`` stores successful and
failed strategies so that future recovery decisions can be informed by
verified past experience rather than purely rule-based heuristics.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrategyOutcome(enum.StrEnum):
    """Outcome of applying a strategy."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ProblemClass(enum.StrEnum):
    """Broad categorization of engineering problems."""

    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    TYPE_ERROR = "type_error"
    LINT_FAILURE = "lint_failure"
    MISSING_DEPENDENCY = "missing_dependency"
    PERMISSION_FAILURE = "permission_failure"
    TIMEOUT = "timeout"
    INCORRECT_IMPLEMENTATION = "incorrect_implementation"
    MISSING_ARTIFACT = "missing_artifact"
    EXECUTION_FAILURE = "execution_failure"
    ARCHITECTURAL = "architectural"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# EngineeringStrategy
# ---------------------------------------------------------------------------


@dataclass
class EngineeringStrategy:
    """A structured representation of an engineering approach.

    Attributes:
        strategy_id: Unique identifier for this strategy.
        name: Human-readable strategy name.
        problem_class: The class of problem this strategy addresses.
        approach: Step-by-step description of the strategy.
        required_capabilities: Agent/tool capabilities needed.
        expected_cost: Relative cost estimate (0.0 = free, 1.0 = expensive).
        expected_risk: Risk of applying this strategy (0.0 = safe, 1.0 = risky).
        expected_success_probability: Estimated probability of success.
        verification_method: How to verify the strategy worked.
        fallback_strategy_id: ID of a fallback strategy if this one fails.
        tags: Searchable tags for retrieval.
        provenance: Where this strategy came from.
        created_at: When the strategy was created.
        last_used_at: When the strategy was last applied.
        use_count: Number of times this strategy has been applied.
        success_count: Number of times this strategy succeeded.
        failure_count: Number of times this strategy failed.
        metadata: Arbitrary additional data.
    """

    strategy_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    problem_class: ProblemClass = ProblemClass.UNKNOWN
    approach: str = ""
    required_capabilities: tuple[str, ...] = ()
    expected_cost: float = 0.5
    expected_risk: float = 0.5
    expected_success_probability: float = 0.5
    verification_method: str = ""
    fallback_strategy_id: str | None = None
    tags: tuple[str, ...] = ()
    provenance: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of applications that succeeded, or 0.5 with no history."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def is_reliable(self) -> bool:
        """Return True when the strategy has a strong enough success record."""
        total = self.success_count + self.failure_count
        return total >= 2 and self.success_rate >= 0.6

    @property
    def is_deprecated(self) -> bool:
        """Return True when the strategy has failed more often than it succeeded
        with sufficient evidence."""
        total = self.success_count + self.failure_count
        return total >= 3 and self.success_rate < 0.4

    def record_use(self, outcome: StrategyOutcome) -> None:
        """Record that this strategy was applied with the given outcome."""
        self.use_count += 1
        self.last_used_at = datetime.now(UTC)
        if outcome is StrategyOutcome.SUCCESS:
            self.success_count += 1
        elif outcome is StrategyOutcome.FAILURE:
            self.failure_count += 1

    def effectiveness_score(self) -> float:
        """Compute a composite effectiveness score.

        Combines historical success rate, recency of use, and base
        probability estimate.  Returns a value between 0.0 and 1.0.
        """
        base = self.expected_success_probability
        historical = self.success_rate
        total = self.success_count + self.failure_count

        if total == 0:
            return base

        # Weight historical evidence more heavily as use count grows
        weight = min(total / 10.0, 1.0)  # saturates at 10 uses
        return (1.0 - weight) * base + weight * historical

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "problem_class": self.problem_class.value,
            "approach": self.approach,
            "required_capabilities": list(self.required_capabilities),
            "expected_cost": self.expected_cost,
            "expected_risk": self.expected_risk,
            "expected_success_probability": self.expected_success_probability,
            "verification_method": self.verification_method,
            "fallback_strategy_id": self.fallback_strategy_id,
            "tags": list(self.tags),
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "effectiveness_score": round(self.effectiveness_score(), 4),
            "is_reliable": self.is_reliable,
            "is_deprecated": self.is_deprecated,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# StrategyComparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    """Structured comparison of candidate strategies.

    Produced by ``StrategyMemory.compare_strategies()`` to help the
    autonomous loop choose the best approach.
    """

    strategies: tuple[EngineeringStrategy, ...]
    scores: tuple[float, ...]
    reasoning: str
    recommended_index: int

    @property
    def recommended(self) -> EngineeringStrategy:
        """Return the recommended strategy."""
        return self.strategies[self.recommended_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": [s.to_dict() for s in self.strategies],
            "scores": list(self.scores),
            "reasoning": self.reasoning,
            "recommended_index": self.recommended_index,
            "recommended_name": self.recommended.name,
        }


# ---------------------------------------------------------------------------
# StrategyMemory
# ---------------------------------------------------------------------------


class StrategyMemory:
    """Stores and retrieves engineering strategies.

    The memory maintains an in-memory collection of strategies.  It
    supports:

    * **Storage**: Record new strategies and update existing ones.
    * **Retrieval**: Find strategies by problem class, tags, or
      capability requirements.
    * **Comparison**: Rank candidate strategies by effectiveness.
    * **Lifecycle**: Mark strategies as deprecated when they underperform.
    """

    def __init__(self, max_strategies: int = 500) -> None:
        self._strategies: dict[str, EngineeringStrategy] = {}
        self._max_strategies = max_strategies
        self._log = logger.bind(component="strategy_memory")

    # -- Storage -----------------------------------------------------------

    def store(self, strategy: EngineeringStrategy) -> None:
        """Add or update a strategy in memory."""
        if len(self._strategies) >= self._max_strategies:
            self._evict()
        self._strategies[strategy.strategy_id] = strategy
        self._log.info(
            "strategy_stored",
            strategy_id=strategy.strategy_id,
            name=strategy.name,
            problem_class=strategy.problem_class.value,
        )

    def get(self, strategy_id: str) -> EngineeringStrategy | None:
        """Retrieve a strategy by ID."""
        return self._strategies.get(strategy_id)

    def remove(self, strategy_id: str) -> bool:
        """Remove a strategy from memory."""
        removed = self._strategies.pop(strategy_id, None)
        if removed is not None:
            self._log.info("strategy_removed", strategy_id=strategy_id)
        return removed is not None

    # -- Retrieval ---------------------------------------------------------

    def retrieve_for_problem(
        self,
        problem_class: ProblemClass,
        *,
        tags: tuple[str, ...] = (),
        limit: int = 5,
    ) -> list[EngineeringStrategy]:
        """Find strategies that address a given problem class.

        Returns up to ``limit`` strategies sorted by effectiveness score
        (best first).  Deprecated strategies are excluded unless no
        reliable alternatives exist.
        """
        candidates = [s for s in self._strategies.values() if s.problem_class == problem_class]

        if tags:
            tag_set = set(tags)
            candidates = [s for s in candidates if tag_set & set(s.tags)]

        # Separate reliable and non-reliable
        reliable = [s for s in candidates if s.is_reliable and not s.is_deprecated]
        if reliable:
            reliable.sort(key=lambda s: s.effectiveness_score(), reverse=True)
            return reliable[:limit]

        # Fall back to all non-deprecated
        acceptable = [s for s in candidates if not s.is_deprecated]
        acceptable.sort(key=lambda s: s.effectiveness_score(), reverse=True)
        if acceptable:
            return acceptable[:limit]

        # Last resort: include deprecated (but sorted worst-first so caller
        # can decide whether to risk it)
        candidates.sort(key=lambda s: s.effectiveness_score(), reverse=True)
        return candidates[:limit]

    def retrieve_by_tags(
        self,
        tags: tuple[str, ...],
        *,
        limit: int = 10,
    ) -> list[EngineeringStrategy]:
        """Find strategies matching any of the given tags."""
        tag_set = set(tags)
        matches = [s for s in self._strategies.values() if tag_set & set(s.tags)]
        matches.sort(key=lambda s: s.effectiveness_score(), reverse=True)
        return matches[:limit]

    def all_strategies(self) -> list[EngineeringStrategy]:
        """Return all strategies sorted by effectiveness."""
        strategies = list(self._strategies.values())
        strategies.sort(key=lambda s: s.effectiveness_score(), reverse=True)
        return strategies

    # -- Comparison --------------------------------------------------------

    def compare_strategies(
        self,
        candidates: list[EngineeringStrategy],
    ) -> StrategyComparison | None:
        """Rank candidate strategies and produce a recommendation.

        Returns None when no candidates are provided.
        """
        if not candidates:
            return None

        scores = [s.effectiveness_score() for s in candidates]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        # Build reasoning
        parts: list[str] = []
        for i, (s, score) in enumerate(zip(candidates, scores, strict=True)):
            marker = " <-- recommended" if i == best_idx else ""
            parts.append(
                f"Strategy '{s.name}' (score={score:.3f}, "
                f"uses={s.use_count}, success_rate={s.success_rate:.2f})"
                f"{marker}"
            )
        reasoning = "\n".join(parts)

        return StrategyComparison(
            strategies=tuple(candidates),
            scores=tuple(scores),
            reasoning=reasoning,
            recommended_index=best_idx,
        )

    # -- Lifecycle ---------------------------------------------------------

    def record_outcome(
        self,
        strategy_id: str,
        outcome: StrategyOutcome,
    ) -> EngineeringStrategy | None:
        """Record the outcome of applying a strategy.

        Returns the updated strategy, or None if not found.
        """
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return None
        strategy.record_use(outcome)
        if strategy.is_deprecated:
            self._log.warning(
                "strategy_deprecated",
                strategy_id=strategy_id,
                name=strategy.name,
                success_rate=strategy.success_rate,
            )
        return strategy

    def deprecate(self, strategy_id: str) -> bool:
        """Explicitly deprecate a strategy."""
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return False
        strategy.failure_count = max(strategy.failure_count, 3)
        strategy.success_count = min(strategy.success_count, 0)
        self._log.info("strategy_explicitly_deprecated", strategy_id=strategy_id)
        return True

    # -- Internal ----------------------------------------------------------

    def _evict(self) -> None:
        """Remove the least effective deprecated strategy to make room."""
        deprecated = [s for s in self._strategies.values() if s.is_deprecated]
        if deprecated:
            worst = min(deprecated, key=lambda s: s.effectiveness_score())
            del self._strategies[worst.strategy_id]
            self._log.info("strategy_evicted", strategy_id=worst.strategy_id)
            return

        # If nothing deprecated, remove least-used strategy
        if self._strategies:
            least_used = min(self._strategies.values(), key=lambda s: s.use_count)
            del self._strategies[least_used.strategy_id]
            self._log.info(
                "strategy_evicted_least_used",
                strategy_id=least_used.strategy_id,
            )

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies


# ---------------------------------------------------------------------------
# Default strategies for common failure classes
# ---------------------------------------------------------------------------


def default_strategies() -> list[EngineeringStrategy]:
    """Return a set of built-in strategies for common problem classes.

    These serve as seed strategies that the system can refine through
    actual use.
    """
    return [
        EngineeringStrategy(
            name="retry_test_fix",
            problem_class=ProblemClass.TEST_FAILURE,
            approach=(
                "1. Inspect failing test output.\n"
                "2. Identify the specific assertion or error.\n"
                "3. Locate the affected source file.\n"
                "4. Apply minimal fix to satisfy the test.\n"
                "5. Re-run the targeted test.\n"
                "6. Run regression suite."
            ),
            required_capabilities=("code_generation", "test_execution"),
            expected_cost=0.3,
            expected_risk=0.2,
            expected_success_probability=0.7,
            verification_method="run_tests",
            tags=("test", "fix", "regression"),
            provenance="builtin",
        ),
        EngineeringStrategy(
            name="syntax_error_fix",
            problem_class=ProblemClass.SYNTAX_ERROR,
            approach=(
                "1. Parse the syntax error location from the traceback.\n"
                "2. Read the affected file around the error line.\n"
                "3. Fix the syntax (indentation, missing colon, etc.).\n"
                "4. Verify the file parses with ast.parse().\n"
                "5. Re-run the original command."
            ),
            required_capabilities=("code_generation",),
            expected_cost=0.1,
            expected_risk=0.1,
            expected_success_probability=0.9,
            verification_method="ast_parse",
            tags=("syntax", "quick_fix"),
            provenance="builtin",
        ),
        EngineeringStrategy(
            name="missing_dependency_repair",
            problem_class=ProblemClass.MISSING_DEPENDENCY,
            approach=(
                "1. Identify the missing module from the ImportError.\n"
                "2. Check pyproject.toml / requirements for the dependency.\n"
                "3. If declared but not installed: pip install.\n"
                "4. If not declared: add to appropriate dependency group.\n"
                "5. Verify import succeeds.\n"
                "6. Run tests to confirm no regressions."
            ),
            required_capabilities=("code_generation",),
            expected_cost=0.2,
            expected_risk=0.3,
            expected_success_probability=0.8,
            verification_method="import_check",
            tags=("dependency", "environment"),
            provenance="builtin",
        ),
        EngineeringStrategy(
            name="implementation_retry",
            problem_class=ProblemClass.INCORRECT_IMPLEMENTATION,
            approach=(
                "1. Analyze the verification failure evidence.\n"
                "2. Identify which acceptance criteria were not met.\n"
                "3. Inspect the generated implementation.\n"
                "4. Generate a corrected implementation with failure context.\n"
                "5. Re-run verification."
            ),
            required_capabilities=("code_generation",),
            expected_cost=0.5,
            expected_risk=0.3,
            expected_success_probability=0.5,
            verification_method="verification_engine",
            tags=("implementation", "retry"),
            provenance="builtin",
        ),
        EngineeringStrategy(
            name="timeout_investigation",
            problem_class=ProblemClass.TIMEOUT,
            approach=(
                "1. Identify which operation timed out.\n"
                "2. Check for infinite loops or blocking I/O.\n"
                "3. Reduce scope of the operation.\n"
                "4. If network-related: add retry with backoff.\n"
                "5. Increase timeout only as last resort.\n"
                "6. Re-run with narrower scope."
            ),
            required_capabilities=("code_generation", "debugging"),
            expected_cost=0.4,
            expected_risk=0.2,
            expected_success_probability=0.6,
            verification_method="run_tests",
            tags=("timeout", "performance"),
            provenance="builtin",
        ),
    ]


__all__ = [
    "EngineeringStrategy",
    "StrategyComparison",
    "StrategyMemory",
    "StrategyOutcome",
    "ProblemClass",
    "default_strategies",
]
