from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ActionType(StrEnum):
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    TEST_GENERATION = "test_generation"


class Outcome(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    REVERTED = "reverted"


@dataclass(frozen=True)
class ActionContext:
    action_type: ActionType
    repository: str
    pr_number: int | None = None
    file_paths: tuple[str, ...] = ()
    pattern_ids: tuple[uuid.UUID, ...] = ()
    complexity: float = 0.0


@dataclass(frozen=True)
class ExecutionResult:
    outcome: Outcome
    duration_secs: float
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    build_ok: bool = False
    lint_errors: int = 0
    coverage_delta: float = 0.0
    reverted: bool = False


@dataclass(frozen=True)
class Feedback:
    score: float
    max_score: float = 10.0
    approved: bool = False
    changes_requested: bool = False


class RewardSignal(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    action_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    reward: float
    components: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LearningSignal(BaseModel):
    pattern_id: uuid.UUID
    reward: float
    reinforce: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RewardConfig(BaseModel):
    outcome_weights: dict[Outcome, float] = Field(
        default_factory=lambda: {
            Outcome.SUCCESS: 1.0,
            Outcome.PARTIAL: 0.3,
            Outcome.FAILURE: -0.5,
            Outcome.ERROR: -0.8,
            Outcome.TIMEOUT: -0.6,
            Outcome.REVERTED: -1.2,
        }
    )
    test_weight: float = 0.3
    test_fail_weight: float = 0.4
    ci_bonus: float = 0.2
    ci_penalty: float = 0.3
    review_weight: float = 0.4
    coverage_weight: float = 0.2
    lint_error_penalty: float = 0.05
    revert_penalty: float = 0.5
    pattern_bonus: float = 0.1
    pattern_penalty: float = 0.2


class RewardRepository(ABC):
    @abstractmethod
    async def store_signal(self, signal: RewardSignal) -> None: ...

    @abstractmethod
    async def store_learning_signals(self, signals: list[LearningSignal]) -> None: ...

    @abstractmethod
    async def get_pattern_success_rates(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, float]: ...


def _calc_outcome(outcome: Outcome, weights: dict[Outcome, float]) -> float:
    return weights.get(outcome, 0.0)


def _calc_tests(result: ExecutionResult, config: RewardConfig) -> float:
    if result.tests_total == 0:
        return 0.0
    pass_rate = result.tests_passed / result.tests_total
    fail_rate = result.tests_failed / result.tests_total
    return pass_rate * config.test_weight - fail_rate * config.test_fail_weight


def _calc_ci(result: ExecutionResult, config: RewardConfig) -> float:
    return config.ci_bonus if result.build_ok else -config.ci_penalty


def _calc_review(feedback: Feedback | None, config: RewardConfig) -> float:
    if feedback is None or feedback.max_score <= 0:
        return 0.0
    normalized = feedback.score / feedback.max_score
    centered = (normalized - 0.5) * 2.0
    reward = centered * config.review_weight
    if feedback.changes_requested:
        reward -= 0.2
    return reward


def _calc_coverage(result: ExecutionResult, config: RewardConfig) -> float:
    return result.coverage_delta * config.coverage_weight


def _calc_quality(result: ExecutionResult, config: RewardConfig) -> float:
    return -result.lint_errors * config.lint_error_penalty


def _calc_revert(result: ExecutionResult, config: RewardConfig) -> float:
    return -config.revert_penalty if result.reverted else 0.0


class RewardModel:
    def __init__(
        self,
        config: RewardConfig | None = None,
        repository: RewardRepository | None = None,
    ) -> None:
        self._config = config or RewardConfig()
        self._repo = repository
        self._pattern_rates: dict[uuid.UUID, float] = {}

    async def calculate(
        self,
        context: ActionContext,
        result: ExecutionResult,
        feedback: Feedback | None = None,
    ) -> RewardSignal:
        if context.pattern_ids and self._repo:
            self._pattern_rates = await self._repo.get_pattern_success_rates(
                list(context.pattern_ids)
            )

        components: dict[str, float] = {
            "outcome": _calc_outcome(result.outcome, self._config.outcome_weights),
            "tests": _calc_tests(result, self._config),
            "ci": _calc_ci(result, self._config),
            "review": _calc_review(feedback, self._config),
            "coverage": _calc_coverage(result, self._config),
            "quality": _calc_quality(result, self._config),
            "revert": _calc_revert(result, self._config),
            "pattern": self._calc_pattern(context, result),
        }
        total = round(sum(components.values()), 6)
        confidence = self._calc_confidence(result, feedback)

        signal = RewardSignal(reward=total, components=components, confidence=confidence)

        logger.info(
            "reward_calculated",
            action=context.action_type,
            outcome=result.outcome,
            reward=total,
        )

        if self._repo:
            await self._repo.store_signal(signal)

        return signal

    def generate_signals(
        self, signal: RewardSignal, pattern_ids: tuple[uuid.UUID, ...]
    ) -> list[LearningSignal]:
        if not pattern_ids:
            return []
        weight = 1.0 / len(pattern_ids)
        return [
            LearningSignal(
                pattern_id=pid,
                reward=round(signal.reward * weight, 6),
                reinforce=signal.reward > 0,
            )
            for pid in pattern_ids
        ]

    async def calculate_and_learn(
        self,
        context: ActionContext,
        result: ExecutionResult,
        feedback: Feedback | None = None,
    ) -> tuple[RewardSignal, list[LearningSignal]]:
        signal = await self.calculate(context, result, feedback)
        learning = self.generate_signals(signal, context.pattern_ids)
        if self._repo and learning:
            await self._repo.store_learning_signals(learning)
        return signal, learning

    def _calc_pattern(self, context: ActionContext, result: ExecutionResult) -> float:
        if not context.pattern_ids:
            return 0.0
        total = 0.0
        for pid in context.pattern_ids:
            rate = self._pattern_rates.get(pid, 0.5)
            if result.outcome == Outcome.SUCCESS:
                total += rate * self._config.pattern_bonus
            elif result.outcome in (Outcome.FAILURE, Outcome.ERROR, Outcome.REVERTED):
                total -= (1.0 - rate) * self._config.pattern_penalty
        return total

    @staticmethod
    def _calc_confidence(result: ExecutionResult, feedback: Feedback | None) -> float:
        c = 0.5
        if result.outcome in (Outcome.SUCCESS, Outcome.REVERTED):
            c += 0.2
        elif result.outcome in (Outcome.FAILURE, Outcome.ERROR):
            c += 0.15
        if result.build_ok is not None:
            c += 0.1
        if result.tests_total > 0:
            c += 0.1
        if feedback is not None:
            c += 0.1
        return min(c, 1.0)
