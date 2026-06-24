from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import asyncpg
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class FeedbackSource(StrEnum):
    PR_REVIEW = "pr_review"
    TASK_EXECUTION = "task_execution"
    AGENT_EVALUATION = "agent_evaluation"
    USER_RATING = "user_rating"


class FeedbackSentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class FeedbackSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NormalizedFeedback(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: FeedbackSource
    sentiment: FeedbackSentiment
    severity: FeedbackSeverity = FeedbackSeverity.LOW
    score: float  # normalized [-1.0, 1.0]
    raw_score: float | None = None
    task_id: str | None = None
    agent_id: str | None = None
    repo: str | None = None
    pr_number: int | None = None
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))


class PRReviewFeedback(BaseModel):
    pr_number: int
    repo: str
    reviewer: str
    approved: bool
    comments: list[str] = Field(default_factory=list)
    requested_changes: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    review_state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED
    agent_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskExecutionFeedback(BaseModel):
    task_id: str
    agent_id: str
    success: bool
    exit_code: int | None = None
    duration_ms: float | None = None
    tests_passed: int = 0
    tests_failed: int = 0
    error_message: str | None = None
    retries: int = 0
    repo: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluationFeedback(BaseModel):
    agent_id: str
    task_id: str
    evaluator_id: str
    criteria_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float  # 0.0 - 10.0
    notes: str | None = None
    repo: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("overall_score")
    @classmethod
    def clamp_overall(cls, v: float) -> float:
        return max(0.0, min(10.0, v))


class UserRatingFeedback(BaseModel):
    user_id: str
    task_id: str
    agent_id: str | None = None
    rating: int  # 1-5
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)
    repo: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rating")
    @classmethod
    def clamp_rating(cls, v: int) -> int:
        return max(1, min(5, v))


class FeedbackSummary(BaseModel):
    total: int
    by_source: dict[str, int]
    by_sentiment: dict[str, int]
    average_score: float
    positive_rate: float
    negative_rate: float
    period_start: datetime | None = None
    period_end: datetime | None = None


class FeedbackCollector:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_feedback (
                    id          TEXT PRIMARY KEY,
                    source      TEXT NOT NULL,
                    sentiment   TEXT NOT NULL,
                    severity    TEXT NOT NULL,
                    score       FLOAT NOT NULL,
                    raw_score   FLOAT,
                    task_id     TEXT,
                    agent_id    TEXT,
                    repo        TEXT,
                    pr_number   INTEGER,
                    comment     TEXT,
                    tags        JSONB NOT NULL DEFAULT '[]',
                    metadata    JSONB NOT NULL DEFAULT '{}',
                    created_at  TIMESTAMPTZ NOT NULL
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lf_source ON learning_feedback (source)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lf_task ON learning_feedback (task_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lf_agent ON learning_feedback (agent_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lf_sentiment ON learning_feedback (sentiment)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lf_created ON learning_feedback (created_at DESC)"
            )
        logger.info("FeedbackCollector initialized")

    async def collect_pr_review(self, feedback: PRReviewFeedback) -> NormalizedFeedback:
        normalized = self._normalize_pr_review(feedback)
        await self._persist(normalized)
        logger.info(
            "Collected PR review feedback pr#%d repo=%s sentiment=%s",
            feedback.pr_number,
            feedback.repo,
            normalized.sentiment,
        )
        return normalized

    async def collect_task_execution(self, feedback: TaskExecutionFeedback) -> NormalizedFeedback:
        normalized = self._normalize_task_execution(feedback)
        await self._persist(normalized)
        logger.info(
            "Collected task execution feedback task=%s success=%s score=%.2f",
            feedback.task_id,
            feedback.success,
            normalized.score,
        )
        return normalized

    async def collect_agent_evaluation(
        self, feedback: AgentEvaluationFeedback
    ) -> NormalizedFeedback:
        normalized = self._normalize_agent_evaluation(feedback)
        await self._persist(normalized)
        logger.info(
            "Collected agent evaluation feedback agent=%s score=%.2f",
            feedback.agent_id,
            normalized.score,
        )
        return normalized

    async def collect_user_rating(self, feedback: UserRatingFeedback) -> NormalizedFeedback:
        normalized = self._normalize_user_rating(feedback)
        await self._persist(normalized)
        logger.info(
            "Collected user rating feedback task=%s rating=%d",
            feedback.task_id,
            feedback.rating,
        )
        return normalized

    async def get(self, feedback_id: str) -> NormalizedFeedback | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM learning_feedback WHERE id = $1", feedback_id
            )
        return NormalizedFeedback(**dict(row)) if row else None

    async def list_for_task(self, task_id: str) -> list[NormalizedFeedback]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM learning_feedback WHERE task_id = $1 ORDER BY created_at DESC",
                task_id,
            )
        return [NormalizedFeedback(**dict(r)) for r in rows]

    async def list_for_agent(
        self,
        agent_id: str,
        source: FeedbackSource | None = None,
        limit: int = 100,
    ) -> list[NormalizedFeedback]:
        if source:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM learning_feedback
                       WHERE agent_id = $1 AND source = $2
                       ORDER BY created_at DESC LIMIT $3""",
                    agent_id,
                    source.value,
                    limit,
                )
        else:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM learning_feedback
                       WHERE agent_id = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    agent_id,
                    limit,
                )
        return [NormalizedFeedback(**dict(r)) for r in rows]

    async def list_for_repo(
        self,
        repo: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[NormalizedFeedback]:
        if since:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM learning_feedback
                       WHERE repo = $1 AND created_at >= $2
                       ORDER BY created_at DESC LIMIT $3""",
                    repo,
                    since,
                    limit,
                )
        else:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM learning_feedback
                       WHERE repo = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    repo,
                    limit,
                )
        return [NormalizedFeedback(**dict(r)) for r in rows]

    async def summarize(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        repo: str | None = None,
        since: datetime | None = None,
    ) -> FeedbackSummary:
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if agent_id:
            conditions.append(f"agent_id = ${idx}")
            params.append(agent_id)
            idx += 1
        if task_id:
            conditions.append(f"task_id = ${idx}")
            params.append(task_id)
            idx += 1
        if repo:
            conditions.append(f"repo = ${idx}")
            params.append(repo)
            idx += 1
        if since:
            conditions.append(f"created_at >= ${idx}")
            params.append(since)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT source, sentiment, score, created_at FROM learning_feedback {where}",
                *params,
            )

        if not rows:
            return FeedbackSummary(
                total=0,
                by_source={},
                by_sentiment={},
                average_score=0.0,
                positive_rate=0.0,
                negative_rate=0.0,
            )

        by_source: dict[str, int] = {}
        by_sentiment: dict[str, int] = {}
        total_score = 0.0
        dates: list[datetime] = []

        for row in rows:
            by_source[row["source"]] = by_source.get(row["source"], 0) + 1
            by_sentiment[row["sentiment"]] = by_sentiment.get(row["sentiment"], 0) + 1
            total_score += row["score"]
            dates.append(row["created_at"])

        total = len(rows)
        return FeedbackSummary(
            total=total,
            by_source=by_source,
            by_sentiment=by_sentiment,
            average_score=round(total_score / total, 4),
            positive_rate=round(by_sentiment.get("positive", 0) / total, 4),
            negative_rate=round(by_sentiment.get("negative", 0) / total, 4),
            period_start=min(dates) if dates else None,
            period_end=max(dates) if dates else None,
        )

    async def delete_for_task(self, task_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM learning_feedback WHERE task_id = $1", task_id
            )
        count = int(result.split()[-1])
        logger.info("Deleted %d feedback records for task %s", count, task_id)
        return count

    async def _persist(self, feedback: NormalizedFeedback) -> None:
        import json

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO learning_feedback (
                    id, source, sentiment, severity, score, raw_score,
                    task_id, agent_id, repo, pr_number, comment,
                    tags, metadata, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                ON CONFLICT (id) DO NOTHING
                """,
                feedback.id,
                feedback.source.value,
                feedback.sentiment.value,
                feedback.severity.value,
                feedback.score,
                feedback.raw_score,
                feedback.task_id,
                feedback.agent_id,
                feedback.repo,
                feedback.pr_number,
                feedback.comment,
                json.dumps(feedback.tags),
                json.dumps(feedback.metadata),
                feedback.created_at,
            )

    @staticmethod
    def _normalize_pr_review(fb: PRReviewFeedback) -> NormalizedFeedback:
        if fb.review_state == "APPROVED":
            score = 0.8
            sentiment = FeedbackSentiment.POSITIVE
            severity = FeedbackSeverity.LOW
        elif fb.review_state == "CHANGES_REQUESTED":
            score = -0.6 - min(0.3, len(fb.requested_changes) * 0.05)
            sentiment = FeedbackSentiment.NEGATIVE
            severity = FeedbackSeverity.HIGH if len(fb.requested_changes) > 3 else FeedbackSeverity.MEDIUM
        else:
            score = 0.1
            sentiment = FeedbackSentiment.NEUTRAL
            severity = FeedbackSeverity.LOW

        all_comments = fb.comments + fb.requested_changes
        comment_text = " | ".join(all_comments[:5]) if all_comments else None

        return NormalizedFeedback(
            source=FeedbackSource.PR_REVIEW,
            sentiment=sentiment,
            severity=severity,
            score=score,
            raw_score=None,
            task_id=fb.task_id,
            agent_id=fb.agent_id,
            repo=fb.repo,
            pr_number=fb.pr_number,
            comment=comment_text,
            tags=fb.labels + ["pr_review", fb.review_state.lower()],
            metadata={
                "reviewer": fb.reviewer,
                "approved": fb.approved,
                "change_count": len(fb.requested_changes),
                "comment_count": len(fb.comments),
                **fb.metadata,
            },
        )

    @staticmethod
    def _normalize_task_execution(fb: TaskExecutionFeedback) -> NormalizedFeedback:
        total_tests = fb.tests_passed + fb.tests_failed
        test_pass_rate = fb.tests_passed / total_tests if total_tests > 0 else None

        if fb.success:
            base_score = 0.7
            if test_pass_rate is not None:
                base_score += test_pass_rate * 0.3
            if fb.retries == 0:
                base_score = min(1.0, base_score + 0.05)
            score = base_score
            sentiment = FeedbackSentiment.POSITIVE
            severity = FeedbackSeverity.LOW
        else:
            base_score = -0.5
            if fb.retries > 2:
                base_score -= 0.2
            if test_pass_rate is not None and test_pass_rate < 0.5:
                base_score -= 0.2
            score = base_score
            sentiment = FeedbackSentiment.NEGATIVE
            severity = FeedbackSeverity.CRITICAL if fb.retries > 3 else FeedbackSeverity.HIGH

        tags = ["task_execution"]
        if fb.success:
            tags.append("success")
        else:
            tags.append("failure")
            if fb.error_message:
                tags.append("error")

        return NormalizedFeedback(
            source=FeedbackSource.TASK_EXECUTION,
            sentiment=sentiment,
            severity=severity,
            score=score,
            raw_score=test_pass_rate,
            task_id=fb.task_id,
            agent_id=fb.agent_id,
            repo=fb.repo,
            comment=fb.error_message,
            tags=tags,
            metadata={
                "exit_code": fb.exit_code,
                "duration_ms": fb.duration_ms,
                "tests_passed": fb.tests_passed,
                "tests_failed": fb.tests_failed,
                "retries": fb.retries,
                **fb.metadata,
            },
        )

    @staticmethod
    def _normalize_agent_evaluation(fb: AgentEvaluationFeedback) -> NormalizedFeedback:
        normalized_score = (fb.overall_score / 10.0) * 2.0 - 1.0

        if normalized_score >= 0.4:
            sentiment = FeedbackSentiment.POSITIVE
            severity = FeedbackSeverity.LOW
        elif normalized_score <= -0.2:
            sentiment = FeedbackSentiment.NEGATIVE
            severity = FeedbackSeverity.HIGH if normalized_score < -0.6 else FeedbackSeverity.MEDIUM
        else:
            sentiment = FeedbackSentiment.NEUTRAL
            severity = FeedbackSeverity.LOW

        criteria_tags = [k.replace(" ", "_") for k in fb.criteria_scores]

        return NormalizedFeedback(
            source=FeedbackSource.AGENT_EVALUATION,
            sentiment=sentiment,
            severity=severity,
            score=normalized_score,
            raw_score=fb.overall_score,
            task_id=fb.task_id,
            agent_id=fb.agent_id,
            repo=fb.repo,
            comment=fb.notes,
            tags=["agent_evaluation"] + criteria_tags,
            metadata={
                "evaluator_id": fb.evaluator_id,
                "criteria_scores": fb.criteria_scores,
                "overall_score": fb.overall_score,
                **fb.metadata,
            },
        )

    @staticmethod
    def _normalize_user_rating(fb: UserRatingFeedback) -> NormalizedFeedback:
        # 1-5 → [-1.0, 1.0]: (rating - 3) / 2
        score = (fb.rating - 3) / 2.0

        if fb.rating >= 4:
            sentiment = FeedbackSentiment.POSITIVE
            severity = FeedbackSeverity.LOW
        elif fb.rating <= 2:
            sentiment = FeedbackSentiment.NEGATIVE
            severity = FeedbackSeverity.HIGH if fb.rating == 1 else FeedbackSeverity.MEDIUM
        else:
            sentiment = FeedbackSentiment.NEUTRAL
            severity = FeedbackSeverity.LOW

        return NormalizedFeedback(
            source=FeedbackSource.USER_RATING,
            sentiment=sentiment,
            severity=severity,
            score=score,
            raw_score=float(fb.rating),
            task_id=fb.task_id,
            agent_id=fb.agent_id,
            repo=fb.repo,
            comment=fb.comment,
            tags=["user_rating"] + fb.tags,
            metadata={
                "user_id": fb.user_id,
                "rating": fb.rating,
                **fb.metadata,
            },
        )