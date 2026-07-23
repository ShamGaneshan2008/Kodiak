from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatternType(StrEnum):
    CODING = "coding"
    ARCHITECTURAL = "architectural"
    ANTI_PATTERN = "anti_pattern"
    REFACTORING = "refactoring"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"


class PatternStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    PENDING_REVIEW = "pending_review"


class Pattern(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    pattern_type: PatternType
    status: PatternStatus = PatternStatus.ACTIVE
    language: str
    tags: list[str] = Field(default_factory=list)
    code_template: str | None = None
    example_before: str | None = None
    example_after: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    frequency: int = 1
    success_rate: float = 0.0
    version: int = 1
    source_repo: str | None = None
    content_hash: str = ""
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = f"{self.name}:{self.language}:{self.code_template or ''}"
        return hashlib.sha256(content.encode()).hexdigest()


class PatternFilter(BaseModel):
    pattern_type: PatternType | None = None
    language: str | None = None
    status: PatternStatus | None = None
    tags: list[str] | None = None
    source_repo: str | None = None
    min_success_rate: float | None = None
    limit: int = 50
    offset: int = 0


class SimilarityResult(BaseModel):
    pattern: Pattern
    similarity_score: float


class PatternStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'active',
                    language    TEXT NOT NULL,
                    tags        JSONB NOT NULL DEFAULT '[]',
                    code_template TEXT,
                    example_before TEXT,
                    example_after  TEXT,
                    context     JSONB NOT NULL DEFAULT '{}',
                    frequency   INTEGER NOT NULL DEFAULT 1,
                    success_rate FLOAT NOT NULL DEFAULT 0.0,
                    version     INTEGER NOT NULL DEFAULT 1,
                    source_repo TEXT,
                    content_hash TEXT NOT NULL,
                    embedding   JSONB,
                    created_at  TIMESTAMPTZ NOT NULL,
                    updated_at  TIMESTAMPTZ NOT NULL
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lp_language ON learning_patterns (language)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lp_type ON learning_patterns (pattern_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lp_status ON learning_patterns (status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lp_hash ON learning_patterns (content_hash)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lp_tags ON learning_patterns USING gin (tags)"
            )
        logger.info("PatternStore initialized")

    async def create(self, pattern: Pattern) -> Pattern:
        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id, version FROM learning_patterns WHERE content_hash = $1",
                pattern.content_hash,
            )
            if existing:
                logger.debug("Pattern with hash %s already exists", pattern.content_hash)
                return await self.get(existing["id"])  # type: ignore[return-value]

            await conn.execute(
                """
                INSERT INTO learning_patterns (
                    id, name, description, pattern_type, status, language,
                    tags, code_template, example_before, example_after, context,
                    frequency, success_rate, version, source_repo, content_hash,
                    embedding, created_at, updated_at
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19
                )
                """,
                pattern.id,
                pattern.name,
                pattern.description,
                pattern.pattern_type.value,
                pattern.status.value,
                pattern.language,
                json.dumps(pattern.tags),
                pattern.code_template,
                pattern.example_before,
                pattern.example_after,
                json.dumps(pattern.context),
                pattern.frequency,
                pattern.success_rate,
                pattern.version,
                pattern.source_repo,
                pattern.content_hash,
                json.dumps(pattern.embedding) if pattern.embedding else None,
                pattern.created_at,
                pattern.updated_at,
            )
        logger.info("Created pattern %s (%s)", pattern.id, pattern.name)
        return pattern

    async def get(self, pattern_id: str) -> Pattern | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM learning_patterns WHERE id = $1", pattern_id)
        return self._row_to_pattern(row) if row else None

    async def get_by_hash(self, content_hash: str) -> Pattern | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM learning_patterns WHERE content_hash = $1", content_hash
            )
        return self._row_to_pattern(row) if row else None

    async def update(self, pattern: Pattern) -> Pattern | None:
        pattern.updated_at = datetime.now(UTC)
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE learning_patterns SET
                    name = $1, description = $2, pattern_type = $3, status = $4,
                    language = $5, tags = $6, code_template = $7,
                    example_before = $8, example_after = $9, context = $10,
                    frequency = $11, success_rate = $12, version = $13,
                    source_repo = $14, content_hash = $15, embedding = $16,
                    updated_at = $17
                WHERE id = $18
                """,
                pattern.name,
                pattern.description,
                pattern.pattern_type.value,
                pattern.status.value,
                pattern.language,
                json.dumps(pattern.tags),
                pattern.code_template,
                pattern.example_before,
                pattern.example_after,
                json.dumps(pattern.context),
                pattern.frequency,
                pattern.success_rate,
                pattern.version,
                pattern.source_repo,
                pattern.content_hash,
                json.dumps(pattern.embedding) if pattern.embedding else None,
                pattern.updated_at,
                pattern.id,
            )
        if result == "UPDATE 0":
            return None
        logger.info("Updated pattern %s", pattern.id)
        return pattern

    async def delete(self, pattern_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM learning_patterns WHERE id = $1", pattern_id)
        deleted = result != "DELETE 0"
        if deleted:
            logger.info("Deleted pattern %s", pattern_id)
        return deleted

    async def deprecate(self, pattern_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE learning_patterns SET status = 'deprecated', updated_at = $1 WHERE id = $2",
                datetime.now(UTC),
                pattern_id,
            )
        return result != "UPDATE 0"

    async def list(self, filters: PatternFilter) -> list[Pattern]:
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if filters.pattern_type:
            conditions.append(f"pattern_type = ${idx}")
            params.append(filters.pattern_type.value)
            idx += 1
        if filters.language:
            conditions.append(f"language = ${idx}")
            params.append(filters.language)
            idx += 1
        if filters.status:
            conditions.append(f"status = ${idx}")
            params.append(filters.status.value)
            idx += 1
        if filters.source_repo:
            conditions.append(f"source_repo = ${idx}")
            params.append(filters.source_repo)
            idx += 1
        if filters.min_success_rate is not None:
            conditions.append(f"success_rate >= ${idx}")
            params.append(filters.min_success_rate)
            idx += 1
        if filters.tags:
            conditions.append(f"tags @> ${idx}::jsonb")
            params.append(json.dumps(filters.tags))
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM learning_patterns
            {where}
            ORDER BY frequency DESC, success_rate DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([filters.limit, filters.offset])

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [self._row_to_pattern(r) for r in rows]

    async def increment_frequency(self, pattern_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE learning_patterns SET frequency = frequency + 1, "
                "updated_at = $1 WHERE id = $2",
                datetime.now(UTC),
                pattern_id,
            )

    async def update_success_rate(self, pattern_id: str, success_rate: float) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE learning_patterns SET success_rate = $1, updated_at = $2 WHERE id = $3",
                max(0.0, min(1.0, success_rate)),
                datetime.now(UTC),
                pattern_id,
            )

    async def find_similar_by_embedding(
        self,
        embedding: list[float],
        language: str | None = None,
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[SimilarityResult]:
        """Cosine similarity search using stored embeddings."""
        conditions = ["embedding IS NOT NULL", "status = 'active'"]
        params: list[Any] = []
        idx = 1

        if language:
            conditions.append(f"language = ${idx}")
            params.append(language)
            idx += 1

        params.append(limit * 5)  # over-fetch to filter by threshold

        where = " AND ".join(conditions)
        query = f"""
            SELECT *, embedding FROM learning_patterns
            WHERE {where}
            LIMIT ${idx}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results: list[SimilarityResult] = []
        for row in rows:
            stored_emb = json.loads(row["embedding"])
            score = self._cosine_similarity(embedding, stored_emb)
            if score >= min_similarity:
                results.append(
                    SimilarityResult(pattern=self._row_to_pattern(row), similarity_score=score)
                )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:limit]

    async def find_by_tags(self, tags: list[str], language: str | None = None) -> list[Pattern]:
        params: list[Any] = [json.dumps(tags)]
        conditions = ["tags @> $1::jsonb", "status = 'active'"]
        idx = 2

        if language:
            conditions.append(f"language = ${idx}")
            params.append(language)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM learning_patterns WHERE {where} ORDER BY frequency DESC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [self._row_to_pattern(r) for r in rows]

    async def count(self, filters: PatternFilter | None = None) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if filters:
            if filters.pattern_type:
                conditions.append(f"pattern_type = ${idx}")
                params.append(filters.pattern_type.value)
                idx += 1
            if filters.language:
                conditions.append(f"language = ${idx}")
                params.append(filters.language)
                idx += 1
            if filters.status:
                conditions.append(f"status = ${idx}")
                params.append(filters.status.value)
                idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM learning_patterns {where}", *params)

    async def get_stats(self) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM learning_patterns")
            by_type = await conn.fetch(
                "SELECT pattern_type, COUNT(*) as cnt FROM learning_patterns GROUP BY pattern_type"
            )
            by_lang = await conn.fetch(
                "SELECT language, COUNT(*) as cnt FROM learning_patterns "
                "GROUP BY language ORDER BY cnt DESC LIMIT 10"
            )
            avg_success = await conn.fetchval(
                "SELECT AVG(success_rate) FROM learning_patterns WHERE status = 'active'"
            )

        return {
            "total": total,
            "by_type": {r["pattern_type"]: r["cnt"] for r in by_type},
            "top_languages": {r["language"]: r["cnt"] for r in by_lang},
            "average_success_rate": round(float(avg_success or 0.0), 4),
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _row_to_pattern(row: asyncpg.Record) -> Pattern:
        data = dict(row)
        data["tags"] = json.loads(data["tags"]) if isinstance(data["tags"], str) else data["tags"]
        data["context"] = (
            json.loads(data["context"]) if isinstance(data["context"], str) else data["context"]
        )
        data["embedding"] = (
            json.loads(data["embedding"])
            if isinstance(data.get("embedding"), str)
            else data.get("embedding")
        )
        return Pattern(**data)
