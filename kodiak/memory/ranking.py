# kodiak/memory/ranking.py
"""Ranking engine for scoring and ordering candidate memories."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Sequence

import structlog

from .models import Memory, SearchResult

logger = structlog.get_logger(__name__)

__all__ = ["MemoryRanker"]


class MemoryRanker:
    """Multi-factor memory ranking engine."""

    def __init__(
        self,
        weight_relevance: float = 0.5,
        weight_recency: float = 0.2,
        weight_confidence: float = 0.2,
        weight_tag_match: float = 0.1,
        half_life_seconds: float = 86400.0,
    ) -> None:
        """Initialize MemoryRanker.

        Args:
            weight_relevance: Weight coefficient for text/semantic query relevance.
            weight_recency: Weight coefficient for recency time-decay score.
            weight_confidence: Weight coefficient for memory confidence/significance score.
            weight_tag_match: Weight coefficient for tag match ratio.
            half_life_seconds: Exponential half-life in seconds for recency decay (default 24h).
        """
        self.weight_relevance = weight_relevance
        self.weight_recency = weight_recency
        self.weight_confidence = weight_confidence
        self.weight_tag_match = weight_tag_match
        self.half_life_seconds = max(1.0, half_life_seconds)

    def rank(
        self,
        query: str,
        memories: Sequence[Memory],
        tags: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Score and rank candidate memories against a query and optional tags.

        Args:
            query: Query text string.
            memories: Collection of candidate Memory objects.
            tags: Optional filter/target tags.
            limit: Maximum ranked results to return.

        Returns:
            List of SearchResult instances sorted by total relevance score descending.
        """
        if not memories:
            return []

        now = datetime.now(UTC)
        query_terms = [t.lower() for t in query.split() if t]
        target_tags = set(tags or [])

        scored_results: list[SearchResult] = []

        for memory in memories:
            rel_score = self._compute_relevance_score(query_terms, memory)
            rec_score = self._compute_recency_score(now, memory.created_at)
            conf_score = max(0.0, min(1.0, memory.confidence))
            tag_score = self._compute_tag_score(target_tags, memory.tags)

            composite_score = (
                self.weight_relevance * rel_score
                + self.weight_recency * rec_score
                + self.weight_confidence * conf_score
                + self.weight_tag_match * tag_score
            )

            # Clamp composite score to [0.0, 1.0]
            final_score = max(0.0, min(1.0, composite_score))

            scored_results.append(
                SearchResult(
                    memory=memory,
                    relevance_score=final_score,
                )
            )

        scored_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored_results[:limit]

    def _compute_relevance_score(self, query_terms: list[str], memory: Memory) -> float:
        if not query_terms:
            return 1.0

        full_text = f"{memory.title} {memory.content} {' '.join(memory.tags)}".lower()
        matches = sum(1 for term in query_terms if term in full_text)
        return matches / len(query_terms)

    def _compute_recency_score(self, now: datetime, created_at: datetime) -> float:
        try:
            age_seconds = max(0.0, (now - created_at).total_seconds())
            # Exponential decay: e^(-lambda * t), where lambda = ln(2) / half_life
            decay_constant = math.log(2) / self.half_life_seconds
            return math.exp(-decay_constant * age_seconds)
        except Exception:
            return 0.5

    @staticmethod
    def _compute_tag_score(target_tags: set[str], memory_tags: list[str]) -> float:
        if not target_tags:
            return 1.0
        if not memory_tags:
            return 0.0
        overlap = target_tags & set(memory_tags)
        return len(overlap) / len(target_tags)
