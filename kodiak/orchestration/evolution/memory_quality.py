"""Memory quality control — prevent memory from becoming a junk drawer.

As memory grows, implement quality metrics:
- confidence tracking
- importance scoring
- freshness decay
- evidence strength
- duplication detection
- contradiction detection
- usage frequency

When contradictory memories are found, investigate whether different
task types, environments, or constraints explain the contradiction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single memory entry with quality metadata."""

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    memory_type: str = ""  # episodic, semantic, procedural, strategy, lesson
    confidence: float = 0.5
    importance: float = 0.5
    evidence_strength: float = 0.5
    usage_count: int = 0
    source: str = ""
    tags: tuple[str, ...] = ()
    task_context: str = ""
    strategy_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_modified: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def freshness(self) -> float:
        """Freshness decays over time. 1.0 = fresh, 0.0 = stale."""
        age_hours = (datetime.now(UTC) - self.last_accessed).total_seconds() / 3600
        # Exponential decay: half-life of 168 hours (1 week)
        import math

        return math.exp(-age_hours / 168.0)

    @property
    def quality_score(self) -> float:
        """Composite quality score."""
        return (
            self.confidence * 0.3
            + self.importance * 0.25
            + self.evidence_strength * 0.25
            + self.freshness * 0.1
            + min(self.usage_count / 10.0, 1.0) * 0.1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "importance": self.importance,
            "evidence_strength": self.evidence_strength,
            "usage_count": self.usage_count,
            "source": self.source,
            "tags": list(self.tags),
            "task_context": self.task_context,
            "strategy_id": self.strategy_id,
            "freshness": round(self.freshness, 4),
            "quality_score": round(self.quality_score, 4),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class Contradiction:
    """A detected contradiction between two memory entries."""

    contradiction_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    entry_a_id: str = ""
    entry_b_id: str = ""
    entry_a_content: str = ""
    entry_b_content: str = ""
    explanation: str = ""
    possible_reasons: tuple[str, ...] = ()
    confidence: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "entry_a_id": self.entry_a_id,
            "entry_b_id": self.entry_b_id,
            "entry_a_content": self.entry_a_content,
            "entry_b_content": self.entry_b_content,
            "explanation": self.explanation,
            "possible_reasons": list(self.possible_reasons),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Report on memory quality metrics."""

    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    total_entries: int = 0
    avg_confidence: float = 0.0
    avg_importance: float = 0.0
    avg_freshness: float = 0.0
    avg_quality_score: float = 0.0
    low_quality_count: int = 0
    stale_count: int = 0
    duplicate_count: int = 0
    contradiction_count: int = 0
    type_distribution: dict[str, int] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "total_entries": self.total_entries,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_importance": round(self.avg_importance, 4),
            "avg_freshness": round(self.avg_freshness, 4),
            "avg_quality_score": round(self.avg_quality_score, 4),
            "low_quality_count": self.low_quality_count,
            "stale_count": self.stale_count,
            "duplicate_count": self.duplicate_count,
            "contradiction_count": self.contradiction_count,
            "type_distribution": dict(self.type_distribution),
            "recommendations": list(self.recommendations),
            "created_at": self.created_at.isoformat(),
        }


class MemoryQualityController:
    """Monitors and improves memory quality.

    Detects contradictions, duplicates, stale entries, and low-quality
    memories.  Provides recommendations for cleanup.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.3,
        freshness_threshold: float = 0.1,
        duplicate_similarity_threshold: float = 0.8,
    ) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._confidence_threshold = confidence_threshold
        self._freshness_threshold = freshness_threshold
        self._duplicate_similarity_threshold = duplicate_similarity_threshold
        self._contradictions: list[Contradiction] = []
        self._log = logger.bind(component="memory_quality_controller")

    def add_entry(self, entry: MemoryEntry) -> None:
        self._entries[entry.entry_id] = entry

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def access_entry(self, entry_id: str) -> MemoryEntry | None:
        """Mark an entry as accessed (updates freshness)."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        # Can't modify frozen dataclass, so we need to replace
        new_entry = MemoryEntry(
            entry_id=entry.entry_id,
            content=entry.content,
            memory_type=entry.memory_type,
            confidence=entry.confidence,
            importance=entry.importance,
            evidence_strength=entry.evidence_strength,
            usage_count=entry.usage_count + 1,
            source=entry.source,
            tags=entry.tags,
            task_context=entry.task_context,
            strategy_id=entry.strategy_id,
            created_at=entry.created_at,
            last_accessed=datetime.now(UTC),
            last_modified=entry.last_modified,
        )
        self._entries[entry_id] = new_entry
        return new_entry

    def all_entries(self) -> list[MemoryEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: e.quality_score,
            reverse=True,
        )

    def detect_contradictions(self) -> list[Contradiction]:
        """Scan entries for contradictions.

        Simplified detection: looks for entries with opposite sentiment
        on the same topic (same tags or strategy_id) and different
        outcomes.
        """
        contradictions: list[Contradiction] = []

        entries = list(self._entries.values())
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                contradiction = self._check_contradiction(a, b)
                if contradiction is not None:
                    contradictions.append(contradiction)

        self._contradictions = contradictions
        if contradictions:
            self._log.info(
                "contradictions_detected",
                count=len(contradictions),
            )
        return contradictions

    def find_duplicates(self) -> list[tuple[str, str]]:
        """Find pairs of entries that are likely duplicates."""
        entries = list(self._entries.values())
        duplicates: list[tuple[str, str]] = []

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                if self._is_likely_duplicate(a, b):
                    duplicates.append((a.entry_id, b.entry_id))

        return duplicates

    def find_stale_entries(self) -> list[MemoryEntry]:
        """Find entries that have decayed below the freshness threshold."""
        return [e for e in self._entries.values() if e.freshness < self._freshness_threshold]

    def find_low_quality(self) -> list[MemoryEntry]:
        """Find entries with low quality scores."""
        return [e for e in self._entries.values() if e.quality_score < self._confidence_threshold]

    def compute_quality_report(self) -> QualityReport:
        """Compute a comprehensive quality report."""
        entries = list(self._entries.values())
        if not entries:
            return QualityReport()

        total = len(entries)
        avg_confidence = sum(e.confidence for e in entries) / total
        avg_importance = sum(e.importance for e in entries) / total
        avg_freshness = sum(e.freshness for e in entries) / total
        avg_quality = sum(e.quality_score for e in entries) / total

        low_quality = len(self.find_low_quality())
        stale = len(self.find_stale_entries())
        duplicates = len(self.find_duplicates())
        contradictions = len(self._contradictions)

        # Type distribution
        type_dist: dict[str, int] = {}
        for e in entries:
            type_dist[e.memory_type] = type_dist.get(e.memory_type, 0) + 1

        # Recommendations
        recommendations: list[str] = []
        if low_quality > total * 0.2:
            recommendations.append(
                f"Consider removing {low_quality} low-quality entries "
                f"({low_quality / total:.0%} of total)."
            )
        if stale > total * 0.3:
            recommendations.append(f"{stale} entries are stale. Consider refreshing or archiving.")
        if duplicates > 0:
            recommendations.append(
                f"{duplicates} duplicate pair(s) detected. Consider deduplication."
            )
        if contradictions > 0:
            recommendations.append(
                f"{contradictions} contradiction(s) detected. "
                f"Investigate whether different contexts explain them."
            )
        if avg_confidence < 0.4:
            recommendations.append(
                "Overall confidence is low. Strengthen evidence for key memories."
            )

        return QualityReport(
            total_entries=total,
            avg_confidence=avg_confidence,
            avg_importance=avg_importance,
            avg_freshness=avg_freshness,
            avg_quality_score=avg_quality,
            low_quality_count=low_quality,
            stale_count=stale,
            duplicate_count=duplicates,
            contradiction_count=contradictions,
            type_distribution=type_dist,
            recommendations=tuple(recommendations),
        )

    def prune_low_quality(self, max_prune: int = 50) -> list[str]:
        """Remove the lowest-quality entries. Returns IDs of pruned entries."""
        low = sorted(
            self.find_low_quality(),
            key=lambda e: e.quality_score,
        )
        pruned: list[str] = []
        for entry in low[:max_prune]:
            del self._entries[entry.entry_id]
            pruned.append(entry.entry_id)
        if pruned:
            self._log.info("entries_pruned", count=len(pruned))
        return pruned

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_contradiction(self, a: MemoryEntry, b: MemoryEntry) -> Contradiction | None:
        """Check if two entries contradict each other."""
        # Must share some context to be a meaningful contradiction
        shared_tags = set(a.tags) & set(b.tags)
        same_strategy = a.strategy_id and a.strategy_id == b.strategy_id
        same_type = a.memory_type == b.memory_type and a.memory_type != ""

        if not (shared_tags or same_strategy or same_type):
            return None

        # Check for sentiment/opposite indicators
        a_lower = a.content.lower()
        b_lower = b.content.lower()

        positive_words = {"success", "works", "effective", "good", "improved", "reliable"}
        negative_words = {"fail", "broken", "ineffective", "bad", "degraded", "unreliable"}

        a_positive = sum(1 for w in positive_words if w in a_lower)
        a_negative = sum(1 for w in negative_words if w in a_lower)
        b_positive = sum(1 for w in positive_words if w in b_lower)
        b_negative = sum(1 for w in negative_words if w in b_lower)

        # Contradiction: one is positive, other is negative
        if a_positive > a_negative and b_negative > b_positive:
            reasons = self._infer_contradiction_reasons(a, b)
            return Contradiction(
                entry_a_id=a.entry_id,
                entry_b_id=b.entry_id,
                entry_a_content=a.content[:200],
                entry_b_content=b.content[:200],
                explanation=(
                    f"Entry A is positive ({a_positive} pos indicators) "
                    f"while Entry B is negative ({b_negative} neg indicators)."
                ),
                possible_reasons=reasons,
                confidence=min(abs(a_positive - b_negative) / 5.0, 0.9),
            )
        if a_negative > a_positive and b_positive > b_negative:
            reasons = self._infer_contradiction_reasons(a, b)
            return Contradiction(
                entry_a_id=a.entry_id,
                entry_b_id=b.entry_id,
                entry_a_content=a.content[:200],
                entry_b_content=b.content[:200],
                explanation=(
                    f"Entry A is negative ({a_negative} neg indicators) "
                    f"while Entry B is positive ({b_positive} pos indicators)."
                ),
                possible_reasons=reasons,
                confidence=min(abs(b_positive - a_negative) / 5.0, 0.9),
            )

        return None

    @staticmethod
    def _infer_contradiction_reasons(a: MemoryEntry, b: MemoryEntry) -> tuple[str, ...]:
        reasons: list[str] = []
        if a.task_context != b.task_context and a.task_context and b.task_context:
            reasons.append("Different task contexts")
        if a.source != b.source and a.source and b.source:
            reasons.append("Different sources")
        age_diff = abs((a.created_at - b.created_at).total_seconds())
        if age_diff > 86400:  # > 1 day
            reasons.append("Created at different times")
        if not reasons:
            reasons.append("Context unclear — needs investigation")
        return tuple(reasons)

    def _is_likely_duplicate(self, a: MemoryEntry, b: MemoryEntry) -> bool:
        """Simple duplicate detection based on content similarity."""
        if a.memory_type != b.memory_type:
            return False
        if a.strategy_id and a.strategy_id == b.strategy_id:
            return True

        # Simple word overlap
        words_a = set(a.content.lower().split())
        words_b = set(b.content.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
        return overlap >= self._duplicate_similarity_threshold


__all__ = [
    "Contradiction",
    "MemoryEntry",
    "MemoryQualityController",
    "QualityReport",
]
