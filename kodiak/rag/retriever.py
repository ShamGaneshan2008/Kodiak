"""
In-memory retrieval over precomputed Kodiak V3 repository chunk embeddings.

The retriever consumes ``ChunkEmbedding`` objects produced by
``kodiak.rag.embeddings`` from chunks created by ``kodiak.rag.chunking``. It
does not parse files, rescan repositories, compute embeddings, call an LLM,
build prompts, or modify repository files.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from kodiak.rag.chunking import ChunkSymbolType, RepositoryChunk
from kodiak.rag.dependency_graph import DependencyGraph

if TYPE_CHECKING:
    from kodiak.rag.embeddings import ChunkEmbedding
else:

    @runtime_checkable
    class ChunkEmbedding(Protocol):
        """Runtime shape consumed by the in-memory retriever."""

        chunk: RepositoryChunk
        embedding: tuple[float, ...] | list[float]
        provider: Any
        model: str
        dimensions: int
        metadata: dict[str, Any]


logger = structlog.get_logger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
CONTEXT_SYMBOL_TYPES = frozenset(
    {
        ChunkSymbolType.MODULE,
        ChunkSymbolType.CLASS,
        ChunkSymbolType.FUNCTION,
        ChunkSymbolType.ASYNC_FUNCTION,
    }
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for scoring, filtering, and ranking retrieval results."""

    top_k: int = 10
    min_score: float = 0.0
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    metadata_weight: float = 0.05
    deduplicate: bool = True
    normalize_scores: bool = True
    include_related_modules: bool = True
    related_module_limit: int = 8
    candidate_multiplier: int = 4
    max_candidate_pool: int = 500
    adaptive_retrieval: bool = True
    context_neighbor_window: int = 1
    same_file_symbol_limit: int = 3
    include_parent_context: bool = True
    dependency_weight: float = 0.08
    repository_metadata_weight: float = 0.07
    overlap_merge_gap_lines: int = 5

    def __post_init__(self) -> None:
        """Validate retrieval configuration values."""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        if self.vector_weight < 0.0:
            raise ValueError("vector_weight must be zero or greater")
        if self.keyword_weight < 0.0:
            raise ValueError("keyword_weight must be zero or greater")
        if self.metadata_weight < 0.0:
            raise ValueError("metadata_weight must be zero or greater")
        if self.related_module_limit <= 0:
            raise ValueError("related_module_limit must be greater than zero")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if self.max_candidate_pool <= 0:
            raise ValueError("max_candidate_pool must be greater than zero")
        if self.context_neighbor_window < 0:
            raise ValueError("context_neighbor_window must be zero or greater")
        if self.same_file_symbol_limit < 0:
            raise ValueError("same_file_symbol_limit must be zero or greater")
        if self.dependency_weight < 0.0:
            raise ValueError("dependency_weight must be zero or greater")
        if self.repository_metadata_weight < 0.0:
            raise ValueError("repository_metadata_weight must be zero or greater")
        if self.overlap_merge_gap_lines < 0:
            raise ValueError("overlap_merge_gap_lines must be zero or greater")


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional filters applied before scoring repository chunks."""

    module: str | None = None
    modules: frozenset[str] = field(default_factory=frozenset)
    file_path: str | Path | None = None
    file_paths: frozenset[str] = field(default_factory=frozenset)
    symbol: str | None = None
    symbols: frozenset[str] = field(default_factory=frozenset)
    symbol_type: ChunkSymbolType | str | None = None
    symbol_types: frozenset[ChunkSymbolType | str] = field(default_factory=frozenset)
    parent_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalQuery:
    """
    Query parameters for ``Retriever.retrieve``.

    ``query_embedding`` must be supplied by the caller when vector similarity is
    desired. The retriever intentionally does not compute embeddings.
    """

    text: str | None = None
    query_embedding: tuple[float, ...] | list[float] | None = None
    top_k: int | None = None
    min_score: float | None = None
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    vector_weight: float | None = None
    keyword_weight: float | None = None
    candidate_top_k: int | None = None
    include_context: bool = True


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked repository chunk returned by the retriever."""

    chunk: RepositoryChunk
    score: float
    vector_score: float | None
    keyword_score: float | None
    metadata_score: float
    rank: int
    source: str
    matched_terms: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Return the stable chunk identifier."""
        return self.chunk.id

    @property
    def location(self) -> str:
        """Return a compact file and line location string."""
        return f"{self.chunk.file_path}:{self.chunk.start_line}-{self.chunk.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this result."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(self.score, 6),
            "vector_score": None if self.vector_score is None else round(self.vector_score, 6),
            "keyword_score": None if self.keyword_score is None else round(self.keyword_score, 6),
            "metadata_score": round(self.metadata_score, 6),
            "rank": self.rank,
            "source": self.source,
            "matched_terms": list(self.matched_terms),
            "location": self.location,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RetrievalContext:
    """A primary result set plus graph-related neighboring chunks."""

    primary: tuple[RetrievalResult, ...]
    related: tuple[RetrievalResult, ...]

    @property
    def results(self) -> tuple[RetrievalResult, ...]:
        """Return primary results followed by related context results."""
        return (*self.primary, *self.related)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this context."""
        return {
            "primary": [result.to_dict() for result in self.primary],
            "related": [result.to_dict() for result in self.related],
        }


class Retriever:
    """
    Retrieve relevant repository chunks from precomputed chunk embeddings.

    Args:
        chunk_embeddings: Embedded chunks generated by ``EmbeddingService``.
        dependency_graph: Optional graph used by ``retrieve_related`` and
            ``retrieve_context`` to include neighboring modules.
        config: Default retrieval configuration.
    """

    def __init__(
        self,
        chunk_embeddings: Iterable[ChunkEmbedding],
        *,
        dependency_graph: DependencyGraph | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize an in-memory retrieval index."""
        self.config = config or RetrievalConfig()
        self.dependency_graph = dependency_graph
        self._records: tuple[ChunkEmbedding, ...] = tuple(chunk_embeddings)
        self._by_chunk_id: dict[str, ChunkEmbedding] = {
            record.chunk.id: record for record in self._records
        }
        self._by_module: dict[str, list[ChunkEmbedding]] = self._group_by_module(self._records)
        self._by_symbol: dict[str, list[ChunkEmbedding]] = self._group_by_symbol(self._records)
        self._by_file: dict[str, tuple[ChunkEmbedding, ...]] = self._group_by_file(self._records)
        self._token_cache: dict[str, tuple[str, ...]] = {
            record.chunk.id: self._tokens_for_record(record) for record in self._records
        }
        self._token_index: dict[str, tuple[ChunkEmbedding, ...]] = self._build_token_index()
        self._document_frequencies = self._build_document_frequencies(self._records)
        self._document_count = len(self._records)
        self._average_document_length_value = self._calculate_average_document_length()

        logger.info(
            "retriever_initialized",
            chunks=len(self._records),
            modules=len(self._by_module),
            symbols=len(self._by_symbol),
            files=len(self._by_file),
        )

    async def retrieve(self, query: RetrievalQuery | str) -> list[RetrievalResult]:
        """
        Retrieve top-ranked chunks through a multi-stage in-memory pipeline.

        Passing a plain string performs keyword retrieval only. Passing a
        ``RetrievalQuery`` with ``query_embedding`` enables vector scoring using
        the already-computed query embedding supplied by the caller.
        """
        retrieval_query = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        try:
            query_tokens = self._tokenize(retrieval_query.text or "")
            candidates = self._initial_candidates(retrieval_query, query_tokens)
            filtered = self._filter_candidates(candidates, retrieval_query.filters)
            scored = self._score_records(filtered, retrieval_query, query_tokens=query_tokens)
            ranked = self._rank_candidates(scored, retrieval_query)
            selected = self._select_final_context(ranked, retrieval_query)
        except Exception as exc:
            logger.exception(
                "retrieval_pipeline_failed",
                error=str(exc),
                has_text=bool(retrieval_query.text),
                has_vector=retrieval_query.query_embedding is not None,
            )
            selected = []

        logger.debug(
            "retrieval_complete",
            candidates=len(candidates) if "candidates" in locals() else 0,
            filtered=len(filtered) if "filtered" in locals() else 0,
            scored=len(scored) if "scored" in locals() else 0,
            returned=len(selected),
            has_text=bool(retrieval_query.text),
            has_vector=retrieval_query.query_embedding is not None,
            top_k=self._top_k(retrieval_query),
        )
        return selected

    async def retrieve_top_k(
        self,
        query: str | None = None,
        *,
        query_embedding: tuple[float, ...] | list[float] | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Convenience wrapper for top-K text, vector, or hybrid retrieval."""
        return await self.retrieve(
            RetrievalQuery(
                text=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters or RetrievalFilters(),
            )
        )

    async def retrieve_by_module(
        self,
        module: str,
        *,
        query: str | None = None,
        query_embedding: tuple[float, ...] | list[float] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks scoped to one module."""
        return await self.retrieve(
            RetrievalQuery(
                text=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=RetrievalFilters(module=module),
            )
        )

    async def retrieve_by_symbol(
        self,
        symbol: str,
        *,
        query: str | None = None,
        query_embedding: tuple[float, ...] | list[float] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks matching a symbol name or qualified symbol suffix."""
        return await self.retrieve(
            RetrievalQuery(
                text=query or symbol,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=RetrievalFilters(symbol=symbol),
            )
        )

    async def retrieve_related(
        self,
        module: str,
        *,
        top_k: int | None = None,
        include_self: bool = False,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks from modules related through the dependency graph.

        Related modules include direct dependencies and direct dependents. The
        method does not compute semantic scores; it ranks module/class/function
        chunks deterministically by structural proximity.
        """
        modules = self._related_modules(module, include_self=include_self)
        filters = RetrievalFilters(modules=frozenset(modules))
        query = RetrievalQuery(top_k=top_k, filters=filters)
        try:
            candidates = self._filter_candidates(self._structural_candidates(filters), filters)
            scored = [
                self._structural_result(
                    record,
                    score=self._related_score(module, record.chunk.module_path),
                    source="related",
                )
                for record in candidates
            ]
            return self._rank_candidates(scored, query)[: self._top_k(query)]
        except Exception as exc:
            logger.warning("retrieve_related_failed", module=module, error=str(exc))
            return []

    async def retrieve_context(
        self,
        query: RetrievalQuery | str,
        *,
        related_top_k: int | None = None,
    ) -> RetrievalContext:
        """
        Retrieve primary results plus dependency-graph context around them."""
        retrieval_query = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        primary = tuple(await self.retrieve(retrieval_query))
        if not primary:
            return RetrievalContext(primary=primary, related=())

        related: tuple[RetrievalResult, ...] = ()
        try:
            related = self._context_results(
                primary,
                related_top_k=related_top_k or self.config.related_module_limit,
            )
        except Exception as exc:
            logger.warning("retrieval_context_expansion_failed", error=str(exc))

        return RetrievalContext(primary=primary, related=related)

    def _initial_candidates(
        self,
        query: RetrievalQuery,
        query_tokens: tuple[str, ...],
    ) -> tuple[ChunkEmbedding, ...]:
        """
        Return a bounded candidate pool before precise scoring.

        The retriever has no external ANN index, so vector-only searches still
        need to inspect matching records. Text and structural filters use local
        indexes to reduce work before vector similarity is computed.
        """
        structural = self._structural_candidates(query.filters)
        if not query_tokens:
            return self._limit_candidates(structural, query)

        lexical = self._lexical_candidates(query_tokens, structural)
        if lexical:
            if query.query_embedding is not None and len(lexical) < self._candidate_limit(query):
                lexical_ids = {record.chunk.id for record in lexical}
                lexical = (
                    *lexical,
                    *(record for record in structural if record.chunk.id not in lexical_ids),
                )
            return self._limit_candidates(lexical, query)
        return self._limit_candidates(structural, query)

    def _filter_candidates(
        self,
        records: Iterable[ChunkEmbedding],
        filters: RetrievalFilters,
    ) -> tuple[ChunkEmbedding, ...]:
        try:
            return self._apply_filters(records, filters)
        except Exception as exc:
            logger.warning("retrieval_filtering_failed", error=str(exc))
            return tuple(records)

    def _rank_candidates(
        self,
        results: list[RetrievalResult],
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        try:
            return self._rank(
                results,
                top_k=self._expanded_top_k(query),
                min_score=self._min_score(query),
            )
        except Exception as exc:
            logger.warning("retrieval_ranking_failed", error=str(exc), scored=len(results))
            return self._safe_rank(results, top_k=self._expanded_top_k(query))

    def _safe_rank(self, results: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        ordered = sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
        return [
            self._replace_rank(result, rank=index) for index, result in enumerate(ordered, start=1)
        ]

    def _select_final_context(
        self,
        ranked: list[RetrievalResult],
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Trim ranked candidates while preserving useful context diversity."""
        top_k = self._top_k(query)
        if not ranked:
            return []

        merged = self._merge_overlapping_results(ranked)
        if not query.include_context:
            return [
                self._replace_rank(result, rank=index)
                for index, result in enumerate(merged[:top_k], start=1)
            ]

        selected: list[RetrievalResult] = []
        seen: set[str] = set()
        files_seen: Counter[str] = Counter()
        for result in merged:
            file_key = result.chunk.file_path.as_posix()
            if result.chunk.id in seen:
                continue
            if files_seen[file_key] >= max(self.config.same_file_symbol_limit, 1):
                if len(selected) >= top_k:
                    continue
            selected.append(result)
            seen.add(result.chunk.id)
            files_seen[file_key] += 1
            if len(selected) >= top_k:
                break

        return [
            self._replace_rank(result, rank=index) for index, result in enumerate(selected, start=1)
        ]

    def _context_results(
        self,
        primary: tuple[RetrievalResult, ...],
        *,
        related_top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        primary_ids = {result.chunk.id for result in primary}
        context: list[RetrievalResult] = []
        seen = set(primary_ids)

        for result in primary:
            for record, source, score in self._same_file_context_records(result.chunk):
                if record.chunk.id in seen:
                    continue
                context.append(self._structural_result(record, score=score, source=source))
                seen.add(record.chunk.id)

        if self.dependency_graph is not None and self.config.include_related_modules:
            related_modules: set[str] = set()
            for result in primary:
                related_modules.update(
                    self._related_modules(result.chunk.module_path, include_self=False)
                )

            filters = RetrievalFilters(modules=frozenset(related_modules))
            related_records = self._filter_candidates(self._structural_candidates(filters), filters)
            for record in related_records:
                if record.chunk.id in seen:
                    continue
                origin = self._closest_primary_module(record.chunk.module_path, primary)
                context.append(
                    self._structural_result(
                        record,
                        score=self._related_score(origin, record.chunk.module_path),
                        source="dependency_context",
                    )
                )
                seen.add(record.chunk.id)

        ranked = self._rank(
            context,
            top_k=related_top_k,
            min_score=0.0,
        )
        return tuple(ranked)

    def _same_file_context_records(
        self,
        chunk: RepositoryChunk,
    ) -> tuple[tuple[ChunkEmbedding, str, float], ...]:
        records = self._by_file.get(chunk.file_path.as_posix(), ())
        if not records:
            return ()

        context: list[tuple[ChunkEmbedding, str, float]] = []
        if self.config.include_parent_context:
            parent = self._parent_context_record(chunk, records)
            if parent is not None:
                context.append((parent, "parent_context", 0.72))

        if self.config.context_neighbor_window > 0:
            context.extend(
                (record, "neighbor_context", 0.62)
                for record in self._neighbor_records(chunk, records)
            )

        if self.config.same_file_symbol_limit > 0:
            context.extend(
                (record, "same_file_symbol_context", 0.58)
                for record in self._same_file_symbol_records(chunk, records)
            )

        return tuple(context)

    def _structural_candidates(self, filters: RetrievalFilters) -> tuple[ChunkEmbedding, ...]:
        pools: list[Iterable[ChunkEmbedding]] = []
        if filters.module:
            pools.append(self._by_module.get(filters.module, ()))
        if filters.modules:
            pools.extend(self._by_module.get(module, ()) for module in filters.modules)
        if filters.file_path:
            pools.append(self._records_for_file(filters.file_path))
        if filters.file_paths:
            pools.extend(self._records_for_file(file_path) for file_path in filters.file_paths)
        if filters.symbol:
            pools.append(self._records_for_symbol(filters.symbol))
        if filters.symbols:
            pools.extend(self._records_for_symbol(symbol) for symbol in filters.symbols)

        if not pools:
            return self._records

        records: dict[str, ChunkEmbedding] = {}
        for pool in pools:
            for record in pool:
                records[record.chunk.id] = record
        return tuple(records.values())

    def _lexical_candidates(
        self,
        query_tokens: tuple[str, ...],
        records: tuple[ChunkEmbedding, ...],
    ) -> tuple[ChunkEmbedding, ...]:
        if not query_tokens:
            return ()

        allowed = {record.chunk.id for record in records}
        weighted: dict[str, tuple[int, ChunkEmbedding]] = {}
        for token in dict.fromkeys(query_tokens):
            for record in self._token_index.get(token, ()):
                if record.chunk.id not in allowed:
                    continue
                count, _ = weighted.get(record.chunk.id, (0, record))
                weighted[record.chunk.id] = (count + 1, record)

        return tuple(
            record
            for _, record in sorted(
                weighted.values(),
                key=lambda item: (
                    item[0],
                    self._type_priority(item[1].chunk.symbol_type),
                    -item[1].chunk.start_line,
                ),
                reverse=True,
            )
        )

    def _parent_context_record(
        self,
        chunk: RepositoryChunk,
        records: tuple[ChunkEmbedding, ...],
    ) -> ChunkEmbedding | None:
        if not chunk.parent_class:
            return None
        for record in records:
            candidate = record.chunk
            if candidate.symbol_type != ChunkSymbolType.CLASS:
                continue
            if candidate.symbol_name == chunk.parent_class or candidate.symbol_name.endswith(
                f".{chunk.parent_class}"
            ):
                return record
        return None

    def _neighbor_records(
        self,
        chunk: RepositoryChunk,
        records: tuple[ChunkEmbedding, ...],
    ) -> tuple[ChunkEmbedding, ...]:
        try:
            index = next(idx for idx, record in enumerate(records) if record.chunk.id == chunk.id)
        except StopIteration:
            return ()

        window = self.config.context_neighbor_window
        start = max(index - window, 0)
        end = min(index + window + 1, len(records))
        return tuple(
            record
            for record in records[start:end]
            if record.chunk.id != chunk.id and record.chunk.symbol_type in CONTEXT_SYMBOL_TYPES
        )

    def _same_file_symbol_records(
        self,
        chunk: RepositoryChunk,
        records: tuple[ChunkEmbedding, ...],
    ) -> tuple[ChunkEmbedding, ...]:
        selected: list[ChunkEmbedding] = []
        for record in records:
            candidate = record.chunk
            if candidate.id == chunk.id or candidate.symbol_type not in CONTEXT_SYMBOL_TYPES:
                continue
            if self._overlaps_or_touches(candidate, chunk):
                continue
            if candidate.parent_class and candidate.parent_class == chunk.parent_class:
                selected.append(record)
            elif candidate.symbol_type in {
                ChunkSymbolType.CLASS,
                ChunkSymbolType.FUNCTION,
                ChunkSymbolType.ASYNC_FUNCTION,
            }:
                selected.append(record)
            if len(selected) >= self.config.same_file_symbol_limit:
                break
        return tuple(selected)

    def _records_for_file(self, file_path: str | Path) -> tuple[ChunkEmbedding, ...]:
        expected = Path(file_path).as_posix()
        direct = self._by_file.get(expected)
        if direct is not None:
            return direct
        return tuple(
            record
            for path, records in self._by_file.items()
            if path == expected or path.endswith(expected)
            for record in records
        )

    def _records_for_symbol(self, symbol: str) -> tuple[ChunkEmbedding, ...]:
        direct = self._by_symbol.get(symbol)
        if direct is not None:
            return tuple(direct)
        return tuple(
            record
            for name, records in self._by_symbol.items()
            if self._symbol_matches(name, symbol)
            for record in records
        )

    def _limit_candidates(
        self,
        records: tuple[ChunkEmbedding, ...],
        query: RetrievalQuery,
    ) -> tuple[ChunkEmbedding, ...]:
        limit = self._candidate_limit(query)
        if len(records) <= limit:
            return records
        return records[:limit]

    def _candidate_limit(self, query: RetrievalQuery) -> int:
        if query.candidate_top_k is not None:
            return max(query.candidate_top_k, self._top_k(query))

        top_k = self._top_k(query)
        if not self.config.adaptive_retrieval:
            return min(self.config.max_candidate_pool, top_k * self.config.candidate_multiplier)

        query_tokens = self._tokenize(query.text or "")
        complexity = len(set(query_tokens))
        if query.query_embedding is not None:
            complexity += 4
        if query.filters.metadata:
            complexity += len(query.filters.metadata)
        multiplier = self.config.candidate_multiplier + min(complexity // 4, 4)
        return min(self.config.max_candidate_pool, max(top_k, top_k * multiplier))

    def _expanded_top_k(self, query: RetrievalQuery) -> int:
        top_k = self._top_k(query)
        if not query.include_context:
            return top_k
        return min(
            self.config.max_candidate_pool,
            top_k + self.config.context_neighbor_window * 2 + self.config.same_file_symbol_limit,
        )

    def _score_records(
        self,
        records: tuple[ChunkEmbedding, ...],
        query: RetrievalQuery,
        *,
        query_tokens: tuple[str, ...] | None = None,
    ) -> list[RetrievalResult]:
        if query_tokens is None:
            query_tokens = self._tokenize(query.text or "")
        query_embedding = self._as_float_tuple(query.query_embedding)
        vector_weight = self._vector_weight(query, query_embedding)
        keyword_weight = self._keyword_weight(query, query_tokens)
        total_weight = vector_weight + keyword_weight + self.config.metadata_weight
        if total_weight <= 0.0:
            total_weight = 1.0

        results: list[RetrievalResult] = []
        for record in records:
            try:
                vector_score = (
                    self._cosine_similarity(query_embedding, record.embedding)
                    if query_embedding is not None
                    else None
                )
                keyword_score, matched_terms = (
                    self._keyword_score(query_tokens, record) if query_tokens else (None, ())
                )
                metadata_score = self._metadata_score(query.filters, record.chunk)
                repository_score = self._repository_score(record.chunk)
                dependency_score = self._dependency_score(query.filters, record.chunk)
                combined = (
                    ((vector_score or 0.0) * vector_weight)
                    + ((keyword_score or 0.0) * keyword_weight)
                    + (metadata_score * self.config.metadata_weight)
                    + (repository_score * self.config.repository_metadata_weight)
                    + (dependency_score * self.config.dependency_weight)
                ) / (
                    total_weight
                    + self.config.repository_metadata_weight
                    + self.config.dependency_weight
                )

                source = self._source(vector_score, keyword_score)
                results.append(
                    RetrievalResult(
                        chunk=record.chunk,
                        score=self._clamp_score(combined),
                        vector_score=vector_score,
                        keyword_score=keyword_score,
                        metadata_score=metadata_score,
                        rank=0,
                        source=source,
                        matched_terms=matched_terms,
                        metadata={
                            "embedding_provider": self._provider_name(record),
                            "embedding_model": record.model,
                            "embedding_dimensions": record.dimensions,
                            "repository_score": round(repository_score, 6),
                            "dependency_score": round(dependency_score, 6),
                            **record.metadata,
                        },
                    )
                )
            except Exception as exc:
                logger.warning(
                    "retrieval_record_scoring_failed",
                    chunk_id=getattr(getattr(record, "chunk", None), "id", None),
                    error=str(exc),
                )
        return results

    def _rank(
        self,
        results: list[RetrievalResult],
        *,
        top_k: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        deduped = self._deduplicate(results) if self.config.deduplicate else results
        normalized = self._normalize_scores(deduped) if self.config.normalize_scores else deduped
        filtered = [result for result in normalized if result.score >= min_score]
        ordered = sorted(
            filtered,
            key=lambda result: (
                result.score,
                result.metadata.get("dependency_score", 0.0),
                result.metadata.get("repository_score", 0.0),
                self._type_priority(result.chunk.symbol_type),
                -result.chunk.start_line,
                result.chunk.symbol_name,
            ),
            reverse=True,
        )[:top_k]
        return [
            self._replace_rank(result, rank=index) for index, result in enumerate(ordered, start=1)
        ]

    def _apply_filters(
        self,
        records: Iterable[ChunkEmbedding],
        filters: RetrievalFilters,
    ) -> tuple[ChunkEmbedding, ...]:
        return tuple(record for record in records if self._passes_filters(record.chunk, filters))

    def _passes_filters(self, chunk: RepositoryChunk, filters: RetrievalFilters) -> bool:
        if filters.module and chunk.module_path != filters.module:
            return False
        if filters.modules and chunk.module_path not in filters.modules:
            return False
        if filters.file_path and not self._path_matches(chunk.file_path, filters.file_path):
            return False
        if filters.file_paths and not any(
            self._path_matches(chunk.file_path, file_path) for file_path in filters.file_paths
        ):
            return False
        if filters.symbol and not self._symbol_matches(chunk.symbol_name, filters.symbol):
            return False
        if filters.symbols and not any(
            self._symbol_matches(chunk.symbol_name, symbol) for symbol in filters.symbols
        ):
            return False
        if filters.symbol_type and chunk.symbol_type != self._symbol_type(filters.symbol_type):
            return False
        if filters.symbol_types:
            allowed = {self._symbol_type(symbol_type) for symbol_type in filters.symbol_types}
            if chunk.symbol_type not in allowed:
                return False
        if filters.parent_class and chunk.parent_class != filters.parent_class:
            return False
        return self._metadata_matches(chunk.metadata, filters.metadata)

    def _keyword_score(
        self,
        query_tokens: tuple[str, ...],
        record: ChunkEmbedding,
    ) -> tuple[float, tuple[str, ...]]:
        document_tokens = self._record_tokens(record)
        if not document_tokens:
            return 0.0, ()

        counts = Counter(document_tokens)
        unique_query_tokens = tuple(dict.fromkeys(query_tokens))
        matched_terms = tuple(token for token in unique_query_tokens if token in counts)
        if not matched_terms:
            return 0.0, ()

        score = 0.0
        average_length = max(self._average_document_length(), 1.0)
        document_length = len(document_tokens)
        k1 = 1.2
        b = 0.75
        for token in matched_terms:
            frequency = counts[token]
            document_frequency = self._document_frequencies.get(token, 0)
            inverse_document_frequency = math.log(
                1.0 + (self._document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * document_length / average_length)
            score += inverse_document_frequency * ((frequency * (k1 + 1.0)) / denominator)

        normalized = score / (score + 1.0)
        return self._clamp_score(normalized), matched_terms

    def _metadata_score(self, filters: RetrievalFilters, chunk: RepositoryChunk) -> float:
        checks = 0
        matches = 0
        if filters.module:
            checks += 1
            matches += int(chunk.module_path == filters.module)
        if filters.modules:
            checks += 1
            matches += int(chunk.module_path in filters.modules)
        if filters.file_path:
            checks += 1
            matches += int(self._path_matches(chunk.file_path, filters.file_path))
        if filters.file_paths:
            checks += 1
            matches += int(
                any(self._path_matches(chunk.file_path, path) for path in filters.file_paths)
            )
        if filters.symbol:
            checks += 1
            matches += int(self._symbol_matches(chunk.symbol_name, filters.symbol))
        if filters.symbols:
            checks += 1
            matches += int(
                any(self._symbol_matches(chunk.symbol_name, symbol) for symbol in filters.symbols)
            )
        if filters.symbol_type:
            checks += 1
            matches += int(chunk.symbol_type == self._symbol_type(filters.symbol_type))
        if filters.symbol_types:
            allowed = {self._symbol_type(symbol_type) for symbol_type in filters.symbol_types}
            checks += 1
            matches += int(chunk.symbol_type in allowed)
        if filters.parent_class:
            checks += 1
            matches += int(chunk.parent_class == filters.parent_class)
        for key, expected in filters.metadata.items():
            checks += 1
            matches += int(chunk.metadata.get(key) == expected)
        if checks == 0:
            return 0.0
        return matches / checks

    def _repository_score(self, chunk: RepositoryChunk) -> float:
        metadata = chunk.metadata
        line_count = float(
            metadata.get("line_count") or max(chunk.end_line - chunk.start_line + 1, 1)
        )
        class_count = float(metadata.get("class_count") or 0)
        function_count = float(metadata.get("function_count") or 0)
        import_count = float(metadata.get("import_count") or 0)
        public_symbol = not chunk.symbol_name.rsplit(".", maxsplit=1)[-1].startswith("_")

        score = 0.0
        score += min(line_count / 250.0, 0.3)
        score += min((class_count + function_count) / 30.0, 0.3)
        score += min(import_count / 30.0, 0.15)
        score += 0.15 if public_symbol else 0.0
        if chunk.file_path.name in {"__init__.py", "main.py", "app.py"}:
            score += 0.1
        return self._clamp_score(score)

    def _dependency_score(self, filters: RetrievalFilters, chunk: RepositoryChunk) -> float:
        if self.dependency_graph is None:
            return 0.0

        origin_modules = set(filters.modules)
        if filters.module:
            origin_modules.add(filters.module)
        if not origin_modules:
            return 0.0

        scores = [
            self._related_score(origin, chunk.module_path)
            for origin in origin_modules
            if origin != chunk.module_path
        ]
        if chunk.module_path in origin_modules:
            scores.append(1.0)
        return max(scores, default=0.0)

    def _merge_overlapping_results(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not self.config.deduplicate or not results:
            return results

        selected: list[RetrievalResult] = []
        for result in results:
            duplicate_index = self._overlapping_result_index(selected, result)
            if duplicate_index is None:
                selected.append(result)
                continue

            current = selected[duplicate_index]
            if result.score > current.score:
                selected[duplicate_index] = self._merge_result_metadata(result, current)
            else:
                selected[duplicate_index] = self._merge_result_metadata(current, result)
        return selected

    def _overlapping_result_index(
        self,
        selected: list[RetrievalResult],
        result: RetrievalResult,
    ) -> int | None:
        for index, current in enumerate(selected):
            if current.chunk.file_path != result.chunk.file_path:
                continue
            if self._overlaps_or_touches(current.chunk, result.chunk):
                return index
        return None

    def _overlaps_or_touches(
        self,
        left: RepositoryChunk,
        right: RepositoryChunk,
    ) -> bool:
        gap = self.config.overlap_merge_gap_lines
        return left.start_line <= right.end_line + gap and right.start_line <= left.end_line + gap

    @staticmethod
    def _merge_result_metadata(
        primary: RetrievalResult,
        duplicate: RetrievalResult,
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk=primary.chunk,
            score=primary.score,
            vector_score=primary.vector_score,
            keyword_score=primary.keyword_score,
            metadata_score=primary.metadata_score,
            rank=primary.rank,
            source=primary.source,
            matched_terms=tuple(dict.fromkeys((*primary.matched_terms, *duplicate.matched_terms))),
            metadata={
                **primary.metadata,
                "merged_duplicate_chunk_ids": tuple(
                    dict.fromkeys(
                        (
                            *primary.metadata.get("merged_duplicate_chunk_ids", ()),
                            duplicate.chunk.id,
                        )
                    )
                ),
            },
        )

    def _related_modules(self, module: str, *, include_self: bool) -> set[str]:
        modules = {module} if include_self else set()
        if self.dependency_graph is None:
            return modules
        modules.update(self.dependency_graph.get_dependencies(module))
        modules.update(self.dependency_graph.get_dependents(module))
        return modules

    def _related_score(self, origin_module: str, candidate_module: str) -> float:
        if origin_module == candidate_module:
            return 1.0
        if self.dependency_graph is None:
            return 0.5
        if candidate_module in self.dependency_graph.get_dependencies(origin_module):
            return 0.8
        if candidate_module in self.dependency_graph.get_dependents(origin_module):
            return 0.7
        return 0.5

    def _closest_primary_module(
        self,
        candidate_module: str,
        primary: tuple[RetrievalResult, ...],
    ) -> str:
        if not primary:
            return candidate_module
        return max(
            (result.chunk.module_path for result in primary),
            key=lambda module: self._related_score(module, candidate_module),
        )

    def _structural_result(
        self,
        record: ChunkEmbedding,
        *,
        score: float,
        source: str = "related",
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk=record.chunk,
            score=self._clamp_score(score),
            vector_score=None,
            keyword_score=None,
            metadata_score=0.0,
            rank=0,
            source=source,
            metadata={
                "embedding_provider": self._provider_name(record),
                "embedding_model": record.model,
                "embedding_dimensions": record.dimensions,
                "repository_score": round(self._repository_score(record.chunk), 6),
                "dependency_score": 0.0,
                **record.metadata,
            },
        )

    @staticmethod
    def _provider_name(record: ChunkEmbedding) -> str:
        provider = getattr(record, "provider", "")
        return str(getattr(provider, "value", provider))

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        best_by_id: dict[str, RetrievalResult] = {}
        for result in results:
            current = best_by_id.get(result.chunk.id)
            if current is None or result.score > current.score:
                best_by_id[result.chunk.id] = result
        return list(best_by_id.values())

    def _normalize_scores(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []
        min_value = min(result.score for result in results)
        max_value = max(result.score for result in results)
        if math.isclose(min_value, max_value):
            return results
        return [
            self._replace_score(result, (result.score - min_value) / (max_value - min_value))
            for result in results
        ]

    def _record_tokens(self, record: ChunkEmbedding) -> tuple[str, ...]:
        return self._token_cache.get(record.chunk.id, ())

    def _tokens_for_record(self, record: ChunkEmbedding) -> tuple[str, ...]:
        chunk = record.chunk
        text = " ".join(
            (
                chunk.module_path,
                chunk.symbol_name,
                chunk.symbol_type.value,
                chunk.parent_class or "",
                chunk.docstring or "",
                " ".join(chunk.imports),
                chunk.source_code,
            )
        )
        return self._tokenize(text)

    def _build_document_frequencies(
        self,
        records: tuple[ChunkEmbedding, ...],
    ) -> dict[str, int]:
        frequencies: dict[str, int] = defaultdict(int)
        for record in records:
            for token in set(self._record_tokens(record)):
                frequencies[token] += 1
        return dict(frequencies)

    def _average_document_length(self) -> float:
        return self._average_document_length_value

    def _calculate_average_document_length(self) -> float:
        if not self._records:
            return 0.0
        total = sum(len(tokens) for tokens in self._token_cache.values())
        return total / len(self._records)

    def _top_k(self, query: RetrievalQuery) -> int:
        return query.top_k or self.config.top_k

    def _min_score(self, query: RetrievalQuery) -> float:
        return self.config.min_score if query.min_score is None else query.min_score

    def _vector_weight(
        self,
        query: RetrievalQuery,
        query_embedding: tuple[float, ...] | None,
    ) -> float:
        if query_embedding is None:
            return 0.0
        return self.config.vector_weight if query.vector_weight is None else query.vector_weight

    def _keyword_weight(
        self,
        query: RetrievalQuery,
        query_tokens: tuple[str, ...],
    ) -> float:
        if not query_tokens:
            return 0.0
        return self.config.keyword_weight if query.keyword_weight is None else query.keyword_weight

    @staticmethod
    def _group_by_module(records: tuple[ChunkEmbedding, ...]) -> dict[str, list[ChunkEmbedding]]:
        grouped: dict[str, list[ChunkEmbedding]] = defaultdict(list)
        for record in records:
            grouped[record.chunk.module_path].append(record)
        return dict(grouped)

    @staticmethod
    def _group_by_symbol(records: tuple[ChunkEmbedding, ...]) -> dict[str, list[ChunkEmbedding]]:
        grouped: dict[str, list[ChunkEmbedding]] = defaultdict(list)
        for record in records:
            grouped[record.chunk.symbol_name].append(record)
        return dict(grouped)

    @staticmethod
    def _group_by_file(
        records: tuple[ChunkEmbedding, ...],
    ) -> dict[str, tuple[ChunkEmbedding, ...]]:
        grouped: dict[str, list[ChunkEmbedding]] = defaultdict(list)
        for record in records:
            grouped[record.chunk.file_path.as_posix()].append(record)
        return {
            file_path: tuple(sorted(file_records, key=lambda item: item.chunk.start_line))
            for file_path, file_records in grouped.items()
        }

    def _build_token_index(self) -> dict[str, tuple[ChunkEmbedding, ...]]:
        grouped: dict[str, list[ChunkEmbedding]] = defaultdict(list)
        for record in self._records:
            for token in set(self._record_tokens(record)):
                grouped[token].append(record)
        return {token: tuple(records) for token, records in grouped.items()}

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))

    @staticmethod
    def _as_float_tuple(values: tuple[float, ...] | list[float] | None) -> tuple[float, ...] | None:
        if values is None:
            return None
        return tuple(float(value) for value in values)

    @staticmethod
    def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = dot / (left_norm * right_norm)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    @staticmethod
    def _path_matches(actual: Path, expected: str | Path) -> bool:
        expected_text = Path(expected).as_posix()
        actual_text = actual.as_posix()
        return actual_text == expected_text or actual_text.endswith(expected_text)

    @staticmethod
    def _symbol_matches(actual: str, expected: str) -> bool:
        return actual == expected or actual.endswith(f".{expected}")

    @staticmethod
    def _symbol_type(value: ChunkSymbolType | str) -> ChunkSymbolType:
        return value if isinstance(value, ChunkSymbolType) else ChunkSymbolType(value)

    @staticmethod
    def _metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        for key, value in expected.items():
            actual_value = actual.get(key)
            if isinstance(value, (set, frozenset, tuple, list)):
                if actual_value not in value:
                    return False
            elif actual_value != value:
                return False
        return True

    @staticmethod
    def _source(vector_score: float | None, keyword_score: float | None) -> str:
        if vector_score is not None and keyword_score is not None:
            return "hybrid"
        if vector_score is not None:
            return "vector"
        if keyword_score is not None:
            return "keyword"
        return "metadata"

    @staticmethod
    def _type_priority(symbol_type: ChunkSymbolType) -> int:
        priorities = {
            ChunkSymbolType.CLASS: 4,
            ChunkSymbolType.FUNCTION: 3,
            ChunkSymbolType.ASYNC_FUNCTION: 3,
            ChunkSymbolType.MODULE: 2,
            ChunkSymbolType.CONSTANT_BLOCK: 1,
        }
        return priorities.get(symbol_type, 0)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _replace_score(result: RetrievalResult, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk=result.chunk,
            score=score,
            vector_score=result.vector_score,
            keyword_score=result.keyword_score,
            metadata_score=result.metadata_score,
            rank=result.rank,
            source=result.source,
            matched_terms=result.matched_terms,
            metadata=result.metadata,
        )

    @staticmethod
    def _replace_rank(result: RetrievalResult, rank: int) -> RetrievalResult:
        return RetrievalResult(
            chunk=result.chunk,
            score=result.score,
            vector_score=result.vector_score,
            keyword_score=result.keyword_score,
            metadata_score=result.metadata_score,
            rank=rank,
            source=result.source,
            matched_terms=result.matched_terms,
            metadata=result.metadata,
        )
