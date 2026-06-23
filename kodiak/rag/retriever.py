"""
kodiak/rag/retriever.py

Hybrid retrieval combining semantic vector search with BM25-style keyword
scoring and symbol-aware lookup. Supports query expansion and re-ranking.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from kodiak.rag.embedder import Embedder
from kodiak.rag.reranker import Reranker
from kodiak.rag.symbol_index import SymbolIndex, SymbolMatch
from kodiak.rag.vector_store import SearchResult, VectorStore

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedContext:
    """A single ranked context item returned to the orchestrator."""

    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    name: str | None
    score: float
    source: str  # "vector", "symbol", "hybrid"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "score": round(self.score, 4),
            "source": self.source,
        }

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class RetrievalQuery:
    text: str
    repo_ids: list[str]
    top_k: int = 10
    min_score: float = 0.3
    language_filter: str | None = None
    file_filter: str | None = None
    chunk_type_filter: str | None = None
    use_reranker: bool = True
    expand_query: bool = False


# ---------------------------------------------------------------------------
# Query expander
# ---------------------------------------------------------------------------


def _extract_symbols(query: str) -> list[str]:
    """Pull out identifiers that look like code symbols."""
    # camelCase, snake_case, PascalCase
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query)


def expand_query(original: str) -> list[str]:
    """
    Generate query variants to improve recall.
    E.g. "how does the auth middleware work" →
         ["auth middleware", "authentication middleware", "auth.py middleware"]
    """
    variants = [original]
    symbols = _extract_symbols(original)
    if symbols:
        variants.append(" ".join(symbols))
    # Add "implementation of X" pattern
    variants.append(f"implementation {original}")
    return list(dict.fromkeys(variants))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Hybrid retriever that combines:
    1. Semantic vector search (via VectorStore)
    2. Exact/fuzzy symbol lookup (via SymbolIndex)
    3. Optional neural re-ranking (via Reranker)

    Usage::

        ctx = await retriever.retrieve(RetrievalQuery(
            text="how does rate limiting work",
            repo_ids=["org/repo"],
            top_k=5,
        ))
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        symbol_index: SymbolIndex,
        reranker: Reranker | None = None,
        semantic_weight: float = 0.7,
        symbol_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.symbol_index = symbol_index
        self.reranker = reranker
        self.semantic_weight = semantic_weight
        self.symbol_weight = symbol_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedContext]:
        """
        Main retrieval entry point.

        Returns ranked list of RetrievedContext items.
        """
        queries = expand_query(query.text) if query.expand_query else [query.text]
        log = logger.bind(query=query.text, repos=query.repo_ids, top_k=query.top_k)

        # 1. Embed query (and expansions)
        embedding_results = await self.embedder.embed_many(queries)
        query_embeddings = [r.embedding for r in embedding_results]

        # 2. Vector search (primary query + expansions)
        vector_tasks = [
            self.vector_store.search_multi_repo(
                repo_ids=query.repo_ids,
                query_embedding=emb,
                top_k=query.top_k * 2,  # over-fetch, will deduplicate
            )
            for emb in query_embeddings
        ]
        symbol_task = self.symbol_index.search(
            repo_ids=query.repo_ids,
            query=query.text,
            top_k=query.top_k,
        )

        vector_results_per_query, symbol_matches = await asyncio.gather(
            asyncio.gather(*vector_tasks),
            symbol_task,
        )

        # 3. Flatten + deduplicate vector results
        seen_ids: set[str] = set()
        vector_results: list[SearchResult] = []
        for results in vector_results_per_query:
            for r in results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    vector_results.append(r)

        # 4. Build filters
        filters = self._build_filters(query)

        # 5. Merge semantic + symbol results
        merged = self._merge_results(
            vector_results=vector_results,
            symbol_matches=symbol_matches,
            filters=filters,
            min_score=query.min_score,
        )

        # 6. Optional re-ranking
        if query.use_reranker and self.reranker and merged:
            merged = await self.reranker.rerank(
                query=query.text,
                results=merged,
                top_k=query.top_k,
            )
        else:
            merged = sorted(merged, key=lambda r: r.score, reverse=True)[: query.top_k]

        log.debug("retrieval_done", returned=len(merged))
        return merged

    async def retrieve_for_file(
        self,
        repo_id: str,
        file_path: str,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """Retrieve chunks scoped to a specific file."""
        q = RetrievalQuery(
            text=query,
            repo_ids=[repo_id],
            top_k=top_k,
            file_filter=file_path,
            use_reranker=False,
        )
        return await self.retrieve(q)

    async def retrieve_symbols(
        self,
        repo_ids: list[str],
        symbol_name: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """Direct symbol lookup without semantic search."""
        matches = await self.symbol_index.search(
            repo_ids=repo_ids, query=symbol_name, top_k=top_k
        )
        return [self._symbol_to_context(m) for m in matches]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        symbol_matches: list[SymbolMatch],
        filters: dict[str, Any],
        min_score: float,
    ) -> list[RetrievedContext]:
        score_map: dict[str, float] = {}
        result_map: dict[str, RetrievedContext] = {}

        # Semantic results
        for r in vector_results:
            if not self._passes_filters(r, filters):
                continue
            score = r.score * self.semantic_weight
            if score < min_score:
                continue
            ctx = self._vector_to_context(r, score, source="vector")
            if r.chunk_id not in result_map or score > score_map[r.chunk_id]:
                score_map[r.chunk_id] = score
                result_map[r.chunk_id] = ctx

        # Symbol results (boost if also in vector results)
        for m in symbol_matches:
            sym_score = m.score * self.symbol_weight
            if m.chunk_id in result_map:
                # Boost existing result
                combined = score_map[m.chunk_id] + sym_score
                result_map[m.chunk_id].score = combined
                result_map[m.chunk_id].source = "hybrid"
                score_map[m.chunk_id] = combined
            else:
                ctx = self._symbol_to_context(m, score=sym_score)
                result_map[m.chunk_id] = ctx
                score_map[m.chunk_id] = sym_score

        return list(result_map.values())

    def _passes_filters(self, result: SearchResult, filters: dict[str, Any]) -> bool:
        if "language" in filters and result.language != filters["language"]:
            return False
        if "file_path" in filters and not result.file_path.endswith(filters["file_path"]):
            return False
        if "chunk_type" in filters and result.chunk_type != filters["chunk_type"]:
            return False
        return True

    def _build_filters(self, query: RetrievalQuery) -> dict[str, Any]:
        f: dict[str, Any] = {}
        if query.language_filter:
            f["language"] = query.language_filter
        if query.file_filter:
            f["file_path"] = query.file_filter
        if query.chunk_type_filter:
            f["chunk_type"] = query.chunk_type_filter
        return f

    def _vector_to_context(
        self, r: SearchResult, score: float, source: str
    ) -> RetrievedContext:
        return RetrievedContext(
            chunk_id=r.chunk_id,
            content=r.content,
            file_path=r.file_path,
            start_line=r.start_line,
            end_line=r.end_line,
            language=r.language,
            chunk_type=r.chunk_type,
            name=r.name,
            score=score,
            source=source,
            metadata=r.metadata,
        )

    def _symbol_to_context(self, m: SymbolMatch, score: float | None = None) -> RetrievedContext:
        return RetrievedContext(
            chunk_id=m.chunk_id,
            content=m.content,
            file_path=m.file_path,
            start_line=m.start_line,
            end_line=m.end_line,
            language=m.language,
            chunk_type=m.chunk_type,
            name=m.name,
            score=score if score is not None else m.score * self.symbol_weight,
            source="symbol",
        )