"""
High-level semantic search orchestration for Kodiak V3.

``SemanticSearch`` is the public repository-search interface used by agents.
It coordinates the V3 RAG components without reimplementing their work:

* ``RepositoryIndexer`` produces structural repository indexes.
* ``Chunker`` produces semantic chunks from an existing index.
* ``EmbeddingService`` embeds query text when an embedding service is injected.
* ``Retriever`` performs vector, keyword, hybrid, and metadata-aware ranking.
* ``DependencyGraph`` provides dependency-aware expansion and related code.

This module does not build prompts, call LLMs, modify repositories, parse source
files directly, implement embedding math, or execute project code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import structlog

from kodiak.rag.chunking import ChunkSymbolType, Chunker
from kodiak.rag.dependency_graph import DependencyGraph
from kodiak.rag.embeddings import ChunkEmbedding, EmbeddingService
from kodiak.rag.repository_index import RepositoryIndex, RepositoryIndexer
from kodiak.rag.retriever import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)

logger = structlog.get_logger(__name__)


class SearchKind(str, Enum):
    """Search categories exposed by the agent-facing API."""

    CODE = "code"
    SYMBOL = "symbol"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    FILE = "file"
    DOCUMENTATION = "documentation"
    RELATED = "related"
    DEPENDENCY = "dependency"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SemanticSearchConfig:
    """Configuration for high-level search orchestration."""

    top_k: int = 10
    min_confidence: float = 0.0
    include_dependency_context: bool = True
    dependency_context_top_k: int = 5
    auto_embed_queries: bool = True
    documentation_fallback_to_code: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.dependency_context_top_k <= 0:
            raise ValueError("dependency_context_top_k must be greater than zero")


@dataclass(frozen=True)
class SemanticSearchQuery:
    """
    Public query object accepted by :meth:`SemanticSearch.search`.

    ``query_embedding`` may be supplied by callers that already embedded the
    query. When omitted and ``EmbeddingService`` is injected, ``SemanticSearch``
    asks that service for a query embedding instead of implementing embedding
    logic locally.
    """

    text: str | None = None
    query_embedding: tuple[float, ...] | list[float] | None = None
    kind: SearchKind | str = SearchKind.CODE
    top_k: int | None = None
    min_confidence: float | None = None
    module: str | None = None
    modules: frozenset[str] = field(default_factory=frozenset)
    file_path: str | Path | None = None
    file_paths: frozenset[str] = field(default_factory=frozenset)
    symbol: str | None = None
    symbols: frozenset[str] = field(default_factory=frozenset)
    parent_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    use_embedding: bool | None = None
    include_dependency_context: bool | None = None

    def __post_init__(self) -> None:
        """Normalize enum-like fields for stable downstream behavior."""
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", SearchKind(self.kind))


@dataclass(frozen=True)
class SemanticSearchResult:
    """One ranked repository search result with confidence and location data."""

    chunk_id: str
    module_path: str
    file_path: Path
    symbol_name: str
    symbol_type: ChunkSymbolType
    source_code: str
    start_line: int
    end_line: int
    confidence: float
    rank: int
    search_kind: SearchKind
    retrieval_source: str
    docstring: str | None = None
    parent_class: str | None = None
    matched_terms: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def location(self) -> str:
        """Return a compact file and line location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this result."""
        return {
            "chunk_id": self.chunk_id,
            "module_path": self.module_path,
            "file_path": str(self.file_path),
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type.value,
            "source_code": self.source_code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": round(self.confidence, 6),
            "rank": self.rank,
            "search_kind": self.search_kind.value,
            "retrieval_source": self.retrieval_source,
            "docstring": self.docstring,
            "parent_class": self.parent_class,
            "matched_terms": list(self.matched_terms),
            "location": self.location,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SemanticSearchResponse:
    """Search response containing primary hits and optional graph context."""

    query: SemanticSearchQuery
    results: tuple[SemanticSearchResult, ...]
    related_results: tuple[SemanticSearchResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_results(self) -> tuple[SemanticSearchResult, ...]:
        """Return primary results followed by dependency-context results."""
        return (*self.results, *self.related_results)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this response."""
        return {
            "query": {
                "text": self.query.text,
                "kind": self.query.kind.value,
                "top_k": self.query.top_k,
                "min_confidence": self.query.min_confidence,
                "module": self.query.module,
                "modules": sorted(self.query.modules),
                "file_path": str(self.query.file_path) if self.query.file_path else None,
                "file_paths": sorted(self.query.file_paths),
                "symbol": self.query.symbol,
                "symbols": sorted(self.query.symbols),
                "parent_class": self.query.parent_class,
                "metadata": self.query.metadata,
                "use_embedding": self.query.use_embedding,
                "include_dependency_context": self.query.include_dependency_context,
            },
            "results": [result.to_dict() for result in self.results],
            "related_results": [result.to_dict() for result in self.related_results],
            "metadata": self.metadata,
        }


class SemanticSearch:
    """
    Agent-facing facade for repository semantic search.

    Args:
        repository_index: Existing structural index produced by
            :class:`RepositoryIndexer`.
        retriever: Retriever over precomputed chunk embeddings.
        dependency_graph: Optional graph for dependency-aware expansion.
        embedding_service: Optional service used only to embed query text.
        config: High-level search configuration.
    """

    def __init__(
        self,
        *,
        repository_index: RepositoryIndex,
        retriever: Retriever,
        dependency_graph: DependencyGraph | None = None,
        embedding_service: EmbeddingService | None = None,
        config: SemanticSearchConfig | None = None,
    ) -> None:
        """Initialize the semantic search facade from existing components."""
        self.repository_index = repository_index
        self.retriever = retriever
        self.dependency_graph = dependency_graph
        self.embedding_service = embedding_service
        self.config = config or SemanticSearchConfig()

    @classmethod
    def from_embeddings(
        cls,
        *,
        repository_index: RepositoryIndex,
        chunk_embeddings: Iterable[ChunkEmbedding],
        dependency_graph: DependencyGraph | None = None,
        embedding_service: EmbeddingService | None = None,
        search_config: SemanticSearchConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> SemanticSearch:
        """
        Wire ``SemanticSearch`` from an existing index and chunk embeddings.

        This constructor intentionally accepts already-computed embeddings. It
        builds the dependency graph and retriever through their public APIs and
        does not parse files or embed repository chunks itself.
        """
        graph = dependency_graph or DependencyGraph.from_index(repository_index)
        retriever = Retriever(
            chunk_embeddings,
            dependency_graph=graph,
            config=retrieval_config,
        )
        return cls(
            repository_index=repository_index,
            retriever=retriever,
            dependency_graph=graph,
            embedding_service=embedding_service,
            config=search_config,
        )

    @classmethod
    async def prepare_repository(
        cls,
        root_path: str | Path,
        *,
        embedding_service: EmbeddingService,
        repository_indexer: RepositoryIndexer | None = None,
        chunker: Chunker | None = None,
        dependency_graph: DependencyGraph | None = None,
        search_config: SemanticSearchConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> SemanticSearch:
        """
        Build a ready-to-use search service for a repository.

        This setup helper delegates indexing, chunking, graph construction, and
        embedding to the existing V3 services. It is provided for composition
        layers that want one orchestration call; production callers with stored
        indexes and embeddings should prefer :meth:`from_embeddings`.
        """
        indexer = repository_indexer or RepositoryIndexer()
        repository_index = await asyncio.to_thread(indexer.index_repository, root_path)
        chunks = await asyncio.to_thread((chunker or Chunker()).chunk_repository, repository_index)
        chunk_embeddings = await embedding_service.embed_chunks(chunks)
        graph = dependency_graph or DependencyGraph.from_index(repository_index)
        return cls.from_embeddings(
            repository_index=repository_index,
            chunk_embeddings=chunk_embeddings,
            dependency_graph=graph,
            embedding_service=embedding_service,
            search_config=search_config,
            retrieval_config=retrieval_config,
        )

    async def search(self, query: SemanticSearchQuery | str) -> SemanticSearchResponse:
        """
        Run a repository search using keyword, vector, hybrid, and metadata signals.

        Passing a string performs a code search. Passing ``SemanticSearchQuery``
        allows callers to constrain by symbol, module, file, parent class, and
        arbitrary chunk metadata.
        """
        search_query = self._normalize_query(query)
        embedded_query = await self._with_query_embedding(search_query)
        retrieval_query = self._to_retrieval_query(embedded_query)
        context = await self._retrieve_context(embedded_query, retrieval_query)
        response = self._response_from_context(embedded_query, context)

        logger.debug(
            "semantic_search_complete",
            kind=embedded_query.kind.value,
            results=len(response.results),
            related=len(response.related_results),
            used_embedding=embedded_query.query_embedding is not None,
        )
        return response

    async def search_code(
        self,
        query: str,
        *,
        query_embedding: tuple[float, ...] | list[float] | None = None,
        top_k: int | None = None,
        min_confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search repository code by natural language, vector, or both."""
        return await self.search(
            SemanticSearchQuery(
                text=query,
                query_embedding=query_embedding,
                kind=SearchKind.HYBRID if query_embedding is not None else SearchKind.CODE,
                top_k=top_k,
                min_confidence=min_confidence,
                metadata=metadata or {},
            )
        )

    async def search_symbol(
        self,
        symbol: str,
        *,
        query: str | None = None,
        query_embedding: tuple[float, ...] | list[float] | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search for chunks matching a symbol name or qualified symbol suffix."""
        return await self.search(
            SemanticSearchQuery(
                text=query or symbol,
                query_embedding=query_embedding,
                kind=SearchKind.SYMBOL,
                top_k=top_k,
                symbol=symbol,
                metadata=metadata or {},
            )
        )

    async def search_function(
        self,
        function: str,
        *,
        query: str | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search for synchronous or asynchronous function chunks."""
        return await self.search(
            SemanticSearchQuery(
                text=query or function,
                kind=SearchKind.FUNCTION,
                top_k=top_k,
                symbol=function,
                metadata=metadata or {},
            )
        )

    async def search_class(
        self,
        class_name: str,
        *,
        query: str | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search for class chunks by name or qualified suffix."""
        return await self.search(
            SemanticSearchQuery(
                text=query or class_name,
                kind=SearchKind.CLASS,
                top_k=top_k,
                symbol=class_name,
                metadata=metadata or {},
            )
        )

    async def search_module(
        self,
        module: str,
        *,
        query: str | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search the module-level chunk for a module."""
        return await self.search(
            SemanticSearchQuery(
                text=query or module,
                kind=SearchKind.MODULE,
                top_k=top_k,
                module=module,
                metadata=metadata or {},
            )
        )

    async def search_file(
        self,
        file_path: str | Path,
        *,
        query: str | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search chunks from one file path or path suffix."""
        return await self.search(
            SemanticSearchQuery(
                text=query or Path(file_path).name,
                kind=SearchKind.FILE,
                top_k=top_k,
                file_path=file_path,
                metadata=metadata or {},
            )
        )

    async def search_related(
        self,
        module: str,
        *,
        top_k: int | None = None,
        include_self: bool = False,
    ) -> SemanticSearchResponse:
        """Return chunks from modules directly related by dependency graph edges."""
        results = await self.retriever.retrieve_related(
            module,
            top_k=top_k or self.config.top_k,
            include_self=include_self,
        )
        query = SemanticSearchQuery(
            text=module,
            kind=SearchKind.RELATED,
            top_k=top_k,
            module=module,
            use_embedding=False,
            include_dependency_context=False,
        )
        return SemanticSearchResponse(
            query=query,
            results=tuple(
                self._to_search_result(result, SearchKind.RELATED)
                for result in self._rerank(results)
            ),
            metadata={"pipeline": "dependency_graph.related_modules"},
        )

    async def search_dependencies(
        self,
        module: str,
        *,
        include_dependents: bool = True,
        include_transitive: bool = False,
        top_k: int | None = None,
    ) -> SemanticSearchResponse:
        """Search chunks in modules that depend on, or are depended on by, ``module``."""
        modules = self._dependency_modules(
            module,
            include_dependents=include_dependents,
            include_transitive=include_transitive,
        )
        query = SemanticSearchQuery(
            text=module,
            kind=SearchKind.DEPENDENCY,
            top_k=top_k,
            module=module,
            use_embedding=False,
            include_dependency_context=False,
        )
        if not modules:
            return SemanticSearchResponse(
                query=query,
                results=(),
                metadata={"pipeline": "dependency_graph", "dependency_modules": []},
            )

        retrieval_query = RetrievalQuery(
            text=module,
            top_k=top_k or self.config.top_k,
            filters=RetrievalFilters(modules=frozenset(modules)),
        )
        results = await self.retriever.retrieve(retrieval_query)
        return SemanticSearchResponse(
            query=query,
            results=tuple(
                self._to_search_result(result, SearchKind.DEPENDENCY)
                for result in self._rerank(results)
            ),
            metadata={
                "pipeline": "dependency_graph.retriever",
                "dependency_modules": sorted(modules),
            },
        )

    async def search_documentation(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Search docstrings and documentation-bearing code chunks."""
        response = await self.search(
            SemanticSearchQuery(
                text=query,
                kind=SearchKind.DOCUMENTATION,
                top_k=top_k,
                metadata=metadata or {},
            )
        )
        documented = tuple(result for result in response.results if result.docstring)
        if documented or not self.config.documentation_fallback_to_code:
            return replace(
                response,
                results=self._rerank_search_results(documented),
                metadata={
                    **response.metadata,
                    "documentation_filter": "docstring_present",
                },
            )
        return replace(
            response,
            metadata={
                **response.metadata,
                "documentation_filter": "fallback_to_code_results",
            },
        )

    async def hybrid_search(
        self,
        query: str,
        *,
        query_embedding: tuple[float, ...] | list[float] | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticSearchResponse:
        """Explicit hybrid text and vector search convenience method."""
        return await self.search(
            SemanticSearchQuery(
                text=query,
                query_embedding=query_embedding,
                kind=SearchKind.HYBRID,
                top_k=top_k,
                metadata=metadata or {},
            )
        )

    def indexed_modules(self) -> tuple[str, ...]:
        """Return module names known to the repository index."""
        return tuple(module.module_name for module in self.repository_index.modules)

    def indexed_files(self) -> tuple[Path, ...]:
        """Return indexed file paths known to the repository index."""
        return tuple(module.path for module in self.repository_index.modules)

    def find_indexed_symbols(self, name: str) -> dict[str, tuple[str, ...]]:
        """
        Return structural symbol matches from ``RepositoryIndex``.

        This method delegates to the index models and performs no parsing or
        retrieval. It is useful when agents need exact symbol inventory before
        semantic retrieval.
        """
        classes = self.repository_index.find_classes(name)
        functions = self.repository_index.find_functions(name)
        return {
            "classes": tuple(class_info.qualname for class_info in classes),
            "functions": tuple(function.qualname for function in functions),
        }

    def _normalize_query(self, query: SemanticSearchQuery | str) -> SemanticSearchQuery:
        if isinstance(query, SemanticSearchQuery):
            return query
        return SemanticSearchQuery(text=query, kind=SearchKind.CODE)

    async def _with_query_embedding(self, query: SemanticSearchQuery) -> SemanticSearchQuery:
        should_embed = (
            query.use_embedding
            if query.use_embedding is not None
            else self.config.auto_embed_queries
        )
        if (
            not should_embed
            or query.query_embedding is not None
            or not query.text
            or self.embedding_service is None
        ):
            return query

        embedding = await self.embedding_service.embed_query(query.text)
        return replace(query, query_embedding=embedding.embedding)

    async def _retrieve_context(
        self,
        query: SemanticSearchQuery,
        retrieval_query: RetrievalQuery,
    ) -> RetrievalContext:
        include_context = (
            query.include_dependency_context
            if query.include_dependency_context is not None
            else self.config.include_dependency_context
        )
        if include_context:
            return await self.retriever.retrieve_context(
                retrieval_query,
                related_top_k=self.config.dependency_context_top_k,
            )

        primary = tuple(await self.retriever.retrieve(retrieval_query))
        return RetrievalContext(primary=primary, related=())

    def _to_retrieval_query(self, query: SemanticSearchQuery) -> RetrievalQuery:
        return RetrievalQuery(
            text=query.text,
            query_embedding=query.query_embedding,
            top_k=query.top_k or self.config.top_k,
            min_score=self._confidence_threshold(query),
            filters=self._filters_for_query(query),
        )

    def _filters_for_query(self, query: SemanticSearchQuery) -> RetrievalFilters:
        symbol_type: ChunkSymbolType | None = None
        symbol_types: frozenset[ChunkSymbolType] = frozenset()
        if query.kind == SearchKind.CLASS:
            symbol_type = ChunkSymbolType.CLASS
        elif query.kind == SearchKind.FUNCTION:
            symbol_types = frozenset(
                {ChunkSymbolType.FUNCTION, ChunkSymbolType.ASYNC_FUNCTION}
            )
        elif query.kind == SearchKind.MODULE:
            symbol_type = ChunkSymbolType.MODULE

        return RetrievalFilters(
            module=query.module,
            modules=query.modules,
            file_path=query.file_path,
            file_paths=query.file_paths,
            symbol=query.symbol,
            symbols=query.symbols,
            symbol_type=symbol_type,
            symbol_types=symbol_types,
            parent_class=query.parent_class,
            metadata=query.metadata,
        )

    def _response_from_context(
        self,
        query: SemanticSearchQuery,
        context: RetrievalContext,
    ) -> SemanticSearchResponse:
        primary = tuple(
            self._to_search_result(result, query.kind)
            for result in self._rerank(context.primary)
            if self._confidence(result) >= self._confidence_threshold(query)
        )
        related = tuple(
            self._to_search_result(result, SearchKind.RELATED)
            for result in self._rerank(context.related)
        )
        return SemanticSearchResponse(
            query=query,
            results=primary,
            related_results=related,
            metadata={
                "pipeline": self._pipeline_name(query),
                "ranking": "retriever_score_plus_evidence_confidence",
                "used_query_embedding": query.query_embedding is not None,
                "dependency_context": bool(related),
            },
        )

    def _dependency_modules(
        self,
        module: str,
        *,
        include_dependents: bool,
        include_transitive: bool,
    ) -> set[str]:
        if self.dependency_graph is None:
            return set()

        if include_transitive:
            modules = set(self.dependency_graph.get_all_dependencies(module))
            if include_dependents:
                modules.update(self.dependency_graph.get_all_dependents(module))
            return modules

        modules = set(self.dependency_graph.get_dependencies(module))
        if include_dependents:
            modules.update(self.dependency_graph.get_dependents(module))
        return modules

    def _to_search_result(
        self,
        result: RetrievalResult,
        kind: SearchKind,
    ) -> SemanticSearchResult:
        chunk = result.chunk
        return SemanticSearchResult(
            chunk_id=chunk.id,
            module_path=chunk.module_path,
            file_path=chunk.file_path,
            symbol_name=chunk.symbol_name,
            symbol_type=chunk.symbol_type,
            source_code=chunk.source_code,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            confidence=self._confidence(result),
            rank=result.rank,
            search_kind=kind,
            retrieval_source=result.source,
            docstring=chunk.docstring,
            parent_class=chunk.parent_class,
            matched_terms=result.matched_terms,
            metadata={
                **chunk.metadata,
                **result.metadata,
                "retrieval_score": round(result.score, 6),
                "vector_score": (
                    None if result.vector_score is None else round(result.vector_score, 6)
                ),
                "keyword_score": (
                    None if result.keyword_score is None else round(result.keyword_score, 6)
                ),
                "metadata_score": round(result.metadata_score, 6),
            },
        )

    def _pipeline_name(self, query: SemanticSearchQuery) -> str:
        if query.kind == SearchKind.RELATED:
            return "dependency_graph.related_modules"
        if query.kind == SearchKind.DEPENDENCY:
            return "dependency_graph.retriever"
        if query.query_embedding is not None and query.text:
            return "embedding_service.query_embedding.retriever.hybrid"
        if query.query_embedding is not None:
            return "retriever.vector"
        if query.metadata:
            return "retriever.keyword_metadata"
        return "retriever.keyword"

    def _confidence_threshold(self, query: SemanticSearchQuery) -> float:
        return (
            query.min_confidence
            if query.min_confidence is not None
            else self.config.min_confidence
        )

    @staticmethod
    def _rerank(results: Iterable[RetrievalResult]) -> tuple[RetrievalResult, ...]:
        ordered = sorted(
            results,
            key=lambda result: (
                SemanticSearch._confidence(result),
                result.score,
                -result.rank,
            ),
            reverse=True,
        )
        return tuple(
            replace(result, rank=index)
            for index, result in enumerate(ordered, start=1)
        )

    @staticmethod
    def _rerank_search_results(
        results: Iterable[SemanticSearchResult],
    ) -> tuple[SemanticSearchResult, ...]:
        ordered = sorted(
            results,
            key=lambda result: (result.confidence, -result.rank),
            reverse=True,
        )
        return tuple(
            replace(result, rank=index)
            for index, result in enumerate(ordered, start=1)
        )

    @staticmethod
    def _confidence(result: RetrievalResult) -> float:
        """
        Convert retriever score and evidence into a bounded confidence score.

        The retriever owns raw ranking. This method adds small transparent
        bonuses for independent evidence signals so callers can distinguish
        exact metadata hits and hybrid matches from weak keyword-only matches.
        """
        evidence_bonus = 0.0
        if result.vector_score is not None:
            evidence_bonus += 0.05
        if result.keyword_score is not None and result.matched_terms:
            evidence_bonus += 0.05
        if result.metadata_score > 0.0:
            evidence_bonus += 0.03
        return max(0.0, min(1.0, result.score + evidence_bonus))
