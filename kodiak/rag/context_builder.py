"""
Final context assembly stage for Kodiak V3 repository intelligence.

``ContextBuilder`` consumes results from ``SemanticSearch`` and prepares
structured, prompt-ready context for agents. It does not call LLMs, execute
repository code, compute embeddings, parse repositories, rescan files, modify
files, or duplicate retrieval logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

from kodiak.rag.chunking import ChunkSymbolType
from kodiak.rag.dependency_graph import DependencyGraph
from kodiak.rag.semantic_search import (
    SemanticSearch,
    SemanticSearchQuery,
    SemanticSearchResponse,
    SemanticSearchResult,
)

logger = structlog.get_logger(__name__)


class ContextOrdering(StrEnum):
    """Supported ordering strategies for context blocks."""

    IMPORTANCE = "importance"
    FILE_HIERARCHY = "file_hierarchy"
    MODULE_HIERARCHY = "module_hierarchy"
    SYMBOL_HIERARCHY = "symbol_hierarchy"
    SOURCE_ORDER = "source_order"


class ContextPurpose(StrEnum):
    """High-level context use cases supported by the builder."""

    GENERAL = "general"
    REPOSITORY = "repository"
    SYMBOL = "symbol"
    ISSUE = "issue"
    REVIEW = "review"
    TASK = "task"


@dataclass(frozen=True)
class ContextBuilderConfig:
    """Configuration for context assembly and token budgeting.

    Attributes:
        max_tokens: Maximum estimated tokens in the final context.
        reserved_tokens: Tokens reserved for caller instructions or agent output.
        chars_per_token: Approximate character-to-token conversion ratio.
        include_related: Whether dependency-related results should be included.
        include_metadata: Whether prompt blocks include metadata headers.
        include_docstrings: Whether prompt blocks include docstrings when present.
        ordering: Default block ordering strategy.
        truncate_blocks: Whether individual blocks may be truncated to fit budget.
    """

    max_tokens: int = 8000
    reserved_tokens: int = 1000
    chars_per_token: float = 4.0
    include_related: bool = True
    include_metadata: bool = True
    include_docstrings: bool = True
    ordering: ContextOrdering = ContextOrdering.IMPORTANCE
    truncate_blocks: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.reserved_tokens < 0:
            raise ValueError("reserved_tokens must be zero or greater")
        if self.reserved_tokens >= self.max_tokens:
            raise ValueError("reserved_tokens must be less than max_tokens")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than zero")

    @property
    def available_tokens(self) -> int:
        """Return the token budget available for repository context."""
        return self.max_tokens - self.reserved_tokens


@dataclass(frozen=True)
class ContextRequest:
    """Input request for building repository context.

    Attributes:
        task: Natural-language task or search query.
        query_embedding: Optional precomputed query embedding for hybrid search.
        top_k: Optional search result limit.
        max_tokens: Optional context token budget override.
        purpose: Context use case.
        ordering: Optional ordering override.
        module: Optional module filter.
        file_path: Optional file filter.
        symbol: Optional symbol filter.
        parent_class: Optional class filter for methods.
        metadata: Optional metadata filters.
    """

    task: str
    query_embedding: tuple[float, ...] | list[float] | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    purpose: ContextPurpose = ContextPurpose.GENERAL
    ordering: ContextOrdering | None = None
    module: str | None = None
    file_path: str | Path | None = None
    symbol: str | None = None
    parent_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBlock:
    """A structured context block derived from one semantic search result.

    Attributes:
        id: Stable chunk identifier.
        module_path: Dotted module path.
        file_path: Source file path.
        symbol_name: Symbol represented by the block.
        symbol_type: Symbol type represented by the block.
        source_code: Source code included in context.
        start_line: Start line in the source file.
        end_line: End line in the source file.
        importance: Ranking score used during context assembly.
        token_count: Estimated token count for the rendered block.
        dependency_role: Whether the block is primary or dependency-related.
        parent_class: Parent class for methods or nested classes.
        docstring: Optional indexed docstring.
        metadata: Additional context metadata.
    """

    id: str
    module_path: str
    file_path: Path
    symbol_name: str
    symbol_type: ChunkSymbolType
    source_code: str
    start_line: int
    end_line: int
    importance: float
    token_count: int
    dependency_role: str
    parent_class: str | None = None
    docstring: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def location(self) -> str:
        """Return a compact file and line location."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_prompt_block(
        self,
        *,
        include_metadata: bool = True,
        include_docstring: bool = True,
    ) -> str:
        """Render this block as prompt-ready plain text.

        Args:
            include_metadata: Include context headers and metadata.
            include_docstring: Include docstring text when present.

        Returns:
            A prompt-ready context block.
        """
        header = [
            "---",
            f"File: {self.file_path}",
            f"Module: {self.module_path}",
            f"Symbol: {self.symbol_name} ({self.symbol_type.value})",
            f"Lines: {self.start_line}-{self.end_line}",
            f"Role: {self.dependency_role}",
            f"Importance: {self.importance:.4f}",
        ]
        if self.parent_class:
            header.append(f"Parent class: {self.parent_class}")
        if include_docstring and self.docstring:
            header.extend(["Docstring:", self.docstring])
        if include_metadata and self.metadata:
            header.append(f"Metadata: {self._metadata_summary()}")

        return "\n".join([*header, "```python", self.source_code, "```"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this block."""
        return {
            "id": self.id,
            "module_path": self.module_path,
            "file_path": str(self.file_path),
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type.value,
            "source_code": self.source_code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "importance": round(self.importance, 6),
            "token_count": self.token_count,
            "dependency_role": self.dependency_role,
            "parent_class": self.parent_class,
            "docstring": self.docstring,
            "metadata": self.metadata,
        }

    def _metadata_summary(self) -> str:
        scalar_items = {
            key: value
            for key, value in self.metadata.items()
            if isinstance(value, str | int | float | bool) or value is None
        }
        return ", ".join(f"{key}={value}" for key, value in sorted(scalar_items.items()))


@dataclass(frozen=True)
class BuiltContext:
    """Structured and prompt-ready context assembled for an agent.

    Attributes:
        task: Natural-language task used to build context.
        purpose: Context use case.
        blocks: Context blocks included within the token budget.
        prompt_context: Prompt-ready text assembled from blocks.
        token_count: Estimated token count of ``prompt_context``.
        discarded_blocks: Number of candidate blocks omitted by budgeting.
        truncated_blocks: Number of blocks shortened to fit budget.
        file_hierarchy: Included blocks grouped by file path.
        symbol_hierarchy: Included symbols grouped by module.
        dependency_map: Dependency relationships among included modules.
        metadata: Additional assembly metadata.
    """

    task: str
    purpose: ContextPurpose
    blocks: tuple[ContextBlock, ...]
    prompt_context: str
    token_count: int
    discarded_blocks: int
    truncated_blocks: int
    file_hierarchy: dict[str, tuple[str, ...]]
    symbol_hierarchy: dict[str, tuple[str, ...]]
    dependency_map: dict[str, tuple[str, ...]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this context."""
        return {
            "task": self.task,
            "purpose": self.purpose.value,
            "blocks": [block.to_dict() for block in self.blocks],
            "prompt_context": self.prompt_context,
            "token_count": self.token_count,
            "discarded_blocks": self.discarded_blocks,
            "truncated_blocks": self.truncated_blocks,
            "file_hierarchy": {
                path: list(symbols) for path, symbols in self.file_hierarchy.items()
            },
            "symbol_hierarchy": {
                module: list(symbols) for module, symbols in self.symbol_hierarchy.items()
            },
            "dependency_map": {
                module: list(dependencies) for module, dependencies in self.dependency_map.items()
            },
            "metadata": self.metadata,
        }


class ContextBuilder:
    """Build structured, budget-aware repository context for Kodiak agents."""

    def __init__(
        self,
        *,
        semantic_search: SemanticSearch,
        dependency_graph: DependencyGraph | None = None,
        config: ContextBuilderConfig | None = None,
    ) -> None:
        """Initialize the context builder.

        Args:
            semantic_search: Public search service used to retrieve candidates.
            dependency_graph: Optional graph for dependency metadata.
            config: Context assembly configuration.
        """
        self.semantic_search = semantic_search
        self.dependency_graph = dependency_graph or semantic_search.dependency_graph
        self.config = config or ContextBuilderConfig()

    async def build_context(self, request: ContextRequest | str) -> BuiltContext:
        """Build context for a natural-language task.

        Args:
            request: Context request or a plain task string.

        Returns:
            A structured context object with prompt-ready text.
        """
        context_request = (
            request if isinstance(request, ContextRequest) else ContextRequest(task=request)
        )
        response = await self.semantic_search.search(self._search_query(context_request))
        return self._build_from_response(context_request, response)

    async def build_repository_context(
        self,
        task: str,
        *,
        top_k: int | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BuiltContext:
        """Build broad repository context for a natural-language task."""
        return await self.build_context(
            ContextRequest(
                task=task,
                top_k=top_k,
                max_tokens=max_tokens,
                purpose=ContextPurpose.REPOSITORY,
                metadata=metadata or {},
            )
        )

    async def build_symbol_context(
        self,
        symbol: str,
        *,
        task: str | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Build context focused on a symbol."""
        response = await self.semantic_search.search_symbol(symbol, top_k=top_k)
        request = ContextRequest(
            task=task or symbol,
            symbol=symbol,
            top_k=top_k,
            max_tokens=max_tokens,
            purpose=ContextPurpose.SYMBOL,
        )
        return self._build_from_response(request, response)

    async def build_issue_context(
        self,
        issue: str,
        *,
        top_k: int | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BuiltContext:
        """Build context for investigating an issue or bug report."""
        return await self.build_context(
            ContextRequest(
                task=issue,
                top_k=top_k,
                max_tokens=max_tokens,
                purpose=ContextPurpose.ISSUE,
                metadata=metadata or {},
            )
        )

    async def build_review_context(
        self,
        review_task: str,
        *,
        file_path: str | Path | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Build context for code review tasks."""
        return await self.build_context(
            ContextRequest(
                task=review_task,
                file_path=file_path,
                top_k=top_k,
                max_tokens=max_tokens,
                purpose=ContextPurpose.REVIEW,
            )
        )

    async def build_task_context(
        self,
        task: str,
        *,
        module: str | None = None,
        file_path: str | Path | None = None,
        symbol: str | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Build context for an implementation task."""
        return await self.build_context(
            ContextRequest(
                task=task,
                module=module,
                file_path=file_path,
                symbol=symbol,
                top_k=top_k,
                max_tokens=max_tokens,
                purpose=ContextPurpose.TASK,
            )
        )

    def summarize_context(self, context: BuiltContext) -> dict[str, Any]:
        """Summarize assembled context without calling an LLM.

        Args:
            context: Built context to summarize.

        Returns:
            Deterministic summary statistics and hierarchy information.
        """
        return {
            "task": context.task,
            "purpose": context.purpose.value,
            "blocks": len(context.blocks),
            "token_count": context.token_count,
            "discarded_blocks": context.discarded_blocks,
            "truncated_blocks": context.truncated_blocks,
            "files": len(context.file_hierarchy),
            "modules": len(context.symbol_hierarchy),
            "symbols": sum(len(symbols) for symbols in context.symbol_hierarchy.values()),
        }

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count using the configured character ratio.

        Args:
            text: Text to estimate.

        Returns:
            Approximate token count.
        """
        return max(1, int(len(text) / self.config.chars_per_token)) if text else 0

    def truncate_context(self, context: BuiltContext, max_tokens: int) -> BuiltContext:
        """Return a copy of context truncated to ``max_tokens``.

        Args:
            context: Context to truncate.
            max_tokens: Maximum estimated tokens.

        Returns:
            A budget-constrained context.
        """
        blocks, discarded, truncated = self._fit_blocks(context.blocks, max_tokens)
        return self._assemble_context(
            task=context.task,
            purpose=context.purpose,
            blocks=blocks,
            discarded_blocks=context.discarded_blocks + discarded,
            truncated_blocks=context.truncated_blocks + truncated,
            max_tokens=max_tokens,
            candidate_count=len(context.blocks),
        )

    def _build_from_response(
        self,
        request: ContextRequest,
        response: SemanticSearchResponse,
    ) -> BuiltContext:
        candidates = self._deduplicate_results(response.all_results)
        blocks = tuple(self._result_to_block(result, response) for result in candidates)
        ordered = self._order_blocks(blocks, request.ordering or self.config.ordering)
        max_tokens = request.max_tokens or self.config.available_tokens
        fitted, discarded, truncated = self._fit_blocks(ordered, max_tokens)
        return self._assemble_context(
            task=request.task,
            purpose=request.purpose,
            blocks=fitted,
            discarded_blocks=discarded,
            truncated_blocks=truncated,
            max_tokens=max_tokens,
            candidate_count=len(blocks),
        )

    def _search_query(self, request: ContextRequest) -> SemanticSearchQuery:
        return SemanticSearchQuery(
            text=request.task,
            query_embedding=request.query_embedding,
            top_k=request.top_k,
            module=request.module,
            file_path=request.file_path,
            symbol=request.symbol,
            parent_class=request.parent_class,
            metadata=request.metadata,
        )

    def _deduplicate_results(
        self,
        results: Iterable[SemanticSearchResult],
    ) -> tuple[SemanticSearchResult, ...]:
        best_by_id: dict[str, SemanticSearchResult] = {}
        for result in results:
            current = best_by_id.get(result.chunk_id)
            if current is None or result.confidence > current.confidence:
                best_by_id[result.chunk_id] = result
        return tuple(best_by_id.values())

    def _result_to_block(
        self,
        result: SemanticSearchResult,
        response: SemanticSearchResponse,
    ) -> ContextBlock:
        role = "primary" if result in response.results else "related"
        rendered = self._render_preview(result, role)
        return ContextBlock(
            id=result.chunk_id,
            module_path=result.module_path,
            file_path=result.file_path,
            symbol_name=result.symbol_name,
            symbol_type=result.symbol_type,
            source_code=result.source_code,
            start_line=result.start_line,
            end_line=result.end_line,
            importance=self._importance(result, role),
            token_count=self.estimate_token_count(rendered),
            dependency_role=role,
            parent_class=result.parent_class,
            docstring=result.docstring,
            metadata=result.metadata,
        )

    def _order_blocks(
        self,
        blocks: tuple[ContextBlock, ...],
        ordering: ContextOrdering,
    ) -> tuple[ContextBlock, ...]:
        if ordering == ContextOrdering.FILE_HIERARCHY:

            def key(block):
                return (block.file_path.as_posix(), block.start_line)
        elif ordering == ContextOrdering.MODULE_HIERARCHY:

            def key(block):
                return (block.module_path, block.file_path.as_posix(), block.start_line)
        elif ordering == ContextOrdering.SYMBOL_HIERARCHY:

            def key(block):
                return (block.module_path, block.parent_class or "", block.symbol_name)
        elif ordering == ContextOrdering.SOURCE_ORDER:

            def key(block):
                return (block.file_path.as_posix(), block.start_line, block.end_line)
        else:

            def key(block):
                return (-block.importance, block.file_path.as_posix(), block.start_line)

        return tuple(sorted(blocks, key=key))

    def _fit_blocks(
        self,
        blocks: Iterable[ContextBlock],
        max_tokens: int,
    ) -> tuple[tuple[ContextBlock, ...], int, int]:
        included: list[ContextBlock] = []
        used_tokens = 0
        discarded = 0
        truncated = 0

        for block in blocks:
            rendered_tokens = self.estimate_token_count(
                block.to_prompt_block(
                    include_metadata=self.config.include_metadata,
                    include_docstring=self.config.include_docstrings,
                )
            )
            if used_tokens + rendered_tokens <= max_tokens:
                included.append(self._replace_token_count(block, rendered_tokens))
                used_tokens += rendered_tokens
                continue

            remaining = max_tokens - used_tokens
            if self.config.truncate_blocks and remaining > 32:
                shortened = self._truncate_block(block, remaining)
                included.append(shortened)
                used_tokens += shortened.token_count
                truncated += 1
            else:
                discarded += 1

        return tuple(included), discarded, truncated

    def _assemble_context(
        self,
        *,
        task: str,
        purpose: ContextPurpose,
        blocks: tuple[ContextBlock, ...],
        discarded_blocks: int,
        truncated_blocks: int,
        max_tokens: int,
        candidate_count: int,
    ) -> BuiltContext:
        prompt_context = self._render_prompt_context(task, purpose, blocks)
        token_count = self.estimate_token_count(prompt_context)
        return BuiltContext(
            task=task,
            purpose=purpose,
            blocks=blocks,
            prompt_context=prompt_context,
            token_count=token_count,
            discarded_blocks=discarded_blocks,
            truncated_blocks=truncated_blocks,
            file_hierarchy=self._file_hierarchy(blocks),
            symbol_hierarchy=self._symbol_hierarchy(blocks),
            dependency_map=self._dependency_map(blocks),
            metadata={
                "max_tokens": max_tokens,
                "candidate_blocks": candidate_count,
                "ordering": self.config.ordering.value,
            },
        )

    def _render_prompt_context(
        self,
        task: str,
        purpose: ContextPurpose,
        blocks: tuple[ContextBlock, ...],
    ) -> str:
        header = [
            "# Repository Context",
            f"Task: {task}",
            f"Purpose: {purpose.value}",
            f"Blocks: {len(blocks)}",
            "",
        ]
        rendered_blocks = [
            block.to_prompt_block(
                include_metadata=self.config.include_metadata,
                include_docstring=self.config.include_docstrings,
            )
            for block in blocks
        ]
        return "\n".join([*header, *rendered_blocks]).strip()

    def _render_preview(self, result: SemanticSearchResult, role: str) -> str:
        block = ContextBlock(
            id=result.chunk_id,
            module_path=result.module_path,
            file_path=result.file_path,
            symbol_name=result.symbol_name,
            symbol_type=result.symbol_type,
            source_code=result.source_code,
            start_line=result.start_line,
            end_line=result.end_line,
            importance=result.confidence,
            token_count=0,
            dependency_role=role,
            parent_class=result.parent_class,
            docstring=result.docstring,
            metadata=result.metadata,
        )
        return block.to_prompt_block(
            include_metadata=self.config.include_metadata,
            include_docstring=self.config.include_docstrings,
        )

    def _truncate_block(self, block: ContextBlock, max_tokens: int) -> ContextBlock:
        max_chars = max(0, int(max_tokens * self.config.chars_per_token))
        rendered_without_code = block.to_prompt_block(
            include_metadata=self.config.include_metadata,
            include_docstring=self.config.include_docstrings,
        ).replace(block.source_code, "")
        available_chars = max_chars - len(rendered_without_code) - len("\n# ... truncated")
        if available_chars <= 0:
            return self._replace_source(block, "", max_tokens)
        source = block.source_code[:available_chars].rstrip()
        return self._replace_source(block, f"{source}\n# ... truncated", max_tokens)

    def _file_hierarchy(self, blocks: tuple[ContextBlock, ...]) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for block in blocks:
            grouped.setdefault(block.file_path.as_posix(), []).append(block.symbol_name)
        return {path: tuple(symbols) for path, symbols in sorted(grouped.items())}

    def _symbol_hierarchy(self, blocks: tuple[ContextBlock, ...]) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for block in blocks:
            grouped.setdefault(block.module_path, []).append(block.symbol_name)
        return {module: tuple(symbols) for module, symbols in sorted(grouped.items())}

    def _dependency_map(self, blocks: tuple[ContextBlock, ...]) -> dict[str, tuple[str, ...]]:
        if self.dependency_graph is None:
            return {}
        included_modules = {block.module_path for block in blocks}
        dependency_map: dict[str, tuple[str, ...]] = {}
        for module in sorted(included_modules):
            dependencies = self.dependency_graph.get_dependencies(module)
            visible = tuple(
                sorted(dependency for dependency in dependencies if dependency in included_modules)
            )
            dependency_map[module] = visible
        return dependency_map

    @staticmethod
    def _importance(result: SemanticSearchResult, role: str) -> float:
        role_multiplier = 1.0 if role == "primary" else 0.85
        return max(0.0, min(1.0, result.confidence * role_multiplier))

    @staticmethod
    def _replace_token_count(block: ContextBlock, token_count: int) -> ContextBlock:
        return ContextBlock(
            id=block.id,
            module_path=block.module_path,
            file_path=block.file_path,
            symbol_name=block.symbol_name,
            symbol_type=block.symbol_type,
            source_code=block.source_code,
            start_line=block.start_line,
            end_line=block.end_line,
            importance=block.importance,
            token_count=token_count,
            dependency_role=block.dependency_role,
            parent_class=block.parent_class,
            docstring=block.docstring,
            metadata=block.metadata,
        )

    def _replace_source(
        self,
        block: ContextBlock,
        source_code: str,
        token_count: int,
    ) -> ContextBlock:
        return ContextBlock(
            id=block.id,
            module_path=block.module_path,
            file_path=block.file_path,
            symbol_name=block.symbol_name,
            symbol_type=block.symbol_type,
            source_code=source_code,
            start_line=block.start_line,
            end_line=block.end_line,
            importance=block.importance,
            token_count=token_count,
            dependency_role=block.dependency_role,
            parent_class=block.parent_class,
            docstring=block.docstring,
            metadata={**block.metadata, "truncated": True},
        )
