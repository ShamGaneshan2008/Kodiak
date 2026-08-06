# kodiak/memory/service.py
"""Unified facade over Kodiak's Working, Short-Term, Episodic, Semantic, and Procedural Memory systems."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Final, Sequence

import structlog

from .consolidation import ConsolidationResult, MemoryConsolidator
from .context import MemoryContextBuilder
from .episodic import Episode, EpisodicMemory
from .errors import MemoryNotFoundError, MemoryServiceError
from .long_term import LongTermMemory
from .models import Memory, MemoryType, SearchResult
from .persistence import JSONFileMemoryPersistence
from .procedural import ProceduralMemory, Procedure, ProcedureStep
from .ranking import MemoryRanker
from .retrieval import MemoryRetriever
from .semantic import SemanticEntity, SemanticMemory
from .short_term import ShortTermMemory, ShortTermMemoryItem
from .working import WorkingMemory, WorkingMemoryItem

logger = structlog.get_logger(__name__)

__all__ = ["MemoryService"]

_DEFAULT_SEARCH_LIMIT: Final[int] = 10
_DEFAULT_LIST_LIMIT: Final[int] = 100


class MemoryService:
    """Master facade and Dependency Injection container for the Kodiak Memory System.

    Unifies Working Memory, Short-Term Memory, Long-Term Memory (Episodic,
    Semantic, Procedural), Retrieval, Ranking, Context Building, Persistence,
    and Consolidation into a clean, async-first API.
    """

    def __init__(
        self,
        working: WorkingMemory | None = None,
        short_term: ShortTermMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        procedural: ProceduralMemory | None = None,
        long_term: LongTermMemory | None = None,
        ranker: MemoryRanker | None = None,
        retriever: MemoryRetriever | None = None,
        context_builder: MemoryContextBuilder | None = None,
        consolidator: MemoryConsolidator | None = None,
        persistence_path: str | Path | None = None,
    ) -> None:
        """Initialize MemoryService.

        Args:
            working: WorkingMemory component instance.
            short_term: ShortTermMemory component instance.
            episodic: EpisodicMemory component instance.
            semantic: SemanticMemory component instance.
            procedural: ProceduralMemory component instance.
            long_term: LongTermMemory composite component.
            ranker: MemoryRanker scoring engine.
            retriever: MemoryRetriever concurrent retriever.
            context_builder: MemoryContextBuilder context formatting component.
            consolidator: MemoryConsolidator task consolidation worker.
            persistence_path: File path for JSON disk persistence driver.
        """
        self.working = working or WorkingMemory()
        self.short_term = short_term or ShortTermMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()

        self.long_term = long_term or LongTermMemory(
            episodic=self.episodic,
            semantic=self.semantic,
            procedural=self.procedural,
        )

        self.ranker = ranker or MemoryRanker()
        self.retriever = retriever or MemoryRetriever(
            working=self.working,
            short_term=self.short_term,
            long_term=self.long_term,
            ranker=self.ranker,
        )
        self.context_builder = context_builder or MemoryContextBuilder(
            retriever=self.retriever
        )
        self.consolidator = consolidator or MemoryConsolidator(
            working_memory=self.working,
            episodic_memory=self.episodic,
            semantic_memory=self.semantic,
            procedural_memory=self.procedural,
        )

        self.persistence = (
            JSONFileMemoryPersistence(persistence_path) if persistence_path else None
        )

    # Core CLI and Facade CRUD Operations

    async def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory record.

        Args:
            content: Main text content to store.
            memory_type: Type of memory store (EPISODIC, SEMANTIC, PROCEDURAL, WORKING, SHORT_TERM).
            tags: Filter or labeling tags.
            metadata: Additional metadata key-value dictionary.

        Returns:
            Normalized Memory instance.
        """
        tags = tags or []
        metadata = metadata or {}

        try:
            if memory_type is MemoryType.WORKING:
                task_id = self._coerce_uuid(metadata.get("task_id")) or uuid.uuid4()
                item = await self.working.create_working_memory(
                    task_id=task_id,
                    goal=content,
                    context=dict(metadata.get("context") or {}),
                )
                return self.working.to_memory(item)

            if memory_type is MemoryType.SHORT_TERM:
                session_id = str(metadata.get("session_id", "default"))
                role = str(metadata.get("role", "user"))
                item = await self.short_term.add_item(
                    session_id=session_id,
                    content=content,
                    role=role,
                    metadata=metadata,
                )
                return self.short_term.to_memory(item)

            return await self.long_term.add_memory(
                content=content,
                memory_type=memory_type,
                tags=tags,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("memory_add_failed", memory_type=str(memory_type))
            raise MemoryServiceError(f"Failed to add {memory_type} memory") from exc

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search memories across requested store(s).

        Args:
            query: Natural language query string.
            memory_type: Optional scope to specific MemoryType.
            limit: Maximum items to return.
            tags: Filter by tag labels.

        Returns:
            List of SearchResult items ranked by relevance.
        """
        if memory_type in (None, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL):
            return await self.long_term.search(
                query=query,
                memory_type=memory_type,
                limit=limit,
                tags=tags,
            )

        memory_types = [memory_type] if memory_type else None
        return await self.retriever.retrieve(
            query=query,
            memory_types=memory_types,
            tags=tags,
            limit=limit,
        )

    async def list(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = _DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories across stores.

        Args:
            memory_type: Optional scope to MemoryType.
            tags: Filter by tag labels.
            limit: Maximum items to return.
            offset: Offset index for pagination.

        Returns:
            List of Memory models.
        """
        memories: list[Memory] = []

        if memory_type in (None, MemoryType.WORKING):
            working_items = await self.working.list_working_memories(limit=limit)
            memories.extend(self.working.to_memory(w) for w in working_items)

        if memory_type in (None, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL):
            lt_memories = await self.long_term.list_memories(
                memory_type=memory_type,
                tags=tags,
                limit=limit,
                offset=offset,
            )
            memories.extend(lt_memories)

        wanted = set(tags or [])
        memories = [m for m in memories if not wanted or bool(wanted & set(m.tags))]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[offset : offset + limit]

    async def delete(
        self,
        memory_id: uuid.UUID | str,
        memory_type: MemoryType | None = None,
    ) -> bool:
        """Delete a single memory record by ID.

        Args:
            memory_id: UUID or string identifier of memory.
            memory_type: Optional memory type hint.

        Returns:
            True if memory was located and deleted.

        Raises:
            MemoryNotFoundError: If memory record could not be located.
        """
        parsed_id = self._coerce_uuid(memory_id)
        if parsed_id is None:
            raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

        if memory_type in (None, MemoryType.WORKING):
            if await self.working.delete_working_memory(parsed_id):
                return True

        if memory_type in (None, MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL):
            try:
                if await self.long_term.delete(parsed_id, memory_type=memory_type):
                    return True
            except MemoryNotFoundError:
                pass

        raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

    async def delete_by_tags(
        self,
        tags: list[str],
        memory_type: MemoryType | None = None,
    ) -> int:
        """Delete memories matching any of the given tags.

        Args:
            tags: List of target tags.
            memory_type: Optional memory type filter.

        Returns:
            Number of deleted memory records.
        """
        return await self.long_term.delete_by_tags(tags, memory_type=memory_type)

    # Retrieval, Context, Consolidation, and Persistence Operations

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        task_id: uuid.UUID | None = None,
        memory_types: Sequence[MemoryType] | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Perform unified retrieval across all memory systems.

        Args:
            query: Query text string.
            session_id: Session ID string.
            task_id: Task UUID.
            memory_types: Target memory types.
            tags: Filter tags.
            limit: Maximum items to return.

        Returns:
            List of SearchResult objects.
        """
        return await self.retriever.retrieve(
            query=query,
            session_id=session_id,
            task_id=task_id,
            memory_types=memory_types,
            tags=tags,
            limit=limit,
        )

    async def build_context(
        self,
        query: str,
        session_id: str | None = None,
        task_id: uuid.UUID | None = None,
        working_memory_item: WorkingMemoryItem | None = None,
        short_term_items: Sequence[ShortTermMemoryItem] | None = None,
        token_budget: int | None = None,
    ) -> str:
        """Build token-budgeted prompt context for LLMs.

        Args:
            query: Retrieval query.
            session_id: Session ID string.
            task_id: Task UUID.
            working_memory_item: Active working memory item.
            short_term_items: Explicit short-term items list.
            token_budget: Token limit integer.

        Returns:
            Formatted Markdown context string.
        """
        return await self.context_builder.build_context(
            query=query,
            session_id=session_id,
            task_id=task_id,
            working_memory_item=working_memory_item,
            short_term_items=short_term_items,
            token_budget=token_budget,
        )

    async def consolidate(self, limit: int = 50) -> list[ConsolidationResult]:
        """Run pending working memory consolidation jobs.

        Args:
            limit: Maximum tasks to consolidate.

        Returns:
            List of ConsolidationResult summaries.
        """
        return await self.consolidator.run_pending_consolidations(limit=limit)

    async def save_to_disk(self, file_path: str | Path | None = None) -> None:
        """Save memory states to disk via JSON file persistence.

        Args:
            file_path: Optional destination path override.
        """
        persistence = (
            JSONFileMemoryPersistence(file_path) if file_path else self.persistence
        )
        if persistence is None:
            raise MemoryServiceError("No persistence file path configured")

        working_items = await self.working.list_working_memories(limit=10000)
        episodes = await self.episodic.get_recent_episodes(limit=10000)
        facts = await self.semantic.list_facts(limit=10000)
        procedures = await self.procedural.list_procedures(limit=10000)

        await persistence.save(
            working_items=working_items,
            episodes=episodes,
            semantic_entities=facts,
            procedures=procedures,
        )

    async def load_from_disk(self, file_path: str | Path | None = None) -> None:
        """Load memory states from disk via JSON file persistence.

        Args:
            file_path: Optional source path override.
        """
        persistence = (
            JSONFileMemoryPersistence(file_path) if file_path else self.persistence
        )
        if persistence is None:
            raise MemoryServiceError("No persistence file path configured")

        data = await persistence.load()
        for wi in data.get("working_items", []):
            await self.working._repo.create(wi)
        for ep in data.get("episodes", []):
            await self.episodic._repo.create(ep)
        for se in data.get("semantic_entities", []):
            await self.semantic._repo.create(se)
        for pr in data.get("procedures", []):
            await self.procedural._repo.create(pr)

    @staticmethod
    def _coerce_uuid(value: Any) -> uuid.UUID | None:
        if value is None or value == "":
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))
