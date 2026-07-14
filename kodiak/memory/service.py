# kodiak/memory/service.py
"""Unified facade over Kodiak's episodic, semantic, and procedural memory stores."""

from __future__ import annotations

import uuid
from typing import Any, Final, Protocol, runtime_checkable

import structlog

from .episodic import Episode, EpisodeRepository, EpisodicMemory
from .errors import MemoryNotFoundError, MemoryServiceError
from .models import Memory, MemoryType, SearchResult
from .procedural import (
    Procedure,
    ProcedureRepository,
    ProcedureStep,
    ProceduralMemory,
)
from .semantic import SemanticEntity, SemanticRepository, SemanticMemory

logger = structlog.get_logger(__name__)

__all__ = ["MemoryService"]

_DEFAULT_SEARCH_LIMIT: Final[int] = 10
_DEFAULT_LIST_LIMIT: Final[int] = 100


@runtime_checkable
class ListableEpisodeRepository(EpisodeRepository, Protocol):
    """Episode repository extended with listing and deletion support."""

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Episode]: ...

    async def delete(self, episode_id: uuid.UUID) -> bool: ...


@runtime_checkable
class ListableSemanticRepository(SemanticRepository, Protocol):
    """Semantic repository extended with listing support."""

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[SemanticEntity]: ...


@runtime_checkable
class TaggableProcedureRepository(ProcedureRepository, Protocol):
    """Procedure repository extended with listing, deletion, and tag lookup."""

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Procedure]: ...

    async def delete(self, procedure_id: uuid.UUID) -> bool: ...

    async def list_by_tags(self, tags: list[str]) -> list[Procedure]: ...


class MemoryService:
    """Unified CRUD and search interface over Kodiak's long-term memory stores.

    Working memory is intentionally excluded: it is task-scoped and transient,
    managed exclusively through consolidation rather than direct CRUD.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        episodic_repository: ListableEpisodeRepository,
        semantic_memory: SemanticMemory,
        semantic_repository: ListableSemanticRepository,
        procedural_memory: ProceduralMemory,
        procedural_repository: TaggableProcedureRepository,
    ) -> None:
        self._episodic = episodic_memory
        self._episodic_repo = episodic_repository
        self._semantic = semantic_memory
        self._semantic_repo = semantic_repository
        self._procedural = procedural_memory
        self._procedural_repo = procedural_repository

    async def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory record.

        Args:
            content: Primary text of the memory. Interpreted as a semantic
                fact, an episode goal, or a procedure description depending
                on `memory_type`.
            memory_type: Which store to write the memory into.
            tags: Only persisted for procedural memories; episodic and
                semantic records have no native tag field.
            metadata: Type-specific extras. Recognized keys:
                episodic - `outcome`, `task_id`, `steps`, `context`.
                semantic - `category`, `source_task_id`, `confidence`.
                procedural - `name`, `steps`.

        Returns:
            The normalized `Memory` record that was created.

        Raises:
            MemoryServiceError: If the underlying store rejects the write.
        """
        tags = tags or []
        metadata = metadata or {}

        try:
            if memory_type is MemoryType.EPISODIC:
                episode = await self._episodic.create_episode(
                    goal=content,
                    outcome=str(metadata.get("outcome", "")),
                    task_id=metadata.get("task_id"),
                    context=metadata.get("context"),
                    steps=metadata.get("steps"),
                )
                return self._episode_to_memory(episode)

            if memory_type is MemoryType.PROCEDURAL:
                raw_steps = metadata.get("steps") or [content]
                steps = [
                    ProcedureStep(step_number=i + 1, action=str(step))
                    for i, step in enumerate(raw_steps)
                ]
                procedure = await self._procedural.create_procedure(
                    name=str(metadata.get("name", content[:80])),
                    description=content,
                    steps=steps,
                    tags=tags,
                )
                return self._procedure_to_memory(procedure)

            entity = await self._semantic.store_fact(
                content=content,
                category=str(metadata.get("category", "general")),
                source_task_id=metadata.get("source_task_id"),
                confidence=float(metadata.get("confidence", 1.0)),
            )
            return self._semantic_to_memory(entity)
        except Exception as exc:
            logger.exception("memory_add_failed", memory_type=str(memory_type))
            raise MemoryServiceError(f"Failed to add {memory_type} memory") from exc

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> list[SearchResult]:
        """Search across one or all memory stores.

        Args:
            query: Free-text search query.
            memory_type: Restrict the search to a single store. Searches all
                stores when omitted.
            limit: Maximum number of results to return overall.

        Returns:
            Results sorted by descending relevance score, truncated to `limit`.
        """
        results: list[SearchResult] = []

        if memory_type in (None, MemoryType.EPISODIC):
            for r in await self._episodic.search_episodes(query, limit=limit):
                results.append(
                    SearchResult(
                        memory=self._episode_to_memory(r.episode),
                        relevance_score=r.relevance_score,
                    )
                )

        if memory_type in (None, MemoryType.SEMANTIC):
            for r in await self._semantic.search_facts(query, limit=limit):
                results.append(
                    SearchResult(
                        memory=self._semantic_to_memory(r.entity),
                        relevance_score=r.relevance_score,
                    )
                )

        if memory_type in (None, MemoryType.PROCEDURAL):
            for r in await self._procedural.search_procedures(query, limit=limit):
                results.append(
                    SearchResult(
                        memory=self._procedure_to_memory(r.procedure),
                        relevance_score=r.relevance_score,
                    )
                )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    async def list(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = _DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories, optionally filtered by store and tags.

        Args:
            memory_type: Restrict to a single store. Lists all stores when omitted.
            tags: Only return memories containing at least one of these tags.
                Only procedural memories carry native tags; semantic memories
                are matched against their `category` as a pseudo-tag.
            limit: Maximum number of records to return per store.
            offset: Number of records to skip per store.

        Returns:
            Normalized memory records, newest first.
        """
        memories: list[Memory] = []

        if memory_type in (None, MemoryType.EPISODIC):
            episodes = await self._episodic_repo.list_all(limit=limit, offset=offset)
            memories.extend(self._episode_to_memory(e) for e in episodes)

        if memory_type in (None, MemoryType.SEMANTIC):
            entities = await self._semantic_repo.list_all(limit=limit, offset=offset)
            memories.extend(self._semantic_to_memory(e) for e in entities)

        if memory_type in (None, MemoryType.PROCEDURAL):
            procedures = await self._procedural_repo.list_all(limit=limit, offset=offset)
            memories.extend(self._procedure_to_memory(p) for p in procedures)

        if tags:
            wanted = set(tags)
            memories = [m for m in memories if wanted & set(m.tags)]

        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    async def delete(
        self,
        memory_id: uuid.UUID,
        memory_type: MemoryType | None = None,
    ) -> bool:
        """Delete a single memory by id.

        Args:
            memory_id: Identifier of the memory to remove.
            memory_type: Store to delete from. All stores are tried in turn
                when omitted.

        Returns:
            True once the memory has been deleted.

        Raises:
            MemoryNotFoundError: If no matching memory exists in the target store(s).
        """
        stores: list[tuple[MemoryType, Any]] = []
        if memory_type in (None, MemoryType.EPISODIC):
            stores.append((MemoryType.EPISODIC, self._episodic_repo))
        if memory_type in (None, MemoryType.SEMANTIC):
            stores.append((MemoryType.SEMANTIC, self._semantic_repo))
        if memory_type in (None, MemoryType.PROCEDURAL):
            stores.append((MemoryType.PROCEDURAL, self._procedural_repo))

        for m_type, repo in stores:
            deleted = await repo.delete(memory_id)
            if deleted:
                logger.info(
                    "memory_deleted", memory_id=str(memory_id), memory_type=str(m_type)
                )
                return True

        raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

    async def delete_by_tags(
        self,
        tags: list[str],
        memory_type: MemoryType | None = None,
    ) -> int:
        """Delete every memory matching any of the given tags.

        Args:
            tags: Tags to match. Only procedural memories carry native tags;
                semantic memories are matched against their `category`.
                Episodic memories have no tag concept and are never affected.
            memory_type: Restrict deletion to a single store.

        Returns:
            The number of memories deleted.
        """
        if not tags:
            return 0

        deleted_count = 0

        if memory_type in (None, MemoryType.PROCEDURAL):
            matches = await self._procedural_repo.list_by_tags(tags)
            for procedure in matches:
                if await self._procedural_repo.delete(procedure.id):
                    deleted_count += 1

        if memory_type in (None, MemoryType.SEMANTIC):
            wanted = set(tags)
            entities = await self._semantic_repo.list_all(limit=_DEFAULT_LIST_LIMIT)
            for entity in entities:
                if entity.category in wanted and await self._semantic_repo.delete(entity.id):
                    deleted_count += 1

        logger.info("memories_deleted_by_tags", tags=tags, count=deleted_count)
        return deleted_count

    @staticmethod
    def _episode_to_memory(episode: Episode) -> Memory:
        return Memory(
            id=episode.id,
            type=MemoryType.EPISODIC,
            title=episode.goal,
            content=episode.outcome,
            tags=[],
            metadata={
                "context": episode.context,
                "steps": episode.steps,
                "task_id": str(episode.task_id) if episode.task_id else None,
            },
            confidence=episode.significance,
            created_at=episode.created_at,
        )

    @staticmethod
    def _semantic_to_memory(entity: SemanticEntity) -> Memory:
        return Memory(
            id=entity.id,
            type=MemoryType.SEMANTIC,
            title=entity.content[:80],
            content=entity.content,
            tags=[entity.category] if entity.category else [],
            metadata=dict(entity.metadata),
            confidence=entity.confidence,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    @staticmethod
    def _procedure_to_memory(procedure: Procedure) -> Memory:
        return Memory(
            id=procedure.id,
            type=MemoryType.PROCEDURAL,
            title=procedure.name,
            content=procedure.description,
            tags=list(procedure.tags),
            metadata={
                "steps": [s.action for s in procedure.steps],
                "success_count": procedure.success_count,
                "failure_count": procedure.failure_count,
                "success_rate": procedure.success_rate,
            },
            confidence=procedure.success_rate,
            created_at=procedure.created_at,
            updated_at=procedure.updated_at,
        )