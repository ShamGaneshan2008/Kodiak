# kodiak/memory/long_term.py
"""Long-Term Memory component orchestrating episodic, semantic, and procedural memories."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from .episodic import Episode, EpisodicMemory
from .errors import MemoryNotFoundError, MemoryServiceError
from .models import Memory, MemoryType, SearchResult
from .procedural import Procedure, ProceduralMemory, ProcedureStep
from .semantic import SemanticEntity, SemanticMemory

logger = structlog.get_logger(__name__)

__all__ = ["LongTermMemory"]


class LongTermMemory:
    """Composite facade over episodic, semantic, and procedural long-term stores."""

    def __init__(
        self,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        procedural: ProceduralMemory | None = None,
    ) -> None:
        """Initialize Long-Term Memory.

        Args:
            episodic: EpisodicMemory manager instance.
            semantic: SemanticMemory manager instance.
            procedural: ProceduralMemory manager instance.
        """
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a memory record into the appropriate long-term memory store.

        Args:
            content: Main text content to record.
            memory_type: Target MemoryType (EPISODIC, SEMANTIC, or PROCEDURAL).
            tags: List of tag label strings.
            metadata: Associated key-value attributes.

        Returns:
            Normalized Memory instance.

        Raises:
            MemoryServiceError: If memory_type is unsupported or store operation fails.
        """
        tags = tags or []
        metadata = metadata or {}

        try:
            if memory_type is MemoryType.EPISODIC:
                task_id = self._coerce_uuid(metadata.get("task_id"))
                steps = [str(step) for step in metadata.get("steps", [])]
                outcome = str(metadata.get("outcome", content))
                context = dict(metadata.get("context") or {})
                if tags:
                    context["tags"] = tags

                episode = await self.episodic.create_episode(
                    goal=content,
                    outcome=outcome,
                    task_id=task_id,
                    context=context,
                    steps=steps,
                )
                return self.episodic.to_memory(episode)

            if memory_type is MemoryType.PROCEDURAL:
                raw_steps = metadata.get("steps") or [content]
                steps_list: list[ProcedureStep] = []
                for i, raw in enumerate(raw_steps):
                    if isinstance(raw, dict):
                        steps_list.append(ProcedureStep.model_validate(raw))
                    else:
                        steps_list.append(
                            ProcedureStep(step_number=i + 1, action=str(raw))
                        )

                name = str(metadata.get("name", content[:80]))
                description = content
                procedure = await self.procedural.create_procedure(
                    name=name,
                    description=description,
                    steps=steps_list,
                    tags=tags,
                )
                return self.procedural.to_memory(procedure)

            if memory_type is MemoryType.SEMANTIC:
                category = str(metadata.get("category") or (tags[0] if tags else "general"))
                source_task_id = self._coerce_uuid(metadata.get("source_task_id"))
                confidence = float(metadata.get("confidence", 1.0))
                extra_meta = {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"category", "source_task_id", "confidence"}
                }
                if tags:
                    extra_meta["tags"] = tags

                entity = await self.semantic.store_fact(
                    content=content,
                    category=category,
                    source_task_id=source_task_id,
                    confidence=confidence,
                    metadata=extra_meta,
                )
                return self.semantic.to_memory(entity)

            raise MemoryServiceError(f"Unsupported long-term memory type: {memory_type}")
        except Exception as exc:
            logger.exception("long_term_add_failed", memory_type=str(memory_type))
            raise MemoryServiceError(f"Failed to add {memory_type} memory") from exc

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search long-term memory stores.

        Args:
            query: Query text string.
            memory_type: Scope search to specific MemoryType or search all if None.
            limit: Maximum search results.
            tags: Filter by tags if provided.

        Returns:
            List of SearchResult items ranked by relevance score.
        """
        wanted_tags = set(tags or [])
        results: list[SearchResult] = []

        if memory_type in (None, MemoryType.EPISODIC):
            ep_results = await self.episodic.search_episodes(query, limit=limit)
            for res in ep_results:
                mem = self.episodic.to_memory(res.episode)
                if self._matches_tags(mem, wanted_tags):
                    results.append(SearchResult(memory=mem, relevance_score=res.relevance_score))

        if memory_type in (None, MemoryType.SEMANTIC):
            sem_results = await self.semantic.search_facts(query, limit=limit)
            for res in sem_results:
                mem = self.semantic.to_memory(res.entity)
                if self._matches_tags(mem, wanted_tags):
                    results.append(SearchResult(memory=mem, relevance_score=res.relevance_score))

        if memory_type in (None, MemoryType.PROCEDURAL):
            proc_results = await self.procedural.search_procedures(query, limit=limit)
            for res in proc_results:
                mem = self.procedural.to_memory(res.procedure)
                if self._matches_tags(mem, wanted_tags):
                    results.append(SearchResult(memory=mem, relevance_score=res.relevance_score))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    async def list_memories(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List long-term memories.

        Args:
            memory_type: Scope to specific MemoryType or all if None.
            tags: Filter by tag labels.
            limit: Maximum items to return.
            offset: Offset index for pagination.

        Returns:
            List of normalized Memory instances.
        """
        memories: list[Memory] = []

        if memory_type in (None, MemoryType.EPISODIC):
            recent_ep = await self.episodic.get_recent_episodes(limit=limit, offset=offset)
            memories.extend(self.episodic.to_memory(e) for e in recent_ep)

        if memory_type in (None, MemoryType.SEMANTIC):
            facts = await self.semantic.list_facts(limit=limit, offset=offset)
            memories.extend(self.semantic.to_memory(f) for f in facts)

        if memory_type in (None, MemoryType.PROCEDURAL):
            procs = await self.procedural.list_procedures(limit=limit, offset=offset)
            memories.extend(self.procedural.to_memory(p) for p in procs)

        wanted = set(tags or [])
        memories = [m for m in memories if self._matches_tags(m, wanted)]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[offset : offset + limit]

    async def delete(
        self,
        memory_id: uuid.UUID | str,
        memory_type: MemoryType | None = None,
    ) -> bool:
        """Delete a long-term memory record by ID.

        Args:
            memory_id: Target memory UUID or string ID.
            memory_type: Optional scope hint.

        Returns:
            True if memory existed and was deleted.

        Raises:
            MemoryNotFoundError: If memory record could not be found.
        """
        parsed_id = self._coerce_uuid(memory_id)
        if parsed_id is None:
            raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

        if memory_type in (None, MemoryType.EPISODIC):
            if await self.episodic.delete_episode(parsed_id):
                return True

        if memory_type in (None, MemoryType.SEMANTIC):
            if await self.semantic.delete_fact(parsed_id):
                return True

        if memory_type in (None, MemoryType.PROCEDURAL):
            if await self.procedural.delete_procedure(parsed_id):
                return True

        raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

    async def delete_by_tags(
        self,
        tags: list[str],
        memory_type: MemoryType | None = None,
    ) -> int:
        """Delete long-term memories matching any of the given tags.

        Args:
            tags: List of target tags.
            memory_type: Optional memory type filter.

        Returns:
            Number of deleted memory records.
        """
        if not tags:
            return 0

        memories = await self.list_memories(memory_type=memory_type, tags=tags, limit=1000)
        deleted_count = 0
        for memory in memories:
            try:
                if await self.delete(memory.id, memory_type=memory.type):
                    deleted_count += 1
            except MemoryNotFoundError:
                continue

        logger.info("long_term_deleted_by_tags", tags=tags, count=deleted_count)
        return deleted_count

    @staticmethod
    def _matches_tags(memory: Memory, tags: set[str]) -> bool:
        return not tags or bool(tags & set(memory.tags))

    @staticmethod
    def _coerce_uuid(value: Any) -> uuid.UUID | None:
        if value is None or value == "":
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))
