# kodiak/memory/service.py
"""Unified facade over Kodiak's episodic, semantic, and procedural memories."""

from __future__ import annotations

import uuid
from typing import Any, Final

import structlog

from .episodic import Episode
from .errors import MemoryNotFoundError, MemoryServiceError
from .models import Memory, MemoryType, SearchResult
from .procedural import Procedure, ProcedureStep
from .semantic import SemanticEntity

logger = structlog.get_logger(__name__)

__all__ = ["MemoryService"]

_DEFAULT_SEARCH_LIMIT: Final[int] = 10
_DEFAULT_LIST_LIMIT: Final[int] = 100


class MemoryService:
    """Unified CRUD and search interface over Kodiak's memory models.

    The repository contains domain models for episodic, semantic, and
    procedural memory, but no concrete repository implementations. This service
    therefore owns the in-process stores directly instead of requiring
    repository objects that do not exist in the project.
    """

    def __init__(self) -> None:
        self._episodes: dict[uuid.UUID, Episode] = {}
        self._entities: dict[uuid.UUID, SemanticEntity] = {}
        self._procedures: dict[uuid.UUID, Procedure] = {}

    async def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Add a new memory record."""
        tags = tags or []
        metadata = metadata or {}

        try:
            if memory_type is MemoryType.EPISODIC:
                episode = Episode(
                    goal=content,
                    outcome=str(metadata.get("outcome", "")),
                    task_id=self._coerce_uuid(metadata.get("task_id")),
                    context=dict(metadata.get("context") or {}),
                    steps=[str(step) for step in metadata.get("steps", [])],
                )
                self._episodes[episode.id] = episode
                logger.info("episode_created", episode_id=str(episode.id))
                return self._episode_to_memory(episode)

            if memory_type is MemoryType.PROCEDURAL:
                raw_steps = metadata.get("steps") or [content]
                steps = [
                    ProcedureStep(step_number=i + 1, action=str(step))
                    for i, step in enumerate(raw_steps)
                ]
                procedure = Procedure(
                    name=str(metadata.get("name", content[:80])),
                    description=content,
                    steps=steps,
                    tags=tags,
                )
                self._procedures[procedure.id] = procedure
                logger.info("procedure_created", procedure_id=str(procedure.id))
                return self._procedure_to_memory(procedure)

            entity = SemanticEntity(
                content=content,
                category=str(metadata.get("category", "general")),
                source_task_id=self._coerce_uuid(metadata.get("source_task_id")),
                confidence=float(metadata.get("confidence", 1.0)),
                metadata={
                    str(key): str(value)
                    for key, value in metadata.items()
                    if key not in {"category", "source_task_id", "confidence"}
                },
            )
            self._entities[entity.id] = entity
            logger.info("fact_stored", entity_id=str(entity.id))
            return self._semantic_to_memory(entity)
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
        """Search across one or all memory stores."""
        wanted_tags = set(tags or [])
        results: list[SearchResult] = []

        if memory_type in (None, MemoryType.EPISODIC):
            for episode in self._episodes.values():
                memory = self._episode_to_memory(episode)
                if self._matches_tags(memory, wanted_tags):
                    score = self._score(query, episode.goal, episode.outcome, *episode.steps)
                    if score > 0:
                        results.append(SearchResult(memory=memory, relevance_score=score))

        if memory_type in (None, MemoryType.SEMANTIC):
            for entity in self._entities.values():
                memory = self._semantic_to_memory(entity)
                if self._matches_tags(memory, wanted_tags):
                    score = self._score(query, entity.content, entity.category)
                    if score > 0:
                        results.append(SearchResult(memory=memory, relevance_score=score))

        if memory_type in (None, MemoryType.PROCEDURAL):
            for procedure in self._procedures.values():
                memory = self._procedure_to_memory(procedure)
                if self._matches_tags(memory, wanted_tags):
                    score = self._score(
                        query,
                        procedure.name,
                        procedure.description,
                        *procedure.tags,
                        *(step.action for step in procedure.steps),
                    )
                    if score > 0:
                        results.append(SearchResult(memory=memory, relevance_score=score))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    async def list(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = _DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories, optionally filtered by store and tags."""
        memories: list[Memory] = []

        if memory_type in (None, MemoryType.EPISODIC):
            memories.extend(self._episode_to_memory(e) for e in self._episodes.values())

        if memory_type in (None, MemoryType.SEMANTIC):
            memories.extend(self._semantic_to_memory(e) for e in self._entities.values())

        if memory_type in (None, MemoryType.PROCEDURAL):
            memories.extend(self._procedure_to_memory(p) for p in self._procedures.values())

        wanted = set(tags or [])
        memories = [m for m in memories if self._matches_tags(m, wanted)]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[offset : offset + limit]

    async def delete(
        self,
        memory_id: uuid.UUID | str,
        memory_type: MemoryType | None = None,
    ) -> bool:
        """Delete a single memory by id."""
        parsed_id = self._coerce_uuid(memory_id)
        if parsed_id is None:
            raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

        if memory_type in (None, MemoryType.EPISODIC) and self._episodes.pop(parsed_id, None):
            logger.info("memory_deleted", memory_id=str(parsed_id), memory_type=str(MemoryType.EPISODIC))
            return True

        if memory_type in (None, MemoryType.SEMANTIC) and self._entities.pop(parsed_id, None):
            logger.info("memory_deleted", memory_id=str(parsed_id), memory_type=str(MemoryType.SEMANTIC))
            return True

        if memory_type in (None, MemoryType.PROCEDURAL) and self._procedures.pop(parsed_id, None):
            logger.info("memory_deleted", memory_id=str(parsed_id), memory_type=str(MemoryType.PROCEDURAL))
            return True

        raise MemoryNotFoundError(str(memory_id), memory_type=str(memory_type) if memory_type else None)

    async def delete_by_tags(
        self,
        tags: list[str],
        memory_type: MemoryType | None = None,
    ) -> int:
        """Delete every memory matching any of the given tags."""
        if not tags:
            return 0

        wanted = set(tags)
        deleted_count = 0

        if memory_type in (None, MemoryType.SEMANTIC):
            for entity_id, entity in list(self._entities.items()):
                if entity.category in wanted:
                    del self._entities[entity_id]
                    deleted_count += 1

        if memory_type in (None, MemoryType.PROCEDURAL):
            for procedure_id, procedure in list(self._procedures.items()):
                if wanted & set(procedure.tags):
                    del self._procedures[procedure_id]
                    deleted_count += 1

        logger.info("memories_deleted_by_tags", tags=tags, count=deleted_count)
        return deleted_count

    @staticmethod
    def _matches_tags(memory: Memory, tags: set[str]) -> bool:
        return not tags or bool(tags & set(memory.tags))

    @staticmethod
    def _score(query: str, *parts: str) -> float:
        text = " ".join(part for part in parts if part).lower()
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return 1.0
        matches = sum(1 for term in terms if term in text)
        return matches / len(terms)

    @staticmethod
    def _coerce_uuid(value: Any) -> uuid.UUID | None:
        if value is None or value == "":
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))

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
