# kodiak/memory/semantic.py
"""Semantic Memory component for storing facts, entity knowledge, and project rules."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from .errors import FactNotFoundError
from .models import Memory, MemoryType

logger = structlog.get_logger(__name__)

__all__ = [
    "FactNotFoundError",
    "SemanticEntity",
    "SemanticSearchResult",
    "SemanticRepository",
    "SemanticMemory",
]


class SemanticEntity(BaseModel):
    """Semantic fact or entity knowledge record."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str
    category: str = "general"
    source_task_id: uuid.UUID | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticSearchResult(BaseModel):
    """Semantic entity search result with relevance score."""

    entity: SemanticEntity
    relevance_score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class SemanticRepository(Protocol):
    """Protocol for semantic memory storage repository implementations."""

    async def create(self, entity: SemanticEntity) -> SemanticEntity: ...

    async def get_by_id(self, entity_id: uuid.UUID) -> SemanticEntity | None: ...

    async def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[SemanticSearchResult]: ...

    async def update(self, entity: SemanticEntity) -> SemanticEntity: ...

    async def delete(self, entity_id: uuid.UUID) -> bool: ...

    async def list_facts(
        self, category: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[SemanticEntity]: ...


class SemanticMemory:
    """Manager for semantic fact and rule knowledge base."""

    def __init__(self, repository: SemanticRepository | None = None) -> None:
        """Initialize semantic memory manager.

        Args:
            repository: Underlying semantic repository. Defaults to InMemorySemanticRepository.
        """
        if repository is None:
            from .persistence import InMemorySemanticRepository

            repository = InMemorySemanticRepository()
        self._repo = repository

    async def store_fact(
        self,
        content: str,
        category: str = "general",
        source_task_id: uuid.UUID | None = None,
        confidence: float = 1.0,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticEntity:
        """Store a new fact in semantic memory.

        Args:
            content: Fact statement content.
            category: Fact domain category label.
            source_task_id: Associated task UUID if extracted from execution.
            confidence: Confidence score in range [0.0, 1.0].
            embedding: Vector embedding list.
            metadata: Associated key-value metadata dictionary.

        Returns:
            Created SemanticEntity model.
        """
        entity = SemanticEntity(
            content=content,
            category=category,
            source_task_id=source_task_id,
            confidence=confidence,
            embedding=embedding,
            metadata=metadata or {},
        )
        created = await self._repo.create(entity)
        logger.info(
            "fact_stored",
            entity_id=str(created.id),
            category=category,
            content_length=len(content),
        )
        return created

    async def get_fact(self, entity_id: uuid.UUID) -> SemanticEntity:
        """Fetch semantic entity by ID.

        Args:
            entity_id: Entity UUID.

        Returns:
            SemanticEntity instance.

        Raises:
            FactNotFoundError: If fact is not found.
        """
        entity = await self._repo.get_by_id(entity_id)
        if entity is None:
            raise FactNotFoundError(str(entity_id))
        return entity

    async def search_facts(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[SemanticSearchResult]:
        """Search semantic facts matching a query string.

        Args:
            query: Query text string.
            category: Optional category filter.
            limit: Maximum items to return.

        Returns:
            List of SemanticSearchResult items.
        """
        return await self._repo.search(query, category, limit)

    async def list_facts(
        self,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SemanticEntity]:
        """List stored semantic facts.

        Args:
            category: Optional category filter.
            limit: Maximum records to return.
            offset: Offset index for pagination.

        Returns:
            List of SemanticEntity instances.
        """
        return await self._repo.list_facts(category=category, limit=limit, offset=offset)

    async def update_fact(
        self,
        entity_id: uuid.UUID,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticEntity:
        """Update an existing semantic fact entity.

        Args:
            entity_id: Entity UUID.
            content: New content string.
            category: New category label.
            confidence: Updated confidence float score.
            embedding: Updated embedding vector.
            metadata: Updated metadata dictionary.

        Returns:
            Updated SemanticEntity model.
        """
        existing = await self.get_fact(entity_id)

        updated_entity = existing.model_copy(
            update={
                "content": (content if content is not None else existing.content),
                "category": (category if category is not None else existing.category),
                "confidence": (confidence if confidence is not None else existing.confidence),
                "embedding": (embedding if embedding is not None else existing.embedding),
                "metadata": (metadata if metadata is not None else existing.metadata),
                "updated_at": datetime.now(UTC),
            }
        )

        result = await self._repo.update(updated_entity)
        logger.info("fact_updated", entity_id=str(entity_id))
        return result

    async def delete_fact(self, entity_id: uuid.UUID) -> bool:
        """Delete a semantic fact entity.

        Args:
            entity_id: Entity UUID.

        Returns:
            True if entity existed and was deleted, else False.
        """
        deleted = await self._repo.delete(entity_id)
        if deleted:
            logger.info("fact_deleted", entity_id=str(entity_id))
        return deleted

    @staticmethod
    def to_memory(entity: SemanticEntity) -> Memory:
        """Convert SemanticEntity model to normalized Memory representation.

        Args:
            entity: SemanticEntity instance.

        Returns:
            Normalized Memory model.
        """
        tags = []
        if entity.category:
            tags.append(entity.category)
        if "tags" in entity.metadata and isinstance(entity.metadata["tags"], list):
            for t in entity.metadata["tags"]:
                if t not in tags:
                    tags.append(str(t))

        return Memory(
            id=entity.id,
            type=MemoryType.SEMANTIC,
            title=entity.content[:80],
            content=entity.content,
            tags=tags,
            metadata={
                "category": entity.category,
                "source_task_id": str(entity.source_task_id) if entity.source_task_id else None,
                **entity.metadata,
            },
            confidence=entity.confidence,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
