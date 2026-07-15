
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field
from typing import Any

logger = structlog.get_logger(__name__)


class FactNotFoundError(Exception):
    pass


class SemanticEntity(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str
    category: str = "general"
    source_task_id: uuid.UUID | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticSearchResult(BaseModel):
    entity: SemanticEntity
    relevance_score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class SemanticRepository(Protocol):
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


class SemanticMemory:
    def __init__(self, repository: SemanticRepository) -> None:
        self._repo = repository

    async def store_fact(
        self,
        content: str,
        category: str = "general",
        source_task_id: uuid.UUID | None = None,
        confidence: float = 1.0,
        embedding: list[float] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SemanticEntity:
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
        entity = await self._repo.get_by_id(entity_id)
        if entity is None:
            raise FactNotFoundError(f"Semantic entity {entity_id} not found")
        return entity

    async def search_facts(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[SemanticSearchResult]:
        return await self._repo.search(query, category, limit)

    async def update_fact(
            self,
            entity_id: uuid.UUID,
            content: str | None = None,
            category: str | None = None,
            confidence: float | None = None,
            embedding: list[float] | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> SemanticEntity:
        existing = await self.get_fact(entity_id)

        updated_entity = existing.model_copy(
            update={
                "content": (
                    content
                    if content is not None
                    else existing.content
                ),
                "category": (
                    category
                    if category is not None
                    else existing.category
                ),
                "confidence": (
                    confidence
                    if confidence is not None
                    else existing.confidence
                ),
                "embedding": (
                    embedding
                    if embedding is not None
                    else existing.embedding
                ),
                "metadata": (
                    metadata
                    if metadata is not None
                    else existing.metadata
                ),
                "updated_at": datetime.now(timezone.utc),
            }
        )

        result = await self._repo.update(updated_entity)

        logger.info(
            "fact_updated",
            entity_id=str(entity_id),
        )

        return result

    async def delete_fact(self, entity_id: uuid.UUID) -> bool:
        deleted = await self._repo.delete(entity_id)
        if deleted:
            logger.info("fact_deleted", entity_id=str(entity_id))
        return deleted
