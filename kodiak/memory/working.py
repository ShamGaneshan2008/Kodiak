
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class WorkingMemoryStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    CONSOLIDATED = "consolidated"


class WorkingMemoryItem(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task_id: uuid.UUID
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    status: WorkingMemoryStatus = WorkingMemoryStatus.ACTIVE
    outcome: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkingMemoryNotFoundError(Exception):
    pass


@runtime_checkable
class WorkingMemoryRepository(Protocol):
    async def create(self, item: WorkingMemoryItem) -> WorkingMemoryItem: ...

    async def get_by_task_id(self, task_id: uuid.UUID) -> WorkingMemoryItem | None: ...

    async def get_active(self) -> list[WorkingMemoryItem]: ...

    async def update(self, item: WorkingMemoryItem) -> WorkingMemoryItem: ...

    async def get_unconsolidated_tasks(self, limit: int) -> list[dict[str, Any]]: ...


class WorkingMemory:
    def __init__(self, repository: WorkingMemoryRepository) -> None:
        self._repo = repository

    async def create_working_memory(
        self,
        task_id: uuid.UUID,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        item = WorkingMemoryItem(
            task_id=task_id,
            goal=goal,
            context=context or {},
        )
        created = await self._repo.create(item)
        logger.info(
            "working_memory_created",
            task_id=str(task_id),
            memory_id=str(created.id),
        )
        return created

    async def get_working_memory(self, task_id: uuid.UUID) -> WorkingMemoryItem:
        item = await self._repo.get_by_task_id(task_id)
        if item is None:
            raise WorkingMemoryNotFoundError(f"Working memory for task {task_id} not found")
        return item

    async def update_scratchpad(
        self,
        task_id: uuid.UUID,
        key: str,
        value: Any,
    ) -> WorkingMemoryItem:
        item = await self.get_working_memory(task_id)
        item.scratchpad[key] = value
        item.updated_at = datetime.now(timezone.utc)
        updated = await self._repo.update(item)
        logger.debug(
            "scratchpad_updated",
            task_id=str(task_id),
            key=key,
        )
        return updated

    async def append_to_scratchpad(
        self,
        task_id: uuid.UUID,
        key: str,
        value: Any,
    ) -> WorkingMemoryItem:
        item = await self.get_working_memory(task_id)
        if key not in item.scratchpad or not isinstance(item.scratchpad[key], list):
            item.scratchpad[key] = []
        item.scratchpad[key].append(value)
        item.updated_at = datetime.now(timezone.utc)
        return await self._repo.update(item)

    async def set_outcome(self, task_id: uuid.UUID, outcome: str) -> WorkingMemoryItem:
        item = await self.get_working_memory(task_id)
        item.outcome = outcome
        item.updated_at = datetime.now(timezone.utc)
        logger.info(
            "working_memory_outcome_set",
            task_id=str(task_id),
            outcome=outcome,
        )
        return await self._repo.update(item)

    async def complete_working_memory(self, task_id: uuid.UUID) -> WorkingMemoryItem:
        item = await self.get_working_memory(task_id)
        item.status = WorkingMemoryStatus.COMPLETED
        item.updated_at = datetime.now(timezone.utc)
        logger.info("working_memory_completed", task_id=str(task_id))
        return await self._repo.update(item)

    async def abandon_working_memory(self, task_id: uuid.UUID) -> WorkingMemoryItem:
        item = await self.get_working_memory(task_id)
        item.status = WorkingMemoryStatus.ABANDONED
        item.updated_at = datetime.now(timezone.utc)
        logger.info("working_memory_abandoned", task_id=str(task_id))
        return await self._repo.update(item)

    async def get_active_memories(self) -> list[WorkingMemoryItem]:
        return await self._repo.get_active()

    async def get_unconsolidated_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._repo.get_unconsolidated_tasks(limit)
