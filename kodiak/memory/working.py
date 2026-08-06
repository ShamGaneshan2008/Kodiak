# kodiak/memory/working.py
"""Working Memory component for active task goals, state, context, and scratchpad."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from .errors import WorkingMemoryNotFoundError
from .models import Memory, MemoryType

logger = structlog.get_logger(__name__)

__all__ = [
    "WorkingMemoryStatus",
    "WorkingMemoryItem",
    "WorkingMemoryNotFoundError",
    "WorkingMemoryRepository",
    "WorkingMemory",
]


class WorkingMemoryStatus(StrEnum):
    """Execution status for an active working memory item."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    CONSOLIDATED = "consolidated"


class WorkingMemoryItem(BaseModel):
    """Working memory record tracking task progress, context, and scratchpad."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task_id: uuid.UUID
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    status: WorkingMemoryStatus = WorkingMemoryStatus.ACTIVE
    outcome: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class WorkingMemoryRepository(Protocol):
    """Protocol for working memory persistence implementations."""

    async def create(self, item: WorkingMemoryItem) -> WorkingMemoryItem: ...

    async def get_by_task_id(self, task_id: uuid.UUID) -> WorkingMemoryItem | None: ...

    async def get_by_id(self, memory_id: uuid.UUID) -> WorkingMemoryItem | None: ...

    async def get_active(self) -> list[WorkingMemoryItem]: ...

    async def update(self, item: WorkingMemoryItem) -> WorkingMemoryItem: ...

    async def delete(self, task_id: uuid.UUID) -> bool: ...

    async def list_all(self, limit: int = 100) -> list[WorkingMemoryItem]: ...

    async def get_unconsolidated_tasks(self, limit: int) -> list[dict[str, Any]]: ...


class WorkingMemory:
    """Manager for task working memory items."""

    def __init__(self, repository: WorkingMemoryRepository | None = None) -> None:
        """Initialize working memory manager.

        Args:
            repository: Underlying storage repository. Defaults to InMemoryWorkingMemoryRepository.
        """
        if repository is None:
            from .persistence import InMemoryWorkingMemoryRepository

            repository = InMemoryWorkingMemoryRepository()
        self._repo = repository

    async def create_working_memory(
        self,
        task_id: uuid.UUID,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        """Create a new working memory item for a task.

        Args:
            task_id: Associated task UUID.
            goal: Primary task goal string.
            context: Optional contextual parameters.

        Returns:
            Created WorkingMemoryItem.
        """
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
        """Get working memory for a task.

        Args:
            task_id: Task UUID.

        Returns:
            WorkingMemoryItem.

        Raises:
            WorkingMemoryNotFoundError: If no record exists for task_id.
        """
        item = await self._repo.get_by_task_id(task_id)
        if item is None:
            raise WorkingMemoryNotFoundError(str(task_id))
        return item

    async def get_by_id(self, memory_id: uuid.UUID) -> WorkingMemoryItem | None:
        """Get working memory item by its memory ID.

        Args:
            memory_id: Memory UUID.

        Returns:
            WorkingMemoryItem if found, else None.
        """
        return await self._repo.get_by_id(memory_id)

    async def update_scratchpad(
        self,
        task_id: uuid.UUID,
        key: str,
        value: Any,
    ) -> WorkingMemoryItem:
        """Update a key-value pair in task scratchpad.

        Args:
            task_id: Task UUID.
            key: Scratchpad entry key.
            value: Value to associate with key.

        Returns:
            Updated WorkingMemoryItem.
        """
        item = await self.get_working_memory(task_id)
        item.scratchpad[key] = value
        item.updated_at = datetime.now(UTC)
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
        """Append a item to a scratchpad list entry.

        Args:
            task_id: Task UUID.
            key: Scratchpad entry key.
            value: Element to append.

        Returns:
            Updated WorkingMemoryItem.
        """
        item = await self.get_working_memory(task_id)
        if key not in item.scratchpad or not isinstance(item.scratchpad[key], list):
            item.scratchpad[key] = []
        item.scratchpad[key].append(value)
        item.updated_at = datetime.now(UTC)
        return await self._repo.update(item)

    async def set_outcome(self, task_id: uuid.UUID, outcome: str) -> WorkingMemoryItem:
        """Set task outcome string.

        Args:
            task_id: Task UUID.
            outcome: Text summary of task execution outcome.

        Returns:
            Updated WorkingMemoryItem.
        """
        item = await self.get_working_memory(task_id)
        item.outcome = outcome
        item.updated_at = datetime.now(UTC)
        logger.info(
            "working_memory_outcome_set",
            task_id=str(task_id),
            outcome=outcome,
        )
        return await self._repo.update(item)

    async def complete_working_memory(self, task_id: uuid.UUID) -> WorkingMemoryItem:
        """Mark task working memory as completed.

        Args:
            task_id: Task UUID.

        Returns:
            Updated WorkingMemoryItem.
        """
        item = await self.get_working_memory(task_id)
        item.status = WorkingMemoryStatus.COMPLETED
        item.updated_at = datetime.now(UTC)
        logger.info("working_memory_completed", task_id=str(task_id))
        return await self._repo.update(item)

    async def abandon_working_memory(self, task_id: uuid.UUID) -> WorkingMemoryItem:
        """Mark task working memory as abandoned.

        Args:
            task_id: Task UUID.

        Returns:
            Updated WorkingMemoryItem.
        """
        item = await self.get_working_memory(task_id)
        item.status = WorkingMemoryStatus.ABANDONED
        item.updated_at = datetime.now(UTC)
        logger.info("working_memory_abandoned", task_id=str(task_id))
        return await self._repo.update(item)

    async def delete_working_memory(self, task_id: uuid.UUID) -> bool:
        """Delete working memory for a task.

        Args:
            task_id: Task UUID.

        Returns:
            True if deleted, else False.
        """
        return await self._repo.delete(task_id)

    async def get_active_memories(self) -> list[WorkingMemoryItem]:
        """Fetch all active task working memories.

        Returns:
            List of active WorkingMemoryItem instances.
        """
        return await self._repo.get_active()

    async def list_working_memories(self, limit: int = 100) -> list[WorkingMemoryItem]:
        """List stored working memory items.

        Args:
            limit: Maximum items to return.

        Returns:
            List of WorkingMemoryItem instances.
        """
        return await self._repo.list_all(limit=limit)

    async def get_unconsolidated_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch tasks awaiting consolidation.

        Args:
            limit: Maximum tasks to return.

        Returns:
            List of task attribute dictionaries.
        """
        return await self._repo.get_unconsolidated_tasks(limit)

    @staticmethod
    def to_memory(item: WorkingMemoryItem) -> Memory:
        """Convert WorkingMemoryItem to normalized Memory model.

        Args:
            item: WorkingMemoryItem record.

        Returns:
            Normalized Memory instance.
        """
        return Memory(
            id=item.id,
            type=MemoryType.WORKING,
            title=f"Task {item.task_id}: {item.goal[:60]}",
            content=item.goal,
            tags=[item.status.value],
            metadata={
                "task_id": str(item.task_id),
                "context": item.context,
                "scratchpad": item.scratchpad,
                "status": item.status.value,
                "outcome": item.outcome,
            },
            confidence=1.0 if item.status == WorkingMemoryStatus.ACTIVE else 0.8,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
