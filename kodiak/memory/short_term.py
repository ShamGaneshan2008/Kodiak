# kodiak/memory/short_term.py
"""Short-Term Memory component for managing session history and interaction buffers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from .errors import ShortTermMemoryError
from .models import Memory, MemoryType

logger = structlog.get_logger(__name__)

__all__ = [
    "ShortTermMemoryItem",
    "ShortTermMemoryRepository",
    "ShortTermMemory",
]


class ShortTermMemoryItem(BaseModel):
    """Single interactive entry stored in Short-Term Memory."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: str
    role: str = "user"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_expired(self) -> bool:
        """Check if item has passed its TTL duration."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now(UTC) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


@runtime_checkable
class ShortTermMemoryRepository(Protocol):
    """Protocol for short-term memory persistence backends."""

    async def add(self, session_id: str, item_data: dict[str, Any]) -> dict[str, Any]: ...

    async def get_session(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]: ...

    async def clear_session(self, session_id: str) -> bool: ...


class ShortTermMemory:
    """Manager for session interaction context and short-term event memory."""

    def __init__(
        self,
        repository: ShortTermMemoryRepository | None = None,
        max_history_length: int = 100,
    ) -> None:
        """Initialize Short-Term Memory manager.

        Args:
            repository: Underlying storage backend. Defaults to InMemoryShortTermMemoryRepository.
            max_history_length: Default maximum records per session.
        """
        if repository is None:
            from .persistence import InMemoryShortTermMemoryRepository

            repository = InMemoryShortTermMemoryRepository(max_items_per_session=max_history_length)
        self._repo = repository
        self._max_history_length = max_history_length

    async def add_item(
        self,
        session_id: str,
        content: str,
        role: str = "user",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> ShortTermMemoryItem:
        """Add an interaction item to session short-term memory.

        Args:
            session_id: Target session identifier string.
            content: Message content or event description.
            role: Sender role (e.g. 'user', 'assistant', 'system', 'tool').
            metadata: Associated contextual parameters.
            ttl_seconds: Expiration lifespan in seconds.

        Returns:
            Created ShortTermMemoryItem.
        """
        try:
            item = ShortTermMemoryItem(
                session_id=session_id,
                role=role,
                content=content,
                metadata=metadata or {},
                ttl_seconds=ttl_seconds,
            )
            data = item.model_dump(mode="json")
            await self._repo.add(session_id, data)
            logger.debug(
                "short_term_item_added",
                session_id=session_id,
                role=role,
                item_id=str(item.id),
            )
            return item
        except Exception as exc:
            logger.exception("short_term_add_failed", session_id=session_id)
            raise ShortTermMemoryError(f"Failed to add short-term memory item for session {session_id}") from exc

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[ShortTermMemoryItem]:
        """Fetch active (non-expired) recent items for a session.

        Args:
            session_id: Session identifier string.
            limit: Maximum number of items to return.

        Returns:
            List of ShortTermMemoryItem records sorted chronologically.
        """
        raw_items = await self._repo.get_session(session_id, limit=limit)
        items: list[ShortTermMemoryItem] = []
        for data in raw_items:
            item = ShortTermMemoryItem.model_validate(data)
            if not item.is_expired:
                items.append(item)
        return items

    async def clear_session(self, session_id: str) -> bool:
        """Clear all short-term items for a session.

        Args:
            session_id: Target session identifier string.

        Returns:
            True if session existed and was cleared, else False.
        """
        cleared = await self._repo.clear_session(session_id)
        if cleared:
            logger.info("short_term_session_cleared", session_id=session_id)
        return cleared

    @staticmethod
    def to_memory(item: ShortTermMemoryItem) -> Memory:
        """Convert ShortTermMemoryItem to normalized Memory model.

        Args:
            item: ShortTermMemoryItem record.

        Returns:
            Normalized Memory instance.
        """
        return Memory(
            id=item.id,
            type=MemoryType.SHORT_TERM,
            title=f"[{item.role.upper()}] Session {item.session_id}: {item.content[:50]}",
            content=item.content,
            tags=[item.session_id, item.role],
            metadata={
                "session_id": item.session_id,
                "role": item.role,
                "ttl_seconds": item.ttl_seconds,
                **item.metadata,
            },
            confidence=1.0,
            created_at=item.created_at,
        )
