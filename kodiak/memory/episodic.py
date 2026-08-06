# kodiak/memory/episodic.py
"""Episodic Memory component for storing and querying past execution episodes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from .errors import EpisodeNotFoundError
from .models import Memory, MemoryType

logger = structlog.get_logger(__name__)

__all__ = [
    "EpisodeNotFoundError",
    "Episode",
    "EpisodeSearchResult",
    "EpisodeRepository",
    "EpisodicMemory",
]


class Episode(BaseModel):
    """Past task execution episode record."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task_id: uuid.UUID | None = None
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    outcome: str
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EpisodeSearchResult(BaseModel):
    """Episode search match with relevance score."""

    episode: Episode
    relevance_score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class EpisodeRepository(Protocol):
    """Protocol for episodic memory repository implementations."""

    async def create(self, episode: Episode) -> Episode: ...

    async def get_by_id(self, episode_id: uuid.UUID) -> Episode | None: ...

    async def search(self, query: str, limit: int = 10) -> list[EpisodeSearchResult]: ...

    async def get_recent(self, limit: int = 10, offset: int = 0) -> list[Episode]: ...

    async def update_significance(self, episode_id: uuid.UUID, score: float) -> Episode: ...

    async def delete(self, episode_id: uuid.UUID) -> bool: ...


class EpisodicMemory:
    """Manager for episodic memory store."""

    def __init__(self, repository: EpisodeRepository | None = None) -> None:
        """Initialize episodic memory manager.

        Args:
            repository: Underlying episode storage repository. Defaults to InMemoryEpisodeRepository.
        """
        if repository is None:
            from .persistence import InMemoryEpisodeRepository

            repository = InMemoryEpisodeRepository()
        self._repo = repository

    async def create_episode(
        self,
        goal: str,
        outcome: str,
        task_id: uuid.UUID | None = None,
        context: dict[str, Any] | None = None,
        steps: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Episode:
        """Create and store a new episode record.

        Args:
            goal: Task goal string.
            outcome: Task outcome description.
            task_id: Optional associated task UUID.
            context: Context dictionary.
            steps: List of step description strings.
            embedding: Vector embedding float list.

        Returns:
            Created Episode model.
        """
        episode = Episode(
            task_id=task_id,
            goal=goal,
            context=context or {},
            steps=steps or [],
            outcome=outcome,
            embedding=embedding,
        )
        created = await self._repo.create(episode)
        logger.info(
            "episode_created",
            episode_id=str(created.id),
            goal=goal,
            outcome=outcome,
        )
        return created

    async def get_episode(self, episode_id: uuid.UUID) -> Episode:
        """Fetch episode by UUID.

        Args:
            episode_id: Episode UUID.

        Returns:
            Episode model.

        Raises:
            EpisodeNotFoundError: If episode does not exist.
        """
        episode = await self._repo.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(str(episode_id))
        return episode

    async def search_episodes(self, query: str, limit: int = 10) -> list[EpisodeSearchResult]:
        """Search episodes matching a query string.

        Args:
            query: Search query text.
            limit: Maximum matches to return.

        Returns:
            List of EpisodeSearchResult matches.
        """
        return await self._repo.search(query, limit)

    async def get_recent_episodes(self, limit: int = 10, offset: int = 0) -> list[Episode]:
        """Get recent episodes sorted by creation timestamp.

        Args:
            limit: Maximum items to return.
            offset: Offset index for pagination.

        Returns:
            List of Episode items.
        """
        return await self._repo.get_recent(limit=limit, offset=offset)

    async def update_significance(self, episode_id: uuid.UUID, score: float) -> Episode:
        """Update episode significance rating.

        Args:
            episode_id: Episode UUID.
            score: Significance score float in range [0.0, 1.0].

        Returns:
            Updated Episode item.

        Raises:
            ValueError: If score is out of range.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("Significance score must be between 0.0 and 1.0")
        episode = await self._repo.update_significance(episode_id, score)
        logger.debug(
            "episode_significance_updated",
            episode_id=str(episode_id),
            new_score=score,
        )
        return episode

    async def delete_episode(self, episode_id: uuid.UUID) -> bool:
        """Delete an episode record.

        Args:
            episode_id: Episode UUID.

        Returns:
            True if deleted, else False.
        """
        deleted = await self._repo.delete(episode_id)
        if deleted:
            logger.info("episode_deleted", episode_id=str(episode_id))
        return deleted

    @staticmethod
    def to_memory(episode: Episode) -> Memory:
        """Convert Episode model to normalized Memory representation.

        Args:
            episode: Episode model instance.

        Returns:
            Normalized Memory instance.
        """
        return Memory(
            id=episode.id,
            type=MemoryType.EPISODIC,
            title=episode.goal,
            content=episode.outcome,
            tags=[],
            metadata={
                "task_id": str(episode.task_id) if episode.task_id else None,
                "context": episode.context,
                "steps": episode.steps,
                "outcome": episode.outcome,
            },
            confidence=episode.significance,
            created_at=episode.created_at,
        )
