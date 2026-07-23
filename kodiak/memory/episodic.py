from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class EpisodeNotFoundError(Exception):
    pass


class Episode(BaseModel):
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
    episode: Episode
    relevance_score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class EpisodeRepository(Protocol):
    async def create(self, episode: Episode) -> Episode: ...

    async def get_by_id(self, episode_id: uuid.UUID) -> Episode | None: ...

    async def search(self, query: str, limit: int = 10) -> list[EpisodeSearchResult]: ...

    async def get_recent(self, limit: int = 10, offset: int = 0) -> list[Episode]: ...

    async def update_significance(self, episode_id: uuid.UUID, score: float) -> Episode: ...


class EpisodicMemory:
    def __init__(self, repository: EpisodeRepository) -> None:
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
        episode = await self._repo.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")
        return episode

    async def search_episodes(self, query: str, limit: int = 10) -> list[EpisodeSearchResult]:
        return await self._repo.search(query, limit)

    async def get_recent_episodes(self, limit: int = 10, offset: int = 0) -> list[Episode]:
        return await self._repo.get_recent(limit=limit, offset=offset)

    async def update_significance(self, episode_id: uuid.UUID, score: float) -> Episode:
        if not 0.0 <= score <= 1.0:
            raise ValueError("Significance score must be between 0.0 and 1.0")
        episode = await self._repo.update_significance(episode_id, score)
        logger.debug(
            "episode_significance_updated",
            episode_id=str(episode_id),
            new_score=score,
        )
        return episode
