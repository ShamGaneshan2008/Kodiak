# kodiak/memory/persistence.py
"""Persistence backends and in-memory repositories for Kodiak memory stores."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from .episodic import Episode, EpisodeSearchResult
from .errors import MemoryPersistenceError
from .procedural import Procedure, ProcedureSearchResult
from .semantic import SemanticEntity, SemanticSearchResult
from .working import WorkingMemoryItem, WorkingMemoryStatus

logger = structlog.get_logger(__name__)

__all__ = [
    "InMemoryWorkingMemoryRepository",
    "InMemoryShortTermMemoryRepository",
    "InMemoryEpisodeRepository",
    "InMemorySemanticRepository",
    "InMemoryProcedureRepository",
    "JSONFileMemoryPersistence",
]


class InMemoryWorkingMemoryRepository:
    """In-memory repository implementation for working memory items."""

    def __init__(self) -> None:
        self._items: dict[uuid.UUID, WorkingMemoryItem] = {}

    async def create(self, item: WorkingMemoryItem) -> WorkingMemoryItem:
        """Create and store a working memory item."""
        self._items[item.task_id] = item
        return item

    async def get_by_task_id(self, task_id: uuid.UUID) -> WorkingMemoryItem | None:
        """Get working memory item by task ID."""
        return self._items.get(task_id)

    async def get_by_id(self, memory_id: uuid.UUID) -> WorkingMemoryItem | None:
        """Get working memory item by memory ID."""
        for item in self._items.values():
            if item.id == memory_id:
                return item
        return None

    async def get_active(self) -> list[WorkingMemoryItem]:
        """Get all currently active working memory items."""
        return [item for item in self._items.values() if item.status == WorkingMemoryStatus.ACTIVE]

    async def update(self, item: WorkingMemoryItem) -> WorkingMemoryItem:
        """Update an existing working memory item."""
        self._items[item.task_id] = item
        return item

    async def delete(self, task_id: uuid.UUID) -> bool:
        """Delete working memory item by task ID."""
        return self._items.pop(task_id, None) is not None

    async def list_all(self, limit: int = 100) -> list[WorkingMemoryItem]:
        """List stored working memory items."""
        return list(self._items.values())[:limit]

    async def get_unconsolidated_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch tasks that have completed or been abandoned but not yet consolidated."""
        unconsolidated: list[dict[str, Any]] = []
        for item in self._items.values():
            if item.status in (WorkingMemoryStatus.COMPLETED, WorkingMemoryStatus.ABANDONED):
                unconsolidated.append(
                    {
                        "id": str(item.task_id),
                        "goal": item.goal,
                        "context": item.context,
                        "scratchpad": item.scratchpad,
                        "status": item.status.value,
                        "outcome": item.outcome,
                        "completed_at": item.updated_at.isoformat(),
                    }
                )
        return unconsolidated[:limit]


class InMemoryShortTermMemoryRepository:
    """In-memory repository implementation for short-term memory sessions."""

    def __init__(self, max_items_per_session: int = 200) -> None:
        self._max_items_per_session = max_items_per_session
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    async def add(self, session_id: str, item_data: dict[str, Any]) -> dict[str, Any]:
        """Add a short-term memory item to a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        history = self._sessions[session_id]
        history.append(item_data)

        if len(history) > self._max_items_per_session:
            self._sessions[session_id] = history[-self._max_items_per_session :]

        return item_data

    async def get_session(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent history items for a session."""
        history = self._sessions.get(session_id, [])
        return history[-limit:]

    async def clear_session(self, session_id: str) -> bool:
        """Clear all items for a given session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def list_sessions(self) -> list[str]:
        """List all active session identifiers."""
        return list(self._sessions.keys())


class InMemoryEpisodeRepository:
    """In-memory repository implementation for episodic memory records."""

    def __init__(self) -> None:
        self._episodes: dict[uuid.UUID, Episode] = {}

    async def create(self, episode: Episode) -> Episode:
        """Store a new episode."""
        self._episodes[episode.id] = episode
        return episode

    async def get_by_id(self, episode_id: uuid.UUID) -> Episode | None:
        """Retrieve an episode by ID."""
        return self._episodes.get(episode_id)

    async def search(self, query: str, limit: int = 10) -> list[EpisodeSearchResult]:
        """Search episodes by text query relevance."""
        results: list[EpisodeSearchResult] = []
        terms = [t.lower() for t in query.split() if t]

        for episode in self._episodes.values():
            text = f"{episode.goal} {episode.outcome} {' '.join(episode.steps)}".lower()
            if not terms:
                score = 1.0
            else:
                matches = sum(1 for term in terms if term in text)
                score = matches / len(terms)

            if score > 0:
                results.append(EpisodeSearchResult(episode=episode, relevance_score=score))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    async def get_recent(self, limit: int = 10, offset: int = 0) -> list[Episode]:
        """Get recent episodes sorted by creation date."""
        episodes = sorted(self._episodes.values(), key=lambda e: e.created_at, reverse=True)
        return episodes[offset : offset + limit]

    async def update_significance(self, episode_id: uuid.UUID, score: float) -> Episode:
        """Update significance score for an episode."""
        episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"Episode {episode_id} not found")

        updated = episode.model_copy(update={"significance": score})
        self._episodes[episode_id] = updated
        return updated

    async def delete(self, episode_id: uuid.UUID) -> bool:
        """Delete an episode by ID."""
        return self._episodes.pop(episode_id, None) is not None


class InMemorySemanticRepository:
    """In-memory repository implementation for semantic memory entities."""

    def __init__(self) -> None:
        self._entities: dict[uuid.UUID, SemanticEntity] = {}

    async def create(self, entity: SemanticEntity) -> SemanticEntity:
        """Store a semantic entity."""
        self._entities[entity.id] = entity
        return entity

    async def get_by_id(self, entity_id: uuid.UUID) -> SemanticEntity | None:
        """Retrieve a semantic entity by ID."""
        return self._entities.get(entity_id)

    async def search(
        self, query: str, category: str | None = None, limit: int = 10
    ) -> list[SemanticSearchResult]:
        """Search semantic entities by text query and category."""
        results: list[SemanticSearchResult] = []
        terms = [t.lower() for t in query.split() if t]

        for entity in self._entities.values():
            if category and entity.category != category:
                continue

            text = f"{entity.content} {entity.category}".lower()
            if not terms:
                score = 1.0
            else:
                matches = sum(1 for term in terms if term in text)
                score = matches / len(terms)

            if score > 0:
                results.append(SemanticSearchResult(entity=entity, relevance_score=score))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    async def update(self, entity: SemanticEntity) -> SemanticEntity:
        """Update an existing semantic entity."""
        self._entities[entity.id] = entity
        return entity

    async def delete(self, entity_id: uuid.UUID) -> bool:
        """Delete a semantic entity by ID."""
        return self._entities.pop(entity_id, None) is not None

    async def list_facts(
        self, category: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[SemanticEntity]:
        """List stored facts optionally filtered by category."""
        entities = list(self._entities.values())
        if category:
            entities = [e for e in entities if e.category == category]
        entities.sort(key=lambda e: e.created_at, reverse=True)
        return entities[offset : offset + limit]


class InMemoryProcedureRepository:
    """In-memory repository implementation for procedural memory records."""

    def __init__(self) -> None:
        self._procedures: dict[uuid.UUID, Procedure] = {}

    async def create(self, procedure: Procedure) -> Procedure:
        """Store a procedural workflow."""
        self._procedures[procedure.id] = procedure
        return procedure

    async def get_by_id(self, procedure_id: uuid.UUID) -> Procedure | None:
        """Retrieve a procedure by ID."""
        return self._procedures.get(procedure_id)

    async def update(self, procedure: Procedure) -> Procedure:
        """Update an existing procedure."""
        self._procedures[procedure.id] = procedure
        return procedure

    async def search(self, query: str, limit: int = 10) -> list[ProcedureSearchResult]:
        """Search procedures by name, description, tags, and step actions."""
        results: list[ProcedureSearchResult] = []
        terms = [t.lower() for t in query.split() if t]

        for procedure in self._procedures.values():
            text = (
                f"{procedure.name} {procedure.description}"
                f" {' '.join(procedure.tags)}"
                f" {' '.join(s.action for s in procedure.steps)}"
            ).lower()
            if not terms:
                score = 1.0
            else:
                matches = sum(1 for term in terms if term in text)
                score = matches / len(terms)

            if score > 0:
                results.append(ProcedureSearchResult(procedure=procedure, relevance_score=score))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    async def increment_success(self, procedure_id: uuid.UUID) -> Procedure:
        """Increment success count for a procedure."""
        procedure = self._procedures.get(procedure_id)
        if procedure is None:
            raise KeyError(f"Procedure {procedure_id} not found")

        updated = procedure.model_copy(
            update={
                "success_count": procedure.success_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        self._procedures[procedure_id] = updated
        return updated

    async def increment_failure(self, procedure_id: uuid.UUID) -> Procedure:
        """Increment failure count for a procedure."""
        procedure = self._procedures.get(procedure_id)
        if procedure is None:
            raise KeyError(f"Procedure {procedure_id} not found")

        updated = procedure.model_copy(
            update={
                "failure_count": procedure.failure_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        self._procedures[procedure_id] = updated
        return updated

    async def delete(self, procedure_id: uuid.UUID) -> bool:
        """Delete a procedure by ID."""
        return self._procedures.pop(procedure_id, None) is not None

    async def list_procedures(self, limit: int = 100, offset: int = 0) -> list[Procedure]:
        """List stored procedures."""
        procedures = sorted(self._procedures.values(), key=lambda p: p.created_at, reverse=True)
        return procedures[offset : offset + limit]


class JSONFileMemoryPersistence:
    """Asynchronous JSON file persistence driver for exporting and restoring memory states."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    async def save(
        self,
        working_items: list[WorkingMemoryItem],
        episodes: list[Episode],
        semantic_entities: list[SemanticEntity],
        procedures: list[Procedure],
        short_term_data: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        """Save memory states to disk as JSON."""
        try:
            payload = {
                "working_items": [item.model_dump(mode="json") for item in working_items],
                "episodes": [ep.model_dump(mode="json") for ep in episodes],
                "semantic_entities": [se.model_dump(mode="json") for se in semantic_entities],
                "procedures": [p.model_dump(mode="json") for p in procedures],
                "short_term_data": short_term_data or {},
                "saved_at": datetime.now(UTC).isoformat(),
            }
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("memory_persistence_saved", file_path=str(self.file_path))
        except Exception as exc:
            logger.exception("memory_persistence_save_failed", file_path=str(self.file_path))
            raise MemoryPersistenceError(f"Failed to save memories to {self.file_path}") from exc

    async def load(self) -> dict[str, Any]:
        """Load memory states from disk."""
        if not self.file_path.exists():
            return {
                "working_items": [],
                "episodes": [],
                "semantic_entities": [],
                "procedures": [],
                "short_term_data": {},
            }
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            logger.info("memory_persistence_loaded", file_path=str(self.file_path))
            return {
                "working_items": [
                    WorkingMemoryItem.model_validate(item) for item in data.get("working_items", [])
                ],
                "episodes": [Episode.model_validate(ep) for ep in data.get("episodes", [])],
                "semantic_entities": [
                    SemanticEntity.model_validate(se) for se in data.get("semantic_entities", [])
                ],
                "procedures": [Procedure.model_validate(p) for p in data.get("procedures", [])],
                "short_term_data": data.get("short_term_data", {}),
            }
        except Exception as exc:
            logger.exception("memory_persistence_load_failed", file_path=str(self.file_path))
            raise MemoryPersistenceError(f"Failed to load memories from {self.file_path}") from exc
