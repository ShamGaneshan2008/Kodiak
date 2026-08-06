# kodiak/memory/errors.py
"""Custom exceptions for the Kodiak memory subsystem."""

from __future__ import annotations

__all__ = [
    "MemoryServiceError",
    "MemoryNotFoundError",
    "WorkingMemoryNotFoundError",
    "EpisodeNotFoundError",
    "FactNotFoundError",
    "ProcedureNotFoundError",
    "ShortTermMemoryError",
    "MemoryPersistenceError",
]


class MemoryServiceError(Exception):
    """Base exception for all memory service failures."""


class MemoryNotFoundError(MemoryServiceError):
    """Raised when a requested memory record cannot be located."""

    def __init__(self, memory_id: str, memory_type: str | None = None) -> None:
        self.memory_id = memory_id
        self.memory_type = memory_type
        detail = f" of type '{memory_type}'" if memory_type else ""
        super().__init__(f"Memory{detail} with id '{memory_id}' not found")


class WorkingMemoryNotFoundError(MemoryNotFoundError):
    """Raised when working memory for a task cannot be located."""

    def __init__(self, task_id: str) -> None:
        super().__init__(memory_id=task_id, memory_type="working")


class EpisodeNotFoundError(MemoryNotFoundError):
    """Raised when an episode record cannot be located."""

    def __init__(self, episode_id: str) -> None:
        super().__init__(memory_id=episode_id, memory_type="episodic")


class FactNotFoundError(MemoryNotFoundError):
    """Raised when a semantic fact record cannot be located."""

    def __init__(self, fact_id: str) -> None:
        super().__init__(memory_id=fact_id, memory_type="semantic")


class ProcedureNotFoundError(MemoryNotFoundError):
    """Raised when a procedural memory record cannot be located."""

    def __init__(self, procedure_id: str) -> None:
        super().__init__(memory_id=procedure_id, memory_type="procedural")


class ShortTermMemoryError(MemoryServiceError):
    """Raised when short-term memory operations fail."""


class MemoryPersistenceError(MemoryServiceError):
    """Raised when memory persistence operations fail."""
