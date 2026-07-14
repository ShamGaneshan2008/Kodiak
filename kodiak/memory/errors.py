# kodiak/memory/errors.py
"""Custom exceptions for the Kodiak memory subsystem."""

from __future__ import annotations

__all__ = ["MemoryServiceError", "MemoryNotFoundError"]


class MemoryServiceError(Exception):
    """Base exception for all memory service failures."""


class MemoryNotFoundError(MemoryServiceError):
    """Raised when a requested memory record cannot be located."""

    def __init__(self, memory_id: str, memory_type: str | None = None) -> None:
        self.memory_id = memory_id
        self.memory_type = memory_type
        detail = f" of type '{memory_type}'" if memory_type else ""
        super().__init__(f"Memory{detail} with id '{memory_id}' not found")