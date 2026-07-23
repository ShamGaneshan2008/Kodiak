# kodiak/memory/models.py
"""Unified Pydantic models exposed by the Kodiak memory service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["MemoryType", "Memory", "SearchResult"]


class MemoryType(StrEnum):
    """Which underlying memory store a `Memory` record originated from."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory(BaseModel):
    """Normalized view over episodic, semantic, and procedural memory records.

    Flattens the type-specific fields of `Episode`, `SemanticEntity`, and
    `Procedure` into a single shape so callers can add, list, search, and
    delete memories without depending on each store's internal representation.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: MemoryType
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None


class SearchResult(BaseModel):
    """A single ranked match returned by `MemoryService.search`."""

    memory: Memory
    relevance_score: float = Field(ge=0.0, le=1.0)
