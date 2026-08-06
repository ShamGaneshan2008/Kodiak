# kodiak/memory/retrieval.py
"""Concurrent retrieval engine querying across memory stores."""

from __future__ import annotations

import asyncio
import uuid
from typing import Sequence

import structlog

from .long_term import LongTermMemory
from .models import Memory, MemoryType, SearchResult
from .ranking import MemoryRanker
from .short_term import ShortTermMemory
from .working import WorkingMemory

logger = structlog.get_logger(__name__)

__all__ = ["MemoryRetriever"]


class MemoryRetriever:
    """Async-first retriever performing unified memory search and ranking."""

    def __init__(
        self,
        working: WorkingMemory | None = None,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        ranker: MemoryRanker | None = None,
    ) -> None:
        """Initialize MemoryRetriever.

        Args:
            working: WorkingMemory manager.
            short_term: ShortTermMemory manager.
            long_term: LongTermMemory manager.
            ranker: MemoryRanker engine.
        """
        self.working = working or WorkingMemory()
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()
        self.ranker = ranker or MemoryRanker()

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        task_id: uuid.UUID | None = None,
        memory_types: Sequence[MemoryType] | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Retrieve and rank relevant memories across requested memory stores concurrently.

        Args:
            query: Query text string.
            session_id: Optional session identifier for short-term memory search.
            task_id: Optional task UUID for working memory search.
            memory_types: Filter to specific MemoryTypes or all if None.
            tags: Filter by tag labels if provided.
            limit: Maximum items to return.

        Returns:
            List of SearchResult items ranked by relevance.
        """
        target_types = set(memory_types or list(MemoryType))
        candidate_memories: list[Memory] = []

        fetch_tasks: list[asyncio.Task[list[Memory]]] = []

        if MemoryType.WORKING in target_types and task_id:
            fetch_tasks.append(asyncio.create_task(self._fetch_working(task_id)))

        if MemoryType.SHORT_TERM in target_types and session_id:
            fetch_tasks.append(asyncio.create_task(self._fetch_short_term(session_id)))

        long_term_types = [
            t for t in target_types if t in (MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL)
        ]
        if long_term_types:
            for lt_type in long_term_types:
                fetch_tasks.append(
                    asyncio.create_task(
                        self._fetch_long_term(query, lt_type, tags=list(tags or []))
                    )
                )

        if fetch_tasks:
            results_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for res in results_list:
                if isinstance(res, list):
                    candidate_memories.extend(res)
                elif isinstance(res, Exception):
                    logger.warning("memory_retrieval_subtask_failed", error=str(res))

        return self.ranker.rank(query, candidate_memories, tags=tags, limit=limit)

    async def _fetch_working(self, task_id: uuid.UUID) -> list[Memory]:
        try:
            item = await self.working.get_working_memory(task_id)
            return [self.working.to_memory(item)]
        except Exception:
            return []

    async def _fetch_short_term(self, session_id: str) -> list[Memory]:
        try:
            items = await self.short_term.get_session_history(session_id, limit=50)
            return [self.short_term.to_memory(item) for item in items]
        except Exception:
            return []

    async def _fetch_long_term(
        self, query: str, memory_type: MemoryType, tags: list[str]
    ) -> list[Memory]:
        try:
            results = await self.long_term.search(query, memory_type=memory_type, tags=tags, limit=50)
            return [r.memory for r in results]
        except Exception:
            return []
