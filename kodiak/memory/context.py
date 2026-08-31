# kodiak/memory/context.py
"""Context Builder for assembling memory content into token-budgeted LLM prompt sections."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import structlog

from .models import MemoryType, SearchResult
from .retrieval import MemoryRetriever
from .short_term import ShortTermMemoryItem
from .working import WorkingMemoryItem

logger = structlog.get_logger(__name__)

__all__ = ["MemoryContextBuilder"]


class MemoryContextBuilder:
    """Assembles structured prompt context strings from memory records within token budgets."""

    def __init__(
        self,
        retriever: MemoryRetriever | None = None,
        default_token_budget: int = 4000,
        chars_per_token: float = 4.0,
    ) -> None:
        """Initialize MemoryContextBuilder.

        Args:
            retriever: MemoryRetriever instance.
            default_token_budget: Maximum token budget for context text assembly.
            chars_per_token: Average characters per token ratio (default 4.0).
        """
        self.retriever = retriever or MemoryRetriever()
        self.default_token_budget = default_token_budget
        self.chars_per_token = chars_per_token

    async def build_context(
        self,
        query: str,
        session_id: str | None = None,
        task_id: uuid.UUID | None = None,
        working_memory_item: WorkingMemoryItem | None = None,
        short_term_items: Sequence[ShortTermMemoryItem] | None = None,
        memory_types: Sequence[MemoryType] | None = None,
        tags: Sequence[str] | None = None,
        token_budget: int | None = None,
    ) -> str:
        """Build structured markdown context string within token budget constraint.

        Args:
            query: Query text for retrieving relevant long-term memories.
            session_id: Session ID string.
            task_id: Task UUID.
            working_memory_item: Optional active working memory item.
            short_term_items: Optional explicit short-term history.
            memory_types: Target memory types.
            tags: Tag filter.
            token_budget: Custom token budget limit.

        Returns:
            Formatted Markdown context string.
        """
        budget = token_budget if token_budget is not None else self.default_token_budget
        max_chars = int(budget * self.chars_per_token)

        sections: list[str] = []
        current_chars = 0

        # Section 1: Active Working Memory
        if working_memory_item:
            wm_text = (
                f"### Active Task Working Memory\n"
                f"- **Task ID**: {working_memory_item.task_id}\n"
                f"- **Goal**: {working_memory_item.goal}\n"
                f"- **Status**: {working_memory_item.status.value}\n"
            )
            if working_memory_item.scratchpad:
                wm_text += f"- **Scratchpad**: {working_memory_item.scratchpad}\n"
            if working_memory_item.outcome:
                wm_text += f"- **Outcome**: {working_memory_item.outcome}\n"

            sections.append(wm_text)
            current_chars += len(wm_text)

        # Section 2: Short-Term Interaction History
        st_history: list[ShortTermMemoryItem] = list(short_term_items or [])
        if not st_history and session_id:
            try:
                st_history = await self.retriever.short_term.get_session_history(
                    session_id, limit=20
                )
            except Exception:
                pass

        if st_history:
            st_text_lines = ["### Recent Interaction History"]
            for item in st_history[-10:]:
                st_text_lines.append(f"- **{item.role.upper()}**: {item.content}")
            st_text = "\n".join(st_text_lines) + "\n"

            if current_chars + len(st_text) <= max_chars:
                sections.append(st_text)
                current_chars += len(st_text)

        # Section 3: Retrieved Long-Term Knowledge
        retrieved_results: list[SearchResult] = await self.retriever.retrieve(
            query=query,
            session_id=session_id,
            task_id=task_id,
            memory_types=memory_types,
            tags=tags,
            limit=15,
        )

        if retrieved_results:
            lt_lines = ["### Relevant Memories & Knowledge"]
            for res in retrieved_results:
                mem = res.memory
                type_label = mem.type.value.upper()
                entry_text = (
                    f"- [{type_label}] **{mem.title}** "
                    f"(relevance: {res.relevance_score:.2f}): {mem.content}"
                )
                if current_chars + len(entry_text) + 2 > max_chars:
                    break
                lt_lines.append(entry_text)
                current_chars += len(entry_text) + 1

            if len(lt_lines) > 1:
                sections.append("\n".join(lt_lines))

        if not sections:
            return ""

        context_output = "## Memory Context\n\n" + "\n\n".join(sections)
        logger.debug(
            "memory_context_built",
            token_budget=budget,
            char_count=len(context_output),
        )
        return context_output
