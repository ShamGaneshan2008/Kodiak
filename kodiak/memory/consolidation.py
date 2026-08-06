# kodiak/memory/consolidation.py
"""Memory Consolidation background process for transferring working memory to long-term stores."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from .procedural import ProcedureStep

logger = structlog.get_logger(__name__)

__all__ = [
    "ConsolidationStatus",
    "ConsolidationResult",
    "WorkingMemoryReader",
    "EpisodicMemoryWriter",
    "SemanticMemoryWriter",
    "ProceduralMemoryWriter",
    "MemoryConsolidator",
]


class ConsolidationStatus(StrEnum):
    """Lifecycle status for a memory consolidation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsolidationResult(BaseModel):
    """Result summary of consolidating a completed working memory task."""

    task_id: uuid.UUID
    status: ConsolidationStatus
    episodes_created: int = 0
    facts_stored: int = 0
    procedures_created: int = 0
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class WorkingMemoryReader(Protocol):
    """Protocol for reading unconsolidated task working memories."""

    async def get_unconsolidated_tasks(self, limit: int) -> list[dict[str, Any]]: ...


@runtime_checkable
class EpisodicMemoryWriter(Protocol):
    """Protocol for writing extracted episodes to long-term memory."""

    async def create_episode(
        self,
        goal: str,
        outcome: str,
        task_id: uuid.UUID | None = None,
        context: dict[str, Any] | None = None,
        steps: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Any: ...


@runtime_checkable
class SemanticMemoryWriter(Protocol):
    """Protocol for writing extracted facts to long-term memory."""

    async def store_fact(
        self,
        content: str,
        category: str = "general",
        source_task_id: uuid.UUID | None = None,
        confidence: float = 1.0,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...


@runtime_checkable
class ProceduralMemoryWriter(Protocol):
    """Protocol for writing extracted procedures to long-term memory."""

    async def create_procedure(
        self,
        name: str,
        description: str,
        steps: list[ProcedureStep],
        tags: list[str] | None = None,
    ) -> Any: ...


class MemoryConsolidator:
    """Consolidates working memory entries into episodic, semantic, and procedural memories."""

    def __init__(
        self,
        working_memory: WorkingMemoryReader,
        episodic_memory: EpisodicMemoryWriter,
        semantic_memory: SemanticMemoryWriter,
        procedural_memory: ProceduralMemoryWriter,
    ) -> None:
        """Initialize MemoryConsolidator.

        Args:
            working_memory: WorkingMemoryReader component.
            episodic_memory: EpisodicMemoryWriter component.
            semantic_memory: SemanticMemoryWriter component.
            procedural_memory: ProceduralMemoryWriter component.
        """
        self._working = working_memory
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory

    async def run_pending_consolidations(self, limit: int = 50) -> list[ConsolidationResult]:
        """Fetch and consolidate all pending completed or abandoned task working memories.

        Args:
            limit: Maximum task records to consolidate in batch.

        Returns:
            List of ConsolidationResult summaries.
        """
        tasks = await self._working.get_unconsolidated_tasks(limit)
        if not tasks:
            return []

        logger.info("starting_consolidation_batch", task_count=len(tasks))
        results: list[ConsolidationResult] = []

        for task in tasks:
            task_id_raw = task.get("id") or task.get("task_id")
            if not task_id_raw:
                continue
            task_id = uuid.UUID(str(task_id_raw)) if not isinstance(task_id_raw, uuid.UUID) else task_id_raw
            result = await self.consolidate_task(task_id, task)
            results.append(result)

        completed = sum(1 for r in results if r.status == ConsolidationStatus.COMPLETED)
        logger.info(
            "consolidation_batch_completed",
            processed=len(results),
            completed=completed,
            failed=len(results) - completed,
        )
        return results

    async def consolidate_task(
        self, task_id: uuid.UUID, task_data: dict[str, Any]
    ) -> ConsolidationResult:
        """Consolidate single task memory.

        Args:
            task_id: Task UUID.
            task_data: Dictionary of task memory properties.

        Returns:
            ConsolidationResult details.
        """
        result = ConsolidationResult(task_id=task_id, status=ConsolidationStatus.PROCESSING)
        try:
            result.episodes_created = await self._extract_episodic(task_id, task_data)
            result.facts_stored = await self._extract_semantic(task_id, task_data)
            result.procedures_created = await self._extract_procedural(task_id, task_data)
            result.status = ConsolidationStatus.COMPLETED
        except Exception as e:
            logger.exception("task_consolidation_failed", task_id=str(task_id))
            result.status = ConsolidationStatus.FAILED
            result.error = str(e)
        return result

    async def _extract_episodic(self, task_id: uuid.UUID, task_data: dict[str, Any]) -> int:
        outcome = task_data.get("outcome") or task_data.get("status")
        if not outcome:
            return 0

        goal = str(task_data.get("goal", ""))
        context = dict(task_data.get("context") or {})
        steps_raw = task_data.get("steps") or task_data.get("steps_taken") or []
        steps = [str(s) for s in steps_raw]

        await self._episodic.create_episode(
            goal=goal,
            outcome=str(outcome),
            task_id=task_id,
            context=context,
            steps=steps,
        )
        logger.debug("episodic_memory_extracted", task_id=str(task_id))
        return 1

    async def _extract_semantic(self, task_id: uuid.UUID, task_data: dict[str, Any]) -> int:
        scratchpad = task_data.get("scratchpad") if isinstance(task_data.get("scratchpad"), dict) else {}
        learnings = (
            task_data.get("learnings")
            or task_data.get("facts")
            or scratchpad.get("learnings")
            or scratchpad.get("facts")
        )
        if not learnings or not isinstance(learnings, list):
            return 0

        category = str(task_data.get("domain") or task_data.get("category") or "general")
        count = 0
        for fact in learnings:
            if not isinstance(fact, str):
                continue
            await self._semantic.store_fact(
                content=fact,
                category=category,
                source_task_id=task_id,
            )
            count += 1

        if count > 0:
            logger.debug(
                "semantic_memory_extracted",
                task_id=str(task_id),
                count=count,
            )
        return count

    async def _extract_procedural(self, task_id: uuid.UUID, task_data: dict[str, Any]) -> int:
        outcome = str(task_data.get("outcome", "")).lower()
        if outcome not in ("success", "completed") and task_data.get("status") != "completed":
            return 0

        steps_raw = task_data.get("steps_taken") or task_data.get("actions") or task_data.get("steps")
        if not steps_raw or not isinstance(steps_raw, list):
            return 0

        steps_list: list[ProcedureStep] = []
        for i, step in enumerate(steps_raw):
            if isinstance(step, ProcedureStep):
                steps_list.append(step)
            elif isinstance(step, dict):
                steps_list.append(
                    ProcedureStep(
                        step_number=step.get("step_number", i + 1),
                        action=str(step.get("action", "")),
                        tool_name=step.get("tool_name"),
                        parameters=dict(step.get("parameters") or {}),
                        expected_outcome=step.get("expected_outcome"),
                    )
                )
            else:
                steps_list.append(
                    ProcedureStep(
                        step_number=i + 1,
                        action=str(step),
                    )
                )

        name = str(task_data.get("goal", "Extracted Procedure"))[:80]
        description = f"Auto-extracted procedure from task {task_id}"
        tags = [str(t) for t in task_data.get("tags", [])]

        await self._procedural.create_procedure(
            name=name,
            description=description,
            steps=steps_list,
            tags=tags,
        )
        logger.debug("procedural_memory_extracted", task_id=str(task_id))
        return 1
