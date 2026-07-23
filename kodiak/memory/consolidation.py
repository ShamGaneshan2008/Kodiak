from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ConsolidationStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsolidationResult(BaseModel):
    task_id: uuid.UUID
    status: ConsolidationStatus
    episodes_created: int = 0
    facts_stored: int = 0
    procedures_created: int = 0
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class WorkingMemoryReader(Protocol):
    async def get_unconsolidated_tasks(self, limit: int) -> list[dict[str, Any]]: ...


@runtime_checkable
class EpisodicMemoryWriter(Protocol):
    async def create_episode(self, data: dict[str, Any]) -> uuid.UUID: ...


@runtime_checkable
class SemanticMemoryWriter(Protocol):
    async def store_fact(self, data: dict[str, Any]) -> uuid.UUID: ...


@runtime_checkable
class ProceduralMemoryWriter(Protocol):
    async def create_procedure(self, data: dict[str, Any]) -> uuid.UUID: ...


class MemoryConsolidator:
    def __init__(
        self,
        working_memory: WorkingMemoryReader,
        episodic_memory: EpisodicMemoryWriter,
        semantic_memory: SemanticMemoryWriter,
        procedural_memory: ProceduralMemoryWriter,
    ) -> None:
        self._working = working_memory
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory

    async def run_pending_consolidations(self, limit: int = 50) -> list[ConsolidationResult]:
        tasks = await self._working.get_unconsolidated_tasks(limit)
        if not tasks:
            return []

        logger.info("starting_consolidation_batch", task_count=len(tasks))
        results: list[ConsolidationResult] = []

        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            result = await self.consolidate_task(uuid.UUID(task_id), task)
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
        result = ConsolidationResult(task_id=task_id, status=ConsolidationStatus.PROCESSING)
        try:
            result.episodes_created = await self._extract_episodic(task_data)
            result.facts_stored = await self._extract_semantic(task_data)
            result.procedures_created = await self._extract_procedural(task_data)
            result.status = ConsolidationStatus.COMPLETED
        except Exception as e:
            logger.exception("task_consolidation_failed", task_id=str(task_id))
            result.status = ConsolidationStatus.FAILED
            result.error = str(e)
        return result

    async def _extract_episodic(self, task_data: dict[str, Any]) -> int:
        outcome = task_data.get("outcome")
        if not outcome:
            return 0
        episode_data = {
            "task_id": task_data.get("id"),
            "goal": task_data.get("goal", ""),
            "context": task_data.get("context", {}),
            "outcome": outcome,
            "timestamp": task_data.get("completed_at"),
        }
        await self._episodic.create_episode(episode_data)
        logger.debug("episodic_memory_extracted", task_id=task_data.get("id"))
        return 1

    async def _extract_semantic(self, task_data: dict[str, Any]) -> int:
        learnings = task_data.get("learnings") or task_data.get("facts")
        if not learnings or not isinstance(learnings, list):
            return 0
        count = 0
        for fact in learnings:
            if not isinstance(fact, str):
                continue
            fact_data = {
                "content": fact,
                "source_task_id": task_data.get("id"),
                "domain": task_data.get("domain", "general"),
            }
            await self._semantic.store_fact(fact_data)
            count += 1
        if count > 0:
            logger.debug(
                "semantic_memory_extracted",
                task_id=task_data.get("id"),
                count=count,
            )
        return count

    async def _extract_procedural(self, task_data: dict[str, Any]) -> int:
        if task_data.get("outcome") != "success":
            return 0
        steps = task_data.get("steps_taken") or task_data.get("actions")
        if not steps or not isinstance(steps, list):
            return 0
        procedure_data = {
            "name": task_data.get("goal", "Unnamed Procedure"),
            "description": f"Auto-extracted from task {task_data.get('id')}",
            "steps": steps,
            "tags": task_data.get("tags", []),
        }
        await self._procedural.create_procedure(procedure_data)
        logger.debug("procedural_memory_extracted", task_id=task_data.get("id"))
        return 1
