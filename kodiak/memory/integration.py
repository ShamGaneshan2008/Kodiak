"""Bridge between orchestration loop and the Kodiak memory system."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from kodiak.db.models.task import Task
from kodiak.memory.experience import (
    EngineeringExperience,
    ExperienceExtractor,
    ExperienceSanitizer,
)
from kodiak.memory.models import MemoryType, SearchResult
from kodiak.memory.service import MemoryService
from kodiak.orchestration.execution.models import ExecutionResult

logger = structlog.get_logger(__name__)

_DEFAULT_RETRIEVAL_LIMIT = 5
_MIN_RELEVANCE_SCORE = 0.25


class MemoryIntegration:
    """Records execution experiences and retrieves relevant memories for planning."""

    def __init__(
        self,
        memory_service: MemoryService | None = None,
        *,
        retrieval_limit: int = _DEFAULT_RETRIEVAL_LIMIT,
        min_relevance_score: float = _MIN_RELEVANCE_SCORE,
    ) -> None:
        self._memory = memory_service or MemoryService()
        self._extractor = ExperienceExtractor()
        self._sanitizer = ExperienceSanitizer()
        self._retrieval_limit = max(1, retrieval_limit)
        self._min_relevance = max(0.0, min(1.0, min_relevance_score))
        self._logger = logger.bind(component="memory_integration")

    @property
    def memory_service(self) -> MemoryService:
        return self._memory

    async def record_execution(self, task: Task, execution_result: ExecutionResult) -> bool:
        """Extract, validate, and store a useful engineering experience."""
        experience = self._extractor.extract(task, execution_result)
        if experience is None or not self._extractor.should_store(experience):
            self._logger.info("memory_record_skipped", task_id=str(task.id))
            return False

        sanitized = await self._sanitize_experience(experience)
        if await self._contains_blocking_secrets(sanitized):
            self._logger.warning("memory_record_blocked_secrets", task_id=str(task.id))
            return False

        await self._store_episode(task, sanitized)
        await self._store_lessons(task, sanitized)
        self._logger.info(
            "memory_record_stored",
            task_id=str(task.id),
            outcome=sanitized.final_result,
            category=sanitized.failure_category,
        )
        return True

    async def retrieve_for_planning(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve bounded, relevant memories for task planning."""
        ctx = context or {}
        tags = self._planning_tags(ctx)
        query = " ".join(
            part
            for part in [
                goal,
                str(ctx.get("task_type", "")),
                " ".join(str(cap) for cap in ctx.get("required_capabilities", [])[:5]),
            ]
            if part
        )
        results = await self._memory.search(
            query=query,
            memory_type=MemoryType.EPISODIC,
            limit=limit or self._retrieval_limit * 2,
            tags=tags or None,
        )
        semantic_results = await self._memory.search(
            query=query,
            memory_type=MemoryType.SEMANTIC,
            limit=limit or self._retrieval_limit,
            tags=tags or None,
        )

        combined = results + semantic_results
        filtered = [item for item in combined if item.relevance_score >= self._min_relevance]
        filtered.sort(key=lambda item: item.relevance_score, reverse=True)
        bounded = filtered[: limit or self._retrieval_limit]
        self._logger.info(
            "memory_retrieved_for_planning",
            query_preview=query[:80],
            returned=len(bounded),
        )
        return bounded

    async def build_planning_context(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return planning context enriched with relevant memories."""
        enriched = dict(context or {})
        memories = await self.retrieve_for_planning(goal, enriched)
        if not memories:
            return enriched

        enriched["relevant_memories"] = [
            {
                "type": item.memory.type.value,
                "title": item.memory.title,
                "content": item.memory.content,
                "relevance_score": item.relevance_score,
                "tags": item.memory.tags,
            }
            for item in memories
        ]
        enriched["memory_context"] = await self._memory.context_builder.build_context(
            query=goal,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            tags=self._planning_tags(enriched) or None,
            token_budget=1500,
        )
        return enriched

    async def consolidate_if_needed(self, limit: int = 20) -> int:
        """Run pending consolidations and return processed count."""
        results = await self._memory.consolidate(limit=limit)
        return len(results)

    async def _sanitize_experience(
        self, experience: EngineeringExperience
    ) -> EngineeringExperience:
        return EngineeringExperience(
            task_id=experience.task_id,
            goal=await self._sanitizer.sanitize_text(experience.goal),
            task_type=experience.task_type,
            agent_used=await self._sanitizer.sanitize_text(experience.agent_used),
            tools_used=experience.tools_used,
            approach=await self._sanitizer.sanitize_text(experience.approach),
            outcome=experience.outcome,
            verification_status=experience.verification_status,
            failure_category=experience.failure_category,
            root_cause=await self._sanitizer.sanitize_text(experience.root_cause),
            repair_performed=await self._sanitizer.sanitize_text(experience.repair_performed),
            final_result=experience.final_result,
            repository_id=experience.repository_id,
            capabilities=experience.capabilities,
            tags=experience.tags,
            confidence=experience.confidence,
            duration_seconds=experience.duration_seconds,
            timestamp=experience.timestamp,
            metadata=await self._sanitizer.sanitize_mapping(experience.metadata),
        )

    async def _contains_blocking_secrets(self, experience: EngineeringExperience) -> bool:
        probe = " ".join(
            filter(
                None,
                [
                    experience.goal,
                    experience.approach,
                    experience.root_cause or "",
                    experience.repair_performed or "",
                ],
            )
        )
        return not await self._sanitizer._secrets.validate_secret(probe)

    async def _store_episode(self, task: Task, experience: EngineeringExperience) -> None:
        task_uuid = self._coerce_uuid(experience.task_id)
        steps = [
            step
            for step in [
                f"agent={experience.agent_used}" if experience.agent_used else None,
                f"tools={','.join(experience.tools_used)}" if experience.tools_used else None,
                f"approach={experience.approach[:200]}" if experience.approach else None,
            ]
            if step
        ]
        outcome_text = (
            f"{experience.final_result}"
            f"{f' ({experience.verification_status})' if experience.verification_status else ''}"
        )
        if experience.root_cause:
            outcome_text += f" — {experience.root_cause[:300]}"

        await self._memory.episodic.create_episode(
            goal=experience.goal,
            outcome=await self._sanitizer.sanitize_text(outcome_text),
            task_id=task_uuid,
            context={
                "task_type": experience.task_type,
                "agent_used": experience.agent_used,
                "tools_used": experience.tools_used,
                "failure_category": experience.failure_category,
                "repository_id": experience.repository_id,
                "capabilities": experience.capabilities,
                "tags": experience.tags,
            },
            steps=steps,
        )

    async def _store_lessons(self, task: Task, experience: EngineeringExperience) -> None:
        lesson = experience.lesson_text()
        if not lesson:
            return
        sanitized_lesson = await self._sanitizer.sanitize_text(lesson)
        if len(sanitized_lesson.strip()) < 8:
            return

        category = experience.failure_category or experience.task_type or "engineering"
        task_uuid = self._coerce_uuid(experience.task_id)
        await self._memory.semantic.store_fact(
            content=sanitized_lesson,
            category=str(category),
            source_task_id=task_uuid,
            confidence=experience.confidence,
            metadata={
                "outcome": experience.final_result,
                "verification_status": experience.verification_status,
                "tags": experience.tags,
            },
        )

    @staticmethod
    def _planning_tags(context: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for key in ("task_type", "failure_category"):
            value = context.get(key)
            if value:
                tags.append(str(value))
        caps = context.get("required_capabilities") or context.get("capabilities") or []
        if isinstance(caps, list):
            tags.extend(str(cap) for cap in caps[:5])
        return list(dict.fromkeys(tags))

    @staticmethod
    def _coerce_uuid(value: str | None) -> uuid.UUID | None:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except ValueError:
            return None


__all__ = ["MemoryIntegration"]
