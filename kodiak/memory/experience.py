"""Engineering experience extraction and recording for the memory system."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from kodiak.db.models.task import Task
from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.security.secrets import SecretManager

logger = structlog.get_logger(__name__)

_MIN_CONTENT_LENGTH = 8


class EngineeringExperience(BaseModel):
    """Structured engineering experience derived from task execution."""

    task_id: str
    goal: str
    task_type: str = "general"
    agent_used: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    approach: str = ""
    outcome: str
    verification_status: str | None = None
    failure_category: str | None = None
    root_cause: str | None = None
    repair_performed: str | None = None
    final_result: str
    repository_id: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def lesson_text(self) -> str | None:
        """Return a durable lesson string when one exists."""
        if self.repair_performed:
            return self.repair_performed
        if self.root_cause and self.failure_category:
            return f"{self.failure_category}: {self.root_cause}"
        if self.outcome == "success" and self.approach:
            return f"Successful approach: {self.approach}"
        return None


class ExperienceSanitizer:
    """Redacts secrets from experience payloads before persistence."""

    def __init__(self, secret_manager: SecretManager | None = None) -> None:
        self._secrets = secret_manager or SecretManager()

    async def sanitize_text(self, value: str | None) -> str:
        if not value:
            return ""
        return await self._secrets.mask_secrets(value)

    async def sanitize_mapping(self, data: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(data, default=str)
        masked = await self.sanitize_text(raw)
        try:
            parsed = json.loads(masked)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"redacted_summary": masked[:2000]}


class ExperienceExtractor:
    """Builds structured experiences from execution artifacts."""

    def extract(
        self, task: Task, execution_result: ExecutionResult
    ) -> EngineeringExperience | None:
        reflection = execution_result.reflection or task.context.get("reflection") or {}
        verification = execution_result.verification or {}
        correction = task.context.get("correction_context") or {}

        goal = task.title or task.description or "Untitled task"
        capabilities = self._capabilities(task)
        agent_used = self._agent_used(execution_result, task)
        tools_used = self._tools_used(task, execution_result)

        outcome_label = execution_result.outcome.value
        verification_status = verification.get("status")
        failure_category = reflection.get("category") or correction.get("category")
        root_cause = reflection.get("root_cause") or correction.get("root_cause")
        repair = reflection.get("suggested_correction") or correction.get("suggested_correction")

        approach = repair or reflection.get("summary") or task.description or ""
        final_result = "success" if execution_result.is_success else "failure"

        tags = list(
            dict.fromkeys(
                [
                    outcome_label,
                    *([verification_status] if verification_status else []),
                    *([failure_category] if failure_category else []),
                    *capabilities[:5],
                ]
            )
        )

        confidence = float(reflection.get("confidence", 0.7)) if reflection else 0.8
        if execution_result.is_success and verification_status == "verified":
            confidence = max(confidence, 0.85)

        return EngineeringExperience(
            task_id=str(task.id),
            goal=goal,
            task_type=str(task.context.get("task_type", "general")),
            agent_used=agent_used,
            tools_used=tools_used,
            approach=approach[:1000],
            outcome=outcome_label,
            verification_status=verification_status,
            failure_category=failure_category,
            root_cause=root_cause,
            repair_performed=repair,
            final_result=final_result,
            repository_id=str(task.repository_id) if task.repository_id else None,
            capabilities=capabilities,
            tags=[tag for tag in tags if tag],
            confidence=max(0.0, min(1.0, confidence)),
            duration_seconds=execution_result.duration_seconds,
            metadata={
                "attempts": execution_result.attempts,
                "has_reflection": bool(reflection),
                "has_verification": bool(verification),
            },
        )

    @staticmethod
    def should_store(experience: EngineeringExperience) -> bool:
        """Return True when the experience is worth persisting."""
        if len(experience.goal.strip()) < _MIN_CONTENT_LENGTH:
            return False
        if experience.final_result == "success":
            return experience.verification_status in (None, "verified", "inconclusive")
        if experience.failure_category or experience.root_cause or experience.repair_performed:
            return True
        return experience.outcome in {
            ExecutionOutcome.FAILURE.value,
            ExecutionOutcome.TIMEOUT.value,
            ExecutionOutcome.RETRY_EXHAUSTED.value,
        }

    @staticmethod
    def _capabilities(task: Task) -> list[str]:
        caps = task.context.get("required_capabilities") or task.context.get("capabilities") or []
        if isinstance(caps, (list, tuple, set, frozenset)):
            return [str(cap) for cap in caps]
        return []

    @staticmethod
    def _agent_used(execution_result: ExecutionResult, task: Task) -> str | None:
        agent = execution_result.result.get("agent")
        if isinstance(agent, str):
            return agent
        reflection_agent = (task.context.get("reflection") or {}).get("agent_id")
        return str(reflection_agent) if reflection_agent else None

    @staticmethod
    def _tools_used(task: Task, execution_result: ExecutionResult) -> list[str]:
        tools: list[str] = []
        if isinstance(task.context.get("tools"), list):
            tools.extend(str(t) for t in task.context["tools"])
        listing = execution_result.result.get("tool_listing")
        if isinstance(listing, dict):
            tools.append("list_dir")
        return list(dict.fromkeys(tools))


__all__ = [
    "EngineeringExperience",
    "ExperienceExtractor",
    "ExperienceSanitizer",
]
