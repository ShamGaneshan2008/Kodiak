from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from kodiak.config.metrics import (
    ACTIVE_AGENT_TASKS,
    AGENT_TASK_DURATION_SECONDS,
    AGENT_TASKS_TOTAL,
)
from kodiak.config.settings import get_settings

logger = structlog.get_logger(__name__)


class AgentRole(StrEnum):
    PLANNER = "planner"
    REPOSITORY = "repository"
    RETRIEVAL = "retrieval"
    RESEARCH = "research"
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DEBUGGER = "debugger"
    REFLECTION = "reflection"
    GIT = "git"
    MEMORY = "memory"
    LEARNING = "learning"
    EVALUATION = "evaluation"


@dataclass
class AgentInput:
    task_id: str
    project_id: str
    instruction: str
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AgentOutput:
    run_id: str
    agent: AgentRole
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.token_usage.get("input_tokens", 0) + self.token_usage.get("output_tokens", 0)


class BaseAgent(ABC):
    """
    Base class for every Kodiak agent.

    Handles:
        • timing
        • logging
        • Prometheus metrics
        • exception handling

    Child classes only implement `_run()`.
    """

    role: AgentRole

    def __init__(self) -> None:
        self._settings = get_settings()
        self._log = structlog.get_logger(self.__class__.__name__)

    async def run(self, input_: AgentInput) -> AgentOutput:
        log = self._log.bind(
            agent=self.role,
            task_id=input_.task_id,
            run_id=input_.run_id,
        )

        log.info("agent.start")

        ACTIVE_AGENT_TASKS.labels(agent_type=self.role.value).inc()

        start = time.monotonic()

        try:
            output = await self._run(input_)

            output.duration_seconds = time.monotonic() - start

            status = "success" if output.success else "failure"

            AGENT_TASKS_TOTAL.labels(
                agent_type=self.role,
                status=status,
            ).inc()

            log.info(
                "agent.complete",
                success=output.success,
                duration=output.duration_seconds,
                tokens=output.total_tokens,
            )

            return output

        except Exception as exc:
            duration = time.monotonic() - start

            AGENT_TASKS_TOTAL.labels(
                agent_type=self.role,
                status="error",
            ).inc()

            log.exception(
                "agent.error",
                error=str(exc),
            )

            return AgentOutput(
                run_id=input_.run_id,
                agent=self.role,
                success=False,
                error=str(exc),
                duration_seconds=duration,
            )

        finally:
            ACTIVE_AGENT_TASKS.labels(agent_type=self.role.value).dec()

            AGENT_TASK_DURATION_SECONDS.labels(
                agent_type=self.role.value,
            ).observe(time.monotonic() - start)

    @abstractmethod
    async def _run(self, input_: AgentInput) -> AgentOutput:
        """
        Implemented by every agent.
        """

    def _make_output(
        self,
        input_: AgentInput,
        result: dict[str, Any],
        token_usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentOutput:
        return AgentOutput(
            run_id=input_.run_id,
            agent=self.role,
            success=True,
            result=result,
            token_usage=token_usage or {},
            metadata=metadata or {},
        )

    def _make_error(
        self,
        input_: AgentInput,
        error: str,
    ) -> AgentOutput:
        return AgentOutput(
            run_id=input_.run_id,
            agent=self.role,
            success=False,
            error=error,
        )
