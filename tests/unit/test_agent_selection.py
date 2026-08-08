from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from kodiak.agents.manager import AgentManager, NoSuitableAgentError
from kodiak.agents.selector import (
    AgentCandidate,
    AgentHealthStatus,
    AgentSelector,
    SelectionContext,
)
from kodiak.agents.selector import (
    NoSuitableAgentError as SelectorNoSuitableAgentError,
)
from kodiak.db.models.task import Task, TaskPriority, TaskStatus
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome


@dataclass
class TaskLike:
    task_id: str = "task-1"
    task_type: str = "implementation"
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    priority: TaskPriority = TaskPriority.MEDIUM


class FakeAgent:
    def __init__(
        self,
        agent_id: str,
        capabilities: set[str],
        *,
        priority: int = 0,
        healthy: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.name = agent_id
        self.capabilities = frozenset(capabilities)
        self.priority = priority
        self.healthy = healthy
        self.executed: list[str] = []

    async def execute(self, task: TaskLike) -> dict[str, Any]:
        self.executed.append(task.task_id)
        return {"agent": self.agent_id}

    async def health_check(self) -> bool:
        return self.healthy


def test_selector_exact_capability_match() -> None:
    selector = AgentSelector()
    context = SelectionContext(required_capabilities=frozenset({"code_generation"}))

    result = selector.select(
        context,
        [
            AgentCandidate("coder", frozenset({"code_generation"})),
            AgentCandidate("tester", frozenset({"test_execution"})),
        ],
    )

    assert result.selected_agent_id == "coder"
    assert result.matched_capabilities == ("code_generation",)


def test_selector_records_partial_match_without_selecting_incompatible_agent() -> None:
    selector = AgentSelector()
    context = SelectionContext(required_capabilities=frozenset({"code_generation", "file_editing"}))
    candidate = AgentCandidate("partial", frozenset({"code_generation"}))

    score = selector.score(context, candidate)

    assert score.matched_capabilities == ("code_generation",)
    assert score.missing_capabilities == ("file_editing",)
    assert not score.is_compatible
    with pytest.raises(SelectorNoSuitableAgentError):
        selector.select(context, [candidate])


def test_selector_reports_missing_required_capability() -> None:
    selector = AgentSelector()
    context = SelectionContext(required_capabilities=frozenset({"static_analysis"}))

    with pytest.raises(SelectorNoSuitableAgentError, match="missing capabilities"):
        selector.select(context, [AgentCandidate("coder", frozenset({"code_generation"}))])


def test_selector_filters_unavailable_and_unhealthy_agents() -> None:
    selector = AgentSelector()
    context = SelectionContext(required_capabilities=frozenset({"test_execution"}))

    result = selector.select(
        context,
        [
            AgentCandidate("disabled", frozenset({"test_execution"}), enabled=False),
            AgentCandidate(
                "unhealthy",
                frozenset({"test_execution"}),
                health_status=AgentHealthStatus.UNHEALTHY,
            ),
            AgentCandidate("healthy", frozenset({"test_execution"})),
        ],
    )

    assert result.selected_agent_id == "healthy"


def test_selector_uses_priority_and_deterministic_tie_breaking() -> None:
    selector = AgentSelector()
    context = SelectionContext(required_capabilities=frozenset({"code_generation"}))

    priority_result = selector.select(
        context,
        [
            AgentCandidate("low", frozenset({"code_generation"}), priority=10),
            AgentCandidate("high", frozenset({"code_generation"}), priority=90),
        ],
    )
    tie_result = selector.select(
        context,
        [
            AgentCandidate("beta", frozenset({"code_generation"}), priority=50),
            AgentCandidate("alpha", frozenset({"code_generation"}), priority=50),
        ],
    )

    assert priority_result.selected_agent_id == "high"
    assert tie_result.selected_agent_id == "alpha"


def test_selector_falls_back_to_next_valid_candidate() -> None:
    selector = AgentSelector()
    context = SelectionContext(required_capabilities=frozenset({"code_generation"}))

    result = selector.select(
        context,
        [
            AgentCandidate(
                "best-but-down",
                frozenset({"code_generation"}),
                priority=100,
                enabled=False,
            ),
            AgentCandidate("next", frozenset({"code_generation"}), priority=1),
        ],
    )

    assert result.selected_agent_id == "next"


@pytest.mark.asyncio
async def test_agent_manager_selects_and_executes_best_agent() -> None:
    manager = AgentManager()
    coder = FakeAgent("coder", {"code_generation", "file_editing"}, priority=10)
    reviewer = FakeAgent("reviewer", {"code_review"}, priority=100)
    await manager.register(reviewer)
    await manager.register(coder)

    result = await manager.execute(
        TaskLike(required_capabilities=frozenset({"code_generation", "file_editing"}))
    )

    assert result.output == {"agent": "coder"}
    assert coder.executed == ["task-1"]


@pytest.mark.asyncio
async def test_agent_manager_rejects_no_suitable_agent() -> None:
    manager = AgentManager()
    await manager.register(FakeAgent("reviewer", {"code_review"}))

    with pytest.raises(NoSuitableAgentError):
        await manager.select_agent(TaskLike(required_capabilities=frozenset({"test_execution"})))


@pytest.mark.asyncio
async def test_execution_engine_uses_agent_manager_selection() -> None:
    manager = AgentManager()
    await manager.register(FakeAgent("coder", {"code_generation"}, priority=10))
    engine = ExecutionEngine(manager, default_timeout_seconds=1.0)
    task = Task(
        id=str(uuid4()),
        repository_id=str(uuid4()),
        title="implement feature",
        status=TaskStatus.PENDING,
        priority=TaskPriority.HIGH,
        max_retries=0,
        context={"required_capabilities": ["code_generation"]},
    )

    result = await engine.execute(task)

    assert result.outcome is ExecutionOutcome.SUCCESS
    assert result.result == {"agent": "coder"}
    assert task.status is TaskStatus.COMPLETED
