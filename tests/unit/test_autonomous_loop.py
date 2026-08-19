"""End-to-end tests for the autonomous task execution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from kodiak.agents.manager import AgentManager
from kodiak.db.models.task import Task, TaskPriority
from kodiak.db.models.task import TaskStatus as DbTaskStatus
from kodiak.memory.models import MemoryType
from kodiak.memory.service import MemoryService
from kodiak.orchestration.autonomous_loop import AutonomousTaskLoop
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome, RetryPolicy
from kodiak.orchestration.reflection import ReflectionAction, ReflectionService
from kodiak.orchestration.state import TaskState, TaskStatus
from kodiak.orchestration.task_planner import TaskPlanner
from kodiak.orchestration.verification import TaskVerifier, VerificationStatus


@dataclass
class TaskLike:
    task_id: str = "task-1"
    task_type: str = "implementation"
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    priority: Any = None


class FakeAgent:
    def __init__(
        self,
        agent_id: str,
        capabilities: set[str],
        *,
        output: dict[str, Any] | None = None,
        fail: bool = False,
        non_retryable: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.name = agent_id
        self.capabilities = frozenset(capabilities)
        self.output = output or {
            "agent": agent_id,
            "summary": "done",
            "verification_status": "verified",
        }
        self.fail = fail
        self.non_retryable = non_retryable
        self.executed = 0

    async def execute(self, task: TaskLike) -> dict[str, Any]:
        self.executed += 1
        if self.fail:
            if self.non_retryable:
                from kodiak.orchestration.execution.exceptions import NonRetryableExecutionError

                raise NonRetryableExecutionError("non-retryable", RuntimeError("non-retryable"))
            raise RuntimeError("agent execution failed")
        return self.output

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def coder_agent() -> FakeAgent:
    return FakeAgent("coder", {"code_generation", "file_editing"})


@pytest.fixture
def research_agent() -> FakeAgent:
    return FakeAgent("research", {"research", "context_retrieval"})


@pytest.fixture
def reviewer_agent() -> FakeAgent:
    return FakeAgent("reviewer", {"code_review"})


@pytest.fixture
def tester_agent() -> FakeAgent:
    return FakeAgent("tester", {"test_execution"})


@pytest.fixture
def planner_agent() -> FakeAgent:
    return FakeAgent("planner", {"planning"})


@pytest.fixture
def retrieval_agent() -> FakeAgent:
    return FakeAgent("retrieval", {"context_retrieval", "research"})


@pytest.fixture
def debugger_agent() -> FakeAgent:
    return FakeAgent("debugger", {"debugging"})


async def _register_agents(manager: AgentManager, *agents: FakeAgent) -> None:
    for agent in agents:
        await manager.register(agent)


@pytest.mark.asyncio
async def test_autonomous_loop_success_end_to_end(
    coder_agent: FakeAgent,
    research_agent: FakeAgent,
    reviewer_agent: FakeAgent,
    tester_agent: FakeAgent,
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
) -> None:
    manager = AgentManager()
    for agent in (
        coder_agent,
        research_agent,
        reviewer_agent,
        tester_agent,
        planner_agent,
        retrieval_agent,
    ):
        await manager.register(agent)

    memory = MemoryService()
    await memory.add(
        "Previous successful implementation used dependency injection.",
        memory_type=MemoryType.SEMANTIC,
        tags=["implementation"],
    )

    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=memory,
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=5.0),
        max_loop_attempts=2,
    )

    result = await loop.run("implement feature flag support", workspace="D:/Kodiak")

    assert result.success
    assert result.task_state.status is TaskStatus.COMPLETED
    assert result.plan is not None
    assert len(result.plan.tasks) >= 1
    assert result.verification_result is not None
    assert result.verification_result.status is VerificationStatus.VERIFIED
    assert result.memory_stored is True
    assert result.selected_agent is not None
    assert coder_agent.executed >= 1


@pytest.mark.asyncio
async def test_autonomous_loop_enters_planning_state(
    coder_agent: FakeAgent,
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
) -> None:
    manager = AgentManager()
    for agent in (coder_agent, planner_agent, retrieval_agent):
        await manager.register(agent)

    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=2.0),
    )
    result = await loop.run("create utility module")
    assert result.task_state.metadata.get("plan") is not None
    assert result.task_state.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}


@pytest.mark.asyncio
async def test_verification_failure_triggers_reflection(
    coder_agent: FakeAgent,
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
) -> None:
    failing = FakeAgent(
        "coder",
        {"code_generation", "file_editing"},
        output={"agent": "coder", "verification_status": "failed"},
    )
    manager = AgentManager()
    await _register_agents(manager, retrieval_agent, planner_agent, failing)

    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=2.0),
        max_loop_attempts=1,
    )
    result = await loop.run("implement broken feature")

    assert not result.success
    assert result.reflection_results
    assert result.reflection_results[-1].action in {
        ReflectionAction.STOP,
        ReflectionAction.RETRY,
    }


@pytest.mark.asyncio
async def test_retry_path_reexecutes_without_replan(
    coder_agent: FakeAgent,
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
    tester_agent: FakeAgent,
    reviewer_agent: FakeAgent,
) -> None:
    calls = {"count": 0}

    class FlakyAgent(FakeAgent):
        async def execute(self, task: TaskLike) -> dict[str, Any]:
            calls["count"] += 1
            if calls["count"] <= 1:
                raise RuntimeError("agent execution failed")
            return {"agent": "coder", "summary": "fixed", "verification_status": "verified"}

    manager = AgentManager()
    await _register_agents(
        manager,
        retrieval_agent,
        planner_agent,
        FlakyAgent("coder", {"code_generation", "file_editing"}),
        tester_agent,
        reviewer_agent,
    )

    reflection = ReflectionService(max_retries=2, max_replans=0)
    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=2.0),
        reflection=reflection,
        max_loop_attempts=2,
    )
    result = await loop.run("implement flaky module")

    assert result.success
    assert calls["count"] >= 2


@pytest.mark.asyncio
async def test_replan_path_after_multiple_failures(
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
) -> None:
    manager = AgentManager()
    await _register_agents(
        manager,
        retrieval_agent,
        planner_agent,
        FakeAgent(
            "coder",
            {"code_generation", "file_editing"},
            output={"agent": "coder", "verification_status": "failed"},
        ),
    )

    reflection = ReflectionService(max_retries=3, max_replans=2)
    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=2.0),
        reflection=reflection,
        max_loop_attempts=3,
    )
    result = await loop.run("implement complex feature")

    assert result.reflection_results
    assert any(r.action is ReflectionAction.REPLAN for r in result.reflection_results)


@pytest.mark.asyncio
async def test_non_retryable_failure_stops_task(
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
    tester_agent: FakeAgent,
    reviewer_agent: FakeAgent,
) -> None:
    manager = AgentManager()
    await _register_agents(
        manager,
        retrieval_agent,
        planner_agent,
        FakeAgent(
            "coder",
            {"code_generation", "file_editing"},
            fail=True,
            non_retryable=True,
        ),
        tester_agent,
        reviewer_agent,
    )

    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=2.0),
        max_loop_attempts=3,
    )
    result = await loop.run("implement secure auth")

    assert result.task_state.status is TaskStatus.FAILED
    assert result.reflection_results
    assert result.reflection_results[-1].retryable is False


@pytest.mark.asyncio
async def test_cancellation_stops_execution(
    coder_agent: FakeAgent,
    planner_agent: FakeAgent,
    retrieval_agent: FakeAgent,
) -> None:
    manager = AgentManager()
    await _register_agents(manager, coder_agent, planner_agent, retrieval_agent)

    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=2.0),
    )
    loop.cancel()
    result = await loop.run("implement cancellation support")

    assert result.task_state.status is TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_verifier_and_reflection_units() -> None:
    verifier = TaskVerifier()
    state = TaskState(title="t", objective="do work")
    from kodiak.orchestration.execution.models import ExecutionResult

    ok = await verifier.verify(
        goal="do work",
        plan=None,
        execution_result=ExecutionResult(
            task_id=state.task_id,
            outcome=ExecutionOutcome.SUCCESS,
            attempts=1,
            duration_seconds=0.1,
            result={"summary": "done"},
            final_status=DbTaskStatus.COMPLETED,
        ),
        task_state=state,
    )
    assert ok.status is VerificationStatus.VERIFIED

    reflection = ReflectionService(max_retries=1)
    decision = await reflection.reflect(
        task_state=state,
        verification_result=ok,
        execution_result=ExecutionResult(
            task_id=state.task_id,
            outcome=ExecutionOutcome.SUCCESS,
            attempts=1,
            duration_seconds=0.1,
            result={"summary": "done"},
            final_status=DbTaskStatus.COMPLETED,
        ),
        attempt=1,
        replan_count=0,
    )
    assert decision.action is ReflectionAction.STOP


@pytest.mark.asyncio
async def test_execution_engine_still_works_with_manager() -> None:
    manager = AgentManager()
    await manager.register(FakeAgent("coder", {"code_generation"}))
    engine = ExecutionEngine(manager, default_timeout_seconds=1.0)
    task = Task(
        id=str(uuid4()),
        repository_id=str(uuid4()),
        title="implement feature",
        status=DbTaskStatus.PENDING,
        priority=TaskPriority.HIGH,
        max_retries=0,
        context={"required_capabilities": ["code_generation"]},
    )
    result = await engine.execute(task, retry_policy=RetryPolicy(max_attempts=1))
    assert result.outcome is ExecutionOutcome.SUCCESS
