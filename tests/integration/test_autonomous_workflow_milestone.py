from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kodiak.agents.adapters import BaseAgentAdapter
from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent
from kodiak.agents.manager import AgentManager
from kodiak.memory.models import MemoryType
from kodiak.memory.service import MemoryService
from kodiak.orchestration.autonomous_loop import AutonomousTaskLoop
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.git_workflow import (
    AutonomousGitWorkflow,
    GitWorkflowRequest,
    GitWorkflowStatus,
)
from kodiak.orchestration.reflection import ReflectionEngine, RepairStrategy
from kodiak.orchestration.state import TaskState, TaskStatus
from kodiak.orchestration.task_planner import ExecutableTask, ExecutionPlan, TaskPlanner
from kodiak.orchestration.verification import VerificationEngine, VerificationStatus
from kodiak.tools.builtin import register_builtin_tools
from kodiak.tools.registry import ToolRegistry
from kodiak.tools.router import ToolRouter
from kodiak.utils.git_utils import run_git


class StaticPipeline:
    def __init__(self, plan: ExecutionPlan | None = None, exc: Exception | None = None) -> None:
        self._plan = plan
        self.exc = exc

    async def plan(self, goal: str, context: dict[str, Any]) -> ExecutionPlan:
        if self.exc is not None:
            raise self.exc
        assert self._plan is not None
        return self._plan


class ToolFixAgent(BaseAgent):
    role = AgentRole.CODER
    capabilities = frozenset(
        {"code_generation", "file_editing", "filesystem", "read_file", "write_file"}
    )

    def __init__(self, content: str | None = None, delay_seconds: float = 0.0) -> None:
        super().__init__()
        self.content = content
        self.delay_seconds = delay_seconds
        self.tool_calls: list[str] = []

    async def _run(self, input_: AgentInput) -> AgentOutput:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)

        target_file = str(input_.context["target_file"])
        content = self.content if self.content is not None else str(input_.context["content"])

        read_result = await self.invoke_tool("read_file", {"path": target_file})
        self.tool_calls.append("read_file")
        if not read_result.success:
            raise RuntimeError(read_result.error or "read_file failed")

        write_result = await self.invoke_tool(
            "write_file",
            {"path": target_file, "content": content},
        )
        self.tool_calls.append("write_file")
        if not write_result.success:
            raise RuntimeError(write_result.error or "write_file failed")

        return self._make_output(
            input_,
            {
                "agent": self.agent_id,
                "summary": f"Updated {target_file}",
                "tool_calls": list(self.tool_calls),
            },
        )


class ToolFailureAgent(ToolFixAgent):
    async def _run(self, input_: AgentInput) -> AgentOutput:
        result = await self.invoke_tool(
            "write_file",
            {"path": "../outside.py", "content": "x = 1\n"},
        )
        self.tool_calls.append("write_file")
        if not result.success:
            raise RuntimeError(result.error or "write_file failed")
        return self._make_output(input_, {"summary": "unexpected"})


def _make_repo(repo: Path) -> None:
    repo.mkdir()
    (repo / "math_utils.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_math_utils.py").write_text(
        "from math_utils import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )


def _plan(goal: str, repo: Path, content: str) -> ExecutionPlan:
    task = ExecutableTask(
        name="fix_add_function",
        agent_type="coder",
        description="Modify the tiny Python function so the test passes.",
        input_data={
            "target_file": "math_utils.py",
            "content": content,
            "verification": {
                "run_tests": {
                    "test_target": "tests",
                    "options": ["-q", "-o", "cache_dir=.pytest_cache"],
                    "timeout_seconds": 10,
                }
            },
        },
        tool_names=["read_file", "write_file"],
    )
    return ExecutionPlan(
        goal=goal,
        tasks=[task],
        execution_order=[task.id],
        parallel_groups=[[task.id]],
        acceptance_criteria=["pytest tests passes"],
        metadata={"workspace": str(repo)},
    )


async def _loop(
    repo: Path,
    plan: ExecutionPlan,
    agent: BaseAgent | None = None,
    *,
    max_loop_attempts: int = 2,
) -> tuple[AutonomousTaskLoop, ToolFixAgent | None]:
    registry = ToolRegistry()
    register_builtin_tools(registry, workspace_root=repo)
    router = ToolRouter(registry=registry)

    manager = AgentManager(tool_router=router)
    registered_agent = agent if agent is not None else ToolFixAgent()
    await manager.register(BaseAgentAdapter(registered_agent))

    verification = VerificationEngine(tool_router=router)
    reflection = ReflectionEngine()
    engine = ExecutionEngine(
        manager,
        default_timeout_seconds=5.0,
        verification_engine=verification,
        reflection_engine=reflection,
    )
    memory = MemoryService()
    await memory.add(
        "Past fixes should be recalled before planning.",
        memory_type=MemoryType.SEMANTIC,
        tags=["integration"],
    )
    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(pipeline=StaticPipeline(plan)),
        memory_service=memory,
        agent_manager=manager,
        execution_engine=engine,
        verifier=verification,
        reflection=reflection,
        tool_router=router,
        max_loop_attempts=max_loop_attempts,
    )
    return loop, registered_agent if isinstance(registered_agent, ToolFixAgent) else None


@pytest.mark.asyncio
async def test_complete_autonomous_workflow_modifies_code_and_verifies_tests(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Modify add so the existing test passes"
    content = "def add(a, b):\n    return a + b\n"
    loop, agent = await _loop(repo, _plan(goal, repo, content))

    result = await loop.run(goal, workspace=repo)

    assert result.success is True
    assert result.task_state.status is TaskStatus.COMPLETED
    assert result.plan is not None
    assert result.task_state.metadata["workflow"]["status"] == "completed"
    assert result.execution_result is not None
    assert result.execution_result.verification is not None
    assert result.execution_result.verification["status"] == VerificationStatus.VERIFIED.value
    assert result.verification_result is not None
    assert result.verification_result.status is VerificationStatus.VERIFIED
    assert result.memory_stored is True
    assert agent is not None
    assert agent.tool_calls == ["read_file", "write_file"]
    assert "return a + b" in (repo / "math_utils.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_successful_execution_with_failed_verification_reaches_reflection(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Attempt a fix that does not satisfy tests"
    wrong_content = "def add(a, b):\n    return a - b\n"
    loop, _ = await _loop(repo, _plan(goal, repo, wrong_content), max_loop_attempts=2)

    result = await loop.run(goal, workspace=repo)

    assert result.success is False
    assert result.task_state.status is TaskStatus.FAILED
    assert result.attempts == 2
    assert result.execution_result is not None
    assert result.execution_result.error is not None
    assert result.execution_result.error["type"] == "VerificationFailed"
    assert result.execution_result.verification is not None
    assert result.execution_result.verification["status"] == VerificationStatus.FAILED.value
    assert result.reflection_results
    assert result.reflection_results[-1].strategy is RepairStrategy.STOP
    assert result.task_state.error


@pytest.mark.asyncio
async def test_retry_budget_exhaustion_is_terminal(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Keep failing verification until the budget is exhausted"
    loop, _ = await _loop(
        repo,
        _plan(goal, repo, "def add(a, b):\n    return 0\n"),
        max_loop_attempts=1,
    )

    result = await loop.run(goal, workspace=repo)

    assert result.success is False
    assert result.task_state.status is TaskStatus.FAILED
    assert result.attempts == 1
    assert result.reflection_results[-1].should_retry is False
    assert result.task_state.finished_at is not None


@pytest.mark.asyncio
async def test_cancellation_terminates_before_workflow_execution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Cancel before running"
    loop, _ = await _loop(repo, _plan(goal, repo, "def add(a, b):\n    return a + b\n"))

    loop.cancel()
    result = await loop.run(goal, workspace=repo)

    assert result.success is False
    assert result.task_state.status is TaskStatus.CANCELLED
    assert result.execution_result is None
    assert result.task_state.finished_at is not None


@pytest.mark.asyncio
async def test_agent_selection_failure_is_reported_cleanly(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Run without a matching agent"
    plan = _plan(goal, repo, "def add(a, b):\n    return a + b\n")
    registry = ToolRegistry()
    register_builtin_tools(registry, workspace_root=repo)
    router = ToolRouter(registry=registry)
    manager = AgentManager(tool_router=router)
    verification = VerificationEngine(tool_router=router)
    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(pipeline=StaticPipeline(plan)),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, verification_engine=verification),
        verifier=verification,
        reflection=ReflectionEngine(),
        tool_router=router,
        max_loop_attempts=1,
    )

    result = await loop.run(goal, workspace=repo)

    assert result.success is False
    assert result.task_state.status is TaskStatus.FAILED
    assert result.execution_result is not None
    assert result.execution_result.error is not None
    assert "No agents are registered" in result.execution_result.error["message"]
    assert result.task_state.finished_at is not None


@pytest.mark.asyncio
async def test_tool_failure_is_preserved_in_execution_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Try a forbidden file write"
    loop, agent = await _loop(
        repo,
        _plan(goal, repo, "def add(a, b):\n    return a + b\n"),
        agent=ToolFailureAgent(),
        max_loop_attempts=1,
    )

    result = await loop.run(goal, workspace=repo)

    assert result.success is False
    assert result.execution_result is not None
    assert result.execution_result.error is not None
    assert "outside workspace boundary" in result.execution_result.error["message"]
    assert agent is not None
    assert agent.tool_calls == ["write_file"]


@pytest.mark.asyncio
async def test_unexpected_planner_exception_leaves_terminal_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    goal = "Planner raises unexpectedly"
    manager = AgentManager()
    state_error = RuntimeError("planner unavailable")
    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(pipeline=StaticPipeline(exc=state_error)),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager),
        max_loop_attempts=1,
    )
    task_state = TaskState(title=goal, objective=goal)

    with pytest.raises(RuntimeError, match="planner unavailable"):
        await loop.run(goal, workspace=repo, task_state=task_state)

    assert task_state.status is TaskStatus.FAILED
    assert task_state.error == "planner unavailable"
    assert task_state.finished_at is not None
    assert task_state.metadata["failed_stage"] == TaskStatus.PLANNING.value
    assert loop._active_task_state is None


@pytest.mark.asyncio
async def test_real_autonomous_loop_reaches_verified_selective_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    repo = tmp_path / "repo"
    _make_repo(repo)
    run_git(["init", "-b", "main"], repo)
    run_git(["config", "user.email", "kodiak@example.test"], repo)
    run_git(["config", "user.name", "Kodiak Test"], repo)
    run_git(["add", "--", "math_utils.py", "tests/test_math_utils.py"], repo)
    run_git(["commit", "-m", "initial"], repo)
    goal = "Modify add so the existing test passes"
    content = "def add(a, b):\n    return a + b\n"
    loop, agent = await _loop(repo, _plan(goal, repo, content))
    workflow = AutonomousGitWorkflow(loop)

    result = await workflow.run(
        GitWorkflowRequest(
            task_id="real-e2e",
            title="fix add implementation",
            goal=goal,
            repository=repo,
            intended_paths=("math_utils.py",),
        )
    )

    assert result.status is GitWorkflowStatus.COMMITTED
    assert result.commit_sha == run_git(["rev-parse", "HEAD"], repo)
    assert result.changed_files == ("math_utils.py",)
    assert run_git(["show", "--pretty=", "--name-only", "HEAD"], repo) == "math_utils.py"
    assert result.verification is not None
    assert result.verification["status"] == VerificationStatus.VERIFIED.value
    assert agent is not None
    assert agent.tool_calls == ["read_file", "write_file"]
