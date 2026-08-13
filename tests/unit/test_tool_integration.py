"""Integration tests for agent → ToolRouter → tool execution."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from kodiak.agents.adapters import BaseAgentAdapter
from kodiak.agents.base import AgentInput
from kodiak.agents.manager import AgentManager
from kodiak.agents.registry import AgentRegistry
from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome
from kodiak.tools.builtin import ListDirTool, register_builtin_tools
from kodiak.tools.registry import ToolRegistry
from kodiak.tools.router import ToolRouter


@pytest.mark.asyncio
async def test_agent_invokes_tool_through_router(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    registry = ToolRegistry()
    register_builtin_tools(registry, workspace_root=repo)
    router = ToolRouter(registry=registry)

    agent = RepositoryAnalyzerAgent(tool_router=router)
    output = await agent.run(
        AgentInput(
            task_id="t-1",
            project_id="p-1",
            instruction="analyze",
            context={"repository_path": str(repo)},
        )
    )

    assert output.success is True
    assert "tool_listing" in output.result
    assert any(entry["name"] == "main.py" for entry in output.result["tool_listing"]["entries"])


@pytest.mark.asyncio
async def test_agent_manager_binds_tool_router(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")

    tool_registry = ToolRegistry()
    tool_registry.register_tool(ListDirTool(workspace_root=repo))
    router = ToolRouter(registry=tool_registry)

    manager = AgentManager(tool_router=router)
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    class _Task:
        task_id = "repo-task"
        task_type = "analyze repository"
        required_capabilities = frozenset({"repository_analysis"})
        priority = TaskPriority.MEDIUM
        context = {"repository_path": str(repo)}

    result = await manager.execute(_Task())
    assert "tool_listing" in result.output
    assert result.output["analysis"].file_count == 1


@pytest.mark.asyncio
async def test_execution_engine_with_tool_router(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("pass\n", encoding="utf-8")

    tool_registry = ToolRegistry()
    register_builtin_tools(tool_registry, workspace_root=repo)
    router = ToolRouter(registry=tool_registry)

    manager = AgentManager(tool_router=router)
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    task = Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="Analyze repository",
        description="scan repo",
        source=TaskSource.API,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        max_retries=3,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
        },
    )

    engine = ExecutionEngine(agent_manager=manager)
    result = await engine.execute(task)
    assert result.outcome is ExecutionOutcome.SUCCESS
    assert "tool_listing" in result.result


@pytest.mark.asyncio
async def test_agent_manager_tool_permission_check(tmp_path: Path) -> None:
    tool_registry = ToolRegistry()
    register_builtin_tools(tool_registry, workspace_root=tmp_path)
    router = ToolRouter(registry=tool_registry)

    registry = AgentRegistry()
    manager = AgentManager(registry=registry, tool_router=router)
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    allowed = await manager.can_agent_use_tool("repository", "read_file")
    assert allowed is True

    denied = await manager.can_agent_use_tool("repository", "write_file")
    assert denied is False
