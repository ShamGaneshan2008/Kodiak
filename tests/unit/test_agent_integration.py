"""Integration tests for specialized agent standardization."""

from __future__ import annotations

import inspect
from pathlib import Path
from uuid import uuid4

import pytest

from kodiak.agents.adapters import BaseAgentAdapter
from kodiak.agents.base import AgentInput, AgentRole, BaseAgent
from kodiak.agents.discovery import AgentDiscovery
from kodiak.agents.lifecycle import AgentLifecycleManager
from kodiak.agents.manager import AgentManager
from kodiak.agents.registry import AgentRegistry
from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome


CONCRETE_AGENT_MODULES = {
    "debugger": "DebuggerAgent",
    "reflection": "ReflectionAgent",
    "retrieval": "RetrievalAgent",
    "memory_agent": "MemoryAgent",
    "learning": "LearningAgent",
    "evaluation": "EvaluationAgent",
    "planner": "PlannerAgent",
    "repository": "RepositoryAnalyzerAgent",
}

OPTIONAL_HEAVY_AGENT_MODULES = {
    "git": "GitAgent",
    "research": "ResearchAgent",
    "reviewer": "ReviewAgent",
    "tester": "TestAgent",
    "coder": "CodingAgent",
}


@pytest.mark.parametrize("module_name,class_name", list(CONCRETE_AGENT_MODULES.items()))
def test_concrete_agent_inherits_base_agent(module_name: str, class_name: str) -> None:
    module = __import__(f"kodiak.agents.{module_name}", fromlist=[class_name])
    agent_cls = getattr(module, class_name)
    assert issubclass(agent_cls, BaseAgent)
    assert not inspect.isabstract(agent_cls)
    assert isinstance(getattr(agent_cls, "role", None), AgentRole)
    assert agent_cls.resolved_capabilities()


@pytest.mark.parametrize("module_name,class_name", list(OPTIONAL_HEAVY_AGENT_MODULES.items()))
def test_heavy_agent_modules_define_base_agent_subclass(module_name: str, class_name: str) -> None:
    """Verify heavy agents at source level without importing optional dependency chains."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "kodiak" / "agents" / f"{module_name}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and any(base.id == "BaseAgent" for base in node.bases if isinstance(base, ast.Name))
    }
    assert class_name in classes


def test_repository_agent_identity_and_capabilities() -> None:
    agent = RepositoryAnalyzerAgent()
    assert agent.agent_id == "repository"
    assert "repository_analysis" in agent.resolved_capabilities()


@pytest.mark.asyncio
async def test_discovery_finds_repository_agent(tmp_path: Path) -> None:
    registry = AgentRegistry()
    discovery = AgentDiscovery(registry, modules=["kodiak.agents.repository"])
    result = await discovery.discover_and_register()
    assert "repository" in result.registered


@pytest.mark.asyncio
async def test_end_to_end_repository_agent_execution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n", encoding="utf-8")

    registry = AgentRegistry()
    lifecycle = AgentLifecycleManager(registry)
    manager = AgentManager(registry=registry, lifecycle=lifecycle)

    agent = RepositoryAnalyzerAgent()
    await manager.register(BaseAgentAdapter(agent))
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("repository")

    class _Task:
        task_id = "repo-task"
        task_type = "analyze repository"
        required_capabilities = frozenset({"repository_analysis"})
        priority = TaskPriority.MEDIUM
        context = {"repository_path": str(repo)}

    result = await manager.execute(_Task())
    assert "analysis" in result.output
    assert result.output["analysis"].file_count == 1


@pytest.mark.asyncio
async def test_execution_engine_runs_through_agent_manager(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")

    registry = AgentRegistry()
    manager = AgentManager(registry=registry)
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
    assert result.result["analysis"].file_count == 1
