"""Integration tests for execution engine verification."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from kodiak.agents.adapters import BaseAgentAdapter
from kodiak.agents.manager import AgentManager
from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome
from kodiak.orchestration.verification import VerificationEngine, VerificationStatus
from kodiak.tools.builtin import TestRunnerTool
from kodiak.tools.registry import ToolRegistry
from kodiak.tools.router import ToolRouter


@pytest.mark.asyncio
async def test_execution_engine_verification_passes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    manager = AgentManager()
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    task = Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="Analyze repository",
        description="scan repo",
        source=TaskSource.API,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        max_retries=0,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
            "verification": {
                "required_output_fields": ["analysis"],
            },
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(),
    )
    result = await engine.execute(task)

    assert result.outcome is ExecutionOutcome.SUCCESS
    assert result.verification is not None
    assert result.verification["status"] == VerificationStatus.VERIFIED.value
    assert task.status is TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_execution_engine_verification_failure_downgrades_result(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    manager = AgentManager()
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    task = Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="Analyze repository",
        description="scan repo",
        source=TaskSource.API,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        max_retries=0,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
            "verification": {
                "required_output_fields": ["missing_field"],
            },
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(),
    )
    result = await engine.execute(task)

    assert result.outcome is ExecutionOutcome.FAILURE
    assert result.verification is not None
    assert result.verification["status"] == VerificationStatus.FAILED.value
    assert task.status is TaskStatus.FAILED


@pytest.mark.asyncio
async def test_execution_engine_test_verification_via_tool_router(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text(
        "def test_passes():\n    assert True\n",
        encoding="utf-8",
    )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(TestRunnerTool(workspace_root=repo))
    router = ToolRouter(registry=tool_registry)

    manager = AgentManager()
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    task = Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="Run tests",
        description="verify tests",
        source=TaskSource.API,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        max_retries=0,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
            "verification": {
                "workspace_root": str(repo),
                "required_output_fields": ["analysis"],
                "run_tests": {"test_target": "tests/test_ok.py"},
            },
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(tool_router=router),
    )
    result = await engine.execute(task)

    assert result.outcome is ExecutionOutcome.SUCCESS
    assert result.verification is not None
    assert result.verification["status"] == VerificationStatus.VERIFIED.value
    verifier_names = {item["verifier"] for item in result.verification["evidence"]}
    assert "test" in verifier_names
