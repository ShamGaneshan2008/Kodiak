"""Integration tests for reflection with execution, verification, and planning."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from kodiak.agents.adapters import BaseAgentAdapter
from kodiak.agents.manager import AgentManager
from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome, RetryPolicy
from kodiak.orchestration.planning import PlanReplanner
from kodiak.orchestration.reflection import ReflectionEngine
from kodiak.orchestration.reflection.loop import SelfRepairLoop
from kodiak.orchestration.reflection.models import ReflectionOutcome, RepairStrategy
from kodiak.orchestration.task_planner import ExecutableTask, ExecutionPlan
from kodiak.orchestration.verification import VerificationEngine


@pytest.mark.asyncio
async def test_execution_engine_reflection_retry_on_verification_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")

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
        max_retries=2,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
            "verification": {
                "required_output_fields": ["analysis", "missing_on_first_attempt"],
            },
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(),
        reflection_engine=ReflectionEngine(),
    )

    result = await engine.execute(task)

    assert result.reflection is not None
    assert result.reflection.get("strategy") in {
        RepairStrategy.RETRY.value,
        RepairStrategy.STOP.value,
        RepairStrategy.REPLAN.value,
    }
    assert "reflection_history" in task.context or "reflection" in task.context


@pytest.mark.asyncio
async def test_self_repair_loop_prevents_infinite_cycles(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

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
            "verification": {"required_output_fields": ["never_present"]},
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(),
        reflection_engine=ReflectionEngine(),
    )
    loop = SelfRepairLoop(engine, max_cycles=2)

    result = await loop.run(task)

    assert result.outcome in {
        ExecutionOutcome.FAILURE,
        ExecutionOutcome.RETRY_EXHAUSTED,
    }
    assert result.reflection is not None


@pytest.mark.asyncio
async def test_reflection_feeds_planner_replan() -> None:
    t1 = ExecutableTask(name="t1", agent_type="retrieval", status="completed")
    t2 = ExecutableTask(name="t2", agent_type="coder", dependencies=[t1.id], status="failed")
    plan = ExecutionPlan(
        goal="Feature goal",
        tasks=[t1, t2],
        execution_order=[t1.id, t2.id],
        parallel_groups=[[t1.id], [t2.id]],
    )

    replanner = PlanReplanner()
    updated = replanner.replan(
        plan,
        {
            "task_id": str(t2.id),
            "error": {"message": "verification failed"},
            "reflection": {
                "root_cause": "Tests did not pass: pytest failed",
                "category": "test_failure",
                "strategy": "replan",
            },
        },
    )

    names = {task.name for task in updated.tasks}
    assert "debug_t2" in names
    assert "repair_t2" in names


@pytest.mark.asyncio
async def test_successful_execution_skips_repair(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")

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
        max_retries=1,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
            "verification": {"required_output_fields": ["analysis"]},
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(),
        reflection_engine=ReflectionEngine(),
    )
    result = await engine.execute(task)

    assert result.outcome is ExecutionOutcome.SUCCESS
    assert result.verification is not None
    assert result.verification["status"] == "verified"
    assert result.reflection is None
