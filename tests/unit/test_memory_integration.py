"""Integration tests for memory with execution, reflection, and planning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kodiak.agents.adapters import BaseAgentAdapter
from kodiak.agents.manager import AgentManager
from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.memory.experience import ExperienceExtractor, ExperienceSanitizer
from kodiak.memory.integration import MemoryIntegration
from kodiak.memory.service import MemoryService
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.orchestration.reflection import ReflectionEngine
from kodiak.orchestration.task_planner import TaskPlanner
from kodiak.orchestration.verification import VerificationEngine


def _make_task(**overrides) -> Task:
    defaults = {
        "id": uuid4(),
        "repository_id": str(uuid4()),
        "title": "Fix pytest missing fixture failure",
        "description": "Add missing fixture for authentication tests",
        "source": TaskSource.API,
        "status": TaskStatus.PENDING,
        "priority": TaskPriority.MEDIUM,
        "max_retries": 1,
        "context": {
            "task_type": "debugging",
            "required_capabilities": ["testing", "pytest"],
        },
    }
    defaults.update(overrides)
    return Task(**defaults)


def _success_result(task: Task, *, verification: dict | None = None) -> ExecutionResult:
    return ExecutionResult(
        task_id=str(task.id),
        outcome=ExecutionOutcome.SUCCESS,
        attempts=1,
        duration_seconds=1.5,
        result={"agent": "tester", "analysis": "done"},
        final_status=TaskStatus.COMPLETED,
        verification=verification or {"status": "verified"},
    )


def _failure_result(task: Task, *, reflection: dict | None = None) -> ExecutionResult:
    return ExecutionResult(
        task_id=str(task.id),
        outcome=ExecutionOutcome.FAILURE,
        attempts=1,
        duration_seconds=2.0,
        result={"agent": "debugger"},
        error={"type": "VerificationFailed", "message": "tests failed"},
        final_status=TaskStatus.FAILED,
        verification={"status": "failed"},
        reflection=reflection,
    )


@pytest.mark.asyncio
async def test_experience_extractor_builds_structured_experience() -> None:
    task = _make_task()
    reflection = {
        "category": "missing_fixture",
        "root_cause": "auth_client fixture was not defined",
        "suggested_correction": "Add auth_client fixture in conftest.py",
        "confidence": 0.9,
    }
    result = _failure_result(task, reflection=reflection)

    experience = ExperienceExtractor().extract(task, result)

    assert experience is not None
    assert experience.task_type == "debugging"
    assert experience.failure_category == "missing_fixture"
    assert experience.root_cause == "auth_client fixture was not defined"
    assert experience.repair_performed == "Add auth_client fixture in conftest.py"
    assert experience.final_result == "failure"


@pytest.mark.asyncio
async def test_successful_experience_storage() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory)
    task = _make_task()
    result = _success_result(task)

    stored = await integration.record_execution(task, result)

    assert stored is True
    episodes = await memory.episodic.get_recent_episodes(limit=10)
    assert len(episodes) >= 1
    assert "pytest" in episodes[0].goal.lower() or "fixture" in episodes[0].goal.lower()


@pytest.mark.asyncio
async def test_failed_experience_storage_with_reflection() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory)
    task = _make_task()
    result = _failure_result(
        task,
        reflection={
            "category": "missing_fixture",
            "root_cause": "fixture not found",
            "suggested_correction": "define fixture in conftest.py",
        },
    )

    stored = await integration.record_execution(task, result)

    assert stored is True
    facts = await memory.semantic.list_facts(limit=10)
    assert any("missing_fixture" in fact.content or "fixture" in fact.content for fact in facts)


@pytest.mark.asyncio
async def test_reflection_to_memory_via_execution_engine(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")

    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory)
    manager = AgentManager()
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))

    task = Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="Analyze repository structure for planning",
        description="scan repository files",
        source=TaskSource.API,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        max_retries=0,
        context={
            "task_type": "analysis",
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
            "verification": {"required_output_fields": ["never_present_field"]},
        },
    )

    engine = ExecutionEngine(
        agent_manager=manager,
        verification_engine=VerificationEngine(),
        reflection_engine=ReflectionEngine(),
        memory_integration=integration,
    )

    result = await engine.execute(task)

    assert result.reflection is not None
    episodes = await memory.episodic.get_recent_episodes(limit=10)
    assert len(episodes) >= 1


@pytest.mark.asyncio
async def test_memory_retrieval_for_related_task() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory, min_relevance_score=0.2)

    related_task = _make_task(
        title="Fix pytest missing fixture in auth tests",
        context={"task_type": "debugging", "required_capabilities": ["pytest"]},
    )
    await integration.record_execution(
        related_task,
        _failure_result(
            related_task,
            reflection={
                "category": "missing_fixture",
                "root_cause": "auth_client fixture missing",
                "suggested_correction": "Add auth_client fixture",
            },
        ),
    )

    memories = await integration.retrieve_for_planning(
        "Fix pytest fixture failure in authentication tests",
        {"task_type": "debugging", "required_capabilities": ["pytest"]},
    )

    assert len(memories) >= 1
    assert memories[0].relevance_score >= 0.2


@pytest.mark.asyncio
async def test_irrelevant_memory_filtering() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory, min_relevance_score=0.5)

    unrelated_task = _make_task(
        title="Deploy kubernetes cluster to production",
        context={"task_type": "deployment", "required_capabilities": ["kubernetes"]},
    )
    await integration.record_execution(
        unrelated_task,
        _success_result(unrelated_task),
    )

    memories = await integration.retrieve_for_planning(
        "Fix pytest missing fixture in unit tests",
        {"task_type": "debugging", "required_capabilities": ["pytest"]},
    )

    assert memories == []


@pytest.mark.asyncio
async def test_secret_redaction_prevents_persisting_raw_secrets() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory)
    secret = "sk-" + "A" * 48
    task = _make_task(
        title=f"Run tests with key {secret}",
        description="debug failing tests",
    )
    result = _failure_result(
        task,
        reflection={
            "category": "config_error",
            "root_cause": f"Invalid API key {secret}",
            "suggested_correction": "Rotate compromised credentials",
        },
    )

    stored = await integration.record_execution(task, result)

    assert stored is True
    episodes = await memory.episodic.get_recent_episodes(limit=10)
    assert len(episodes) >= 1
    stored_text = f"{episodes[0].goal} {episodes[0].outcome}"
    assert secret not in stored_text


@pytest.mark.asyncio
async def test_planner_receives_relevant_memories() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory, min_relevance_score=0.2)

    prior_task = _make_task(title="Fix missing pytest fixture in auth module")
    await integration.record_execution(
        prior_task,
        _failure_result(
            prior_task,
            reflection={
                "category": "missing_fixture",
                "root_cause": "auth fixture undefined",
                "suggested_correction": "Add fixture to conftest.py",
            },
        ),
    )

    class StubPipeline:
        async def plan(self, goal: str, context: dict | None = None):
            from kodiak.orchestration.task_planner import ExecutionPlan

            ctx = context or {}
            return ExecutionPlan(
                goal=goal,
                tasks=[],
                execution_order=[],
                parallel_groups=[],
                metadata={"received_memories": ctx.get("relevant_memories", [])},
            )

    plan = await TaskPlanner(
        pipeline=StubPipeline(),
        memory_integration=integration,
    ).plan_execution(
        "Fix pytest fixture failure in authentication tests",
        {"task_type": "debugging", "required_capabilities": ["pytest"]},
    )

    assert "relevant_memories" in plan.metadata
    assert len(plan.metadata["relevant_memories"]) >= 1


@pytest.mark.asyncio
async def test_empty_memory_store_returns_empty_planning_context() -> None:
    integration = MemoryIntegration(memory_service=MemoryService())

    ctx = await integration.build_planning_context(
        "Implement new caching layer",
        {"task_type": "implementation"},
    )

    assert "relevant_memories" not in ctx


@pytest.mark.asyncio
async def test_memory_storage_failure_does_not_break_execution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")

    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory)
    integration.record_execution = AsyncMock(side_effect=RuntimeError("storage down"))

    manager = AgentManager()
    await manager.register(BaseAgentAdapter(RepositoryAnalyzerAgent()))
    task = Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="Analyze repository layout",
        description="scan repo",
        source=TaskSource.API,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        max_retries=0,
        context={
            "required_capabilities": ["repository_analysis"],
            "repository_path": str(repo),
        },
    )

    engine = ExecutionEngine(agent_manager=manager, memory_integration=integration)
    result = await engine.execute(task)

    assert result.outcome == ExecutionOutcome.SUCCESS


@pytest.mark.asyncio
async def test_experience_sanitizer_masks_secrets() -> None:
    secret = "sk-" + "B" * 48
    sanitizer = ExperienceSanitizer()
    masked = await sanitizer.sanitize_text(f"Token={secret}")

    assert secret not in masked
    assert "[REDACTED_OPENAI_API_KEY]" in masked


@pytest.mark.asyncio
async def test_should_store_skips_low_value_success_without_verification() -> None:
    task = _make_task(title="short")
    result = ExecutionResult(
        task_id=str(task.id),
        outcome=ExecutionOutcome.SUCCESS,
        attempts=1,
        duration_seconds=0.1,
        result={},
        final_status=TaskStatus.COMPLETED,
        verification={"status": "failed"},
    )
    experience = ExperienceExtractor().extract(task, result)

    assert experience is not None
    assert ExperienceExtractor.should_store(experience) is False


@pytest.mark.asyncio
async def test_consolidation_runs_via_integration() -> None:
    memory = MemoryService()
    integration = MemoryIntegration(memory_service=memory)
    task = _make_task()
    await integration.record_execution(task, _success_result(task))

    processed = await integration.consolidate_if_needed(limit=5)

    assert processed >= 0
