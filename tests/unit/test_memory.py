# tests/unit/test_memory.py
"""Comprehensive unit tests for the Kodiak Memory System."""

import uuid
from pathlib import Path

import pytest

from kodiak.memory.consolidation import ConsolidationStatus
from kodiak.memory.episodic import EpisodicMemory
from kodiak.memory.errors import (
    EpisodeNotFoundError,
    FactNotFoundError,
    MemoryNotFoundError,
    ProcedureNotFoundError,
    WorkingMemoryNotFoundError,
)
from kodiak.memory.long_term import LongTermMemory
from kodiak.memory.models import MemoryType
from kodiak.memory.procedural import ProceduralMemory, ProcedureStep
from kodiak.memory.ranking import MemoryRanker
from kodiak.memory.semantic import SemanticMemory
from kodiak.memory.service import MemoryService
from kodiak.memory.short_term import ShortTermMemory
from kodiak.memory.working import WorkingMemory, WorkingMemoryStatus


@pytest.mark.asyncio
async def test_working_memory_lifecycle():
    wm = WorkingMemory()
    task_id = uuid.uuid4()

    item = await wm.create_working_memory(task_id=task_id, goal="Fix authentication bug")
    assert item.task_id == task_id
    assert item.goal == "Fix authentication bug"
    assert item.status == WorkingMemoryStatus.ACTIVE

    await wm.update_scratchpad(task_id, "key1", "val1")
    await wm.append_to_scratchpad(task_id, "logs", "error 500")

    fetched = await wm.get_working_memory(task_id)
    assert fetched.scratchpad["key1"] == "val1"
    assert fetched.scratchpad["logs"] == ["error 500"]

    await wm.set_outcome(task_id, "Fixed by updating token header")
    completed = await wm.complete_working_memory(task_id)
    assert completed.status == WorkingMemoryStatus.COMPLETED
    assert completed.outcome == "Fixed by updating token header"

    memory_model = WorkingMemory.to_memory(completed)
    assert memory_model.type == MemoryType.WORKING
    assert memory_model.content == "Fix authentication bug"

    unconsolidated = await wm.get_unconsolidated_tasks()
    assert len(unconsolidated) == 1
    assert unconsolidated[0]["id"] == str(task_id)


@pytest.mark.asyncio
async def test_working_memory_not_found():
    wm = WorkingMemory()
    with pytest.raises(WorkingMemoryNotFoundError):
        await wm.get_working_memory(uuid.uuid4())


@pytest.mark.asyncio
async def test_short_term_memory():
    stm = ShortTermMemory(max_history_length=5)
    session_id = "sess_123"

    item1 = await stm.add_item(session_id, "Hello, assistant!", role="user")
    await stm.add_item(session_id, "How can I help you?", role="assistant")

    history = await stm.get_session_history(session_id)
    assert len(history) == 2
    assert history[0].content == "Hello, assistant!"
    assert history[1].content == "How can I help you?"

    mem = ShortTermMemory.to_memory(item1)
    assert mem.type == MemoryType.SHORT_TERM
    assert mem.tags == [session_id, "user"]

    cleared = await stm.clear_session(session_id)
    assert cleared is True
    history_after = await stm.get_session_history(session_id)
    assert len(history_after) == 0


@pytest.mark.asyncio
async def test_episodic_memory():
    ep_mem = EpisodicMemory()
    task_id = uuid.uuid4()

    episode = await ep_mem.create_episode(
        goal="Refactor DB schema",
        outcome="Successfully applied Alembic migration",
        task_id=task_id,
        steps=["Create migration script", "Run alembic upgrade head"],
    )
    assert episode.goal == "Refactor DB schema"

    fetched = await ep_mem.get_episode(episode.id)
    assert fetched.id == episode.id

    updated_ep = await ep_mem.update_significance(episode.id, 0.9)
    assert updated_ep.significance == 0.9

    search_res = await ep_mem.search_episodes("Alembic migration")
    assert len(search_res) > 0
    assert search_res[0].episode.id == episode.id

    deleted = await ep_mem.delete_episode(episode.id)
    assert deleted is True
    with pytest.raises(EpisodeNotFoundError):
        await ep_mem.get_episode(episode.id)


@pytest.mark.asyncio
async def test_semantic_memory():
    sem_mem = SemanticMemory()

    fact = await sem_mem.store_fact(
        content="PostgreSQL port is 5432",
        category="database",
        confidence=0.95,
    )
    assert fact.category == "database"

    fetched = await sem_mem.get_fact(fact.id)
    assert fetched.content == "PostgreSQL port is 5432"

    updated = await sem_mem.update_fact(fact.id, content="PostgreSQL default port is 5432")
    assert updated.content == "PostgreSQL default port is 5432"

    search_res = await sem_mem.search_facts("PostgreSQL", category="database")
    assert len(search_res) == 1
    assert search_res[0].entity.id == fact.id

    deleted = await sem_mem.delete_fact(fact.id)
    assert deleted is True
    with pytest.raises(FactNotFoundError):
        await sem_mem.get_fact(fact.id)


@pytest.mark.asyncio
async def test_procedural_memory():
    proc_mem = ProceduralMemory()

    steps = [
        ProcedureStep(step_number=1, action="git pull origin main"),
        ProcedureStep(step_number=2, action="pytest tests/"),
    ]
    procedure = await proc_mem.create_procedure(
        name="Deploy Validation",
        description="Steps to run before pushing code",
        steps=steps,
        tags=["git", "test"],
    )
    assert procedure.name == "Deploy Validation"
    assert procedure.success_rate == 0.0

    succ = await proc_mem.record_success(procedure.id)
    assert succ.success_count == 1
    assert succ.success_rate == 1.0

    fail = await proc_mem.record_failure(procedure.id)
    assert fail.failure_count == 1
    assert fail.success_rate == 0.5

    search_res = await proc_mem.search_procedures("Deploy")
    assert len(search_res) == 1

    deleted = await proc_mem.delete_procedure(procedure.id)
    assert deleted is True
    with pytest.raises(ProcedureNotFoundError):
        await proc_mem.get_procedure(procedure.id)


@pytest.mark.asyncio
async def test_long_term_memory():
    ltm = LongTermMemory()

    mem_sem = await ltm.add_memory(
        "Redis operates in memory", memory_type=MemoryType.SEMANTIC, tags=["redis"]
    )
    assert mem_sem.type == MemoryType.SEMANTIC

    mem_ep = await ltm.add_memory(
        "Optimize Redis queries",
        memory_type=MemoryType.EPISODIC,
        metadata={"outcome": "Reduced latency by 50ms"},
    )
    assert mem_ep.type == MemoryType.EPISODIC

    search_res = await ltm.search("Redis")
    assert len(search_res) >= 2

    listed = await ltm.list_memories(limit=10)
    assert len(listed) >= 2

    del_cnt = await ltm.delete_by_tags(["redis"])
    assert del_cnt >= 1


@pytest.mark.asyncio
async def test_ranking_and_retrieval():
    ranker = MemoryRanker(weight_relevance=0.6, weight_recency=0.2, weight_confidence=0.2)
    service = MemoryService(ranker=ranker)

    task_id = uuid.uuid4()
    await service.working.create_working_memory(task_id=task_id, goal="Fix docker networking")
    await service.short_term.add_item("sess_1", "docker network create my_net")
    await service.add(
        "Docker uses bridge networks by default", memory_type=MemoryType.SEMANTIC, tags=["docker"]
    )

    results = await service.retrieve(
        query="docker network",
        session_id="sess_1",
        task_id=task_id,
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].relevance_score > 0.0


@pytest.mark.asyncio
async def test_context_builder():
    service = MemoryService()
    task_id = uuid.uuid4()
    wm_item = await service.working.create_working_memory(task_id=task_id, goal="Build AST parser")
    await service.short_term.add_item("sess_ast", "Created parser module")
    await service.add("Tree-sitter provides AST nodes", memory_type=MemoryType.SEMANTIC)

    context = await service.build_context(
        query="AST parser",
        session_id="sess_ast",
        task_id=task_id,
        working_memory_item=wm_item,
        token_budget=1000,
    )

    assert "## Memory Context" in context
    assert "Active Task Working Memory" in context
    assert "Build AST parser" in context
    assert "Recent Interaction History" in context
    assert "Tree-sitter" in context


@pytest.mark.asyncio
async def test_memory_consolidation():
    service = MemoryService()
    task_id = uuid.uuid4()

    await service.working.create_working_memory(task_id=task_id, goal="Implement Celery worker")
    await service.working.update_scratchpad(task_id, "learnings", ["Celery requires Redis broker"])
    await service.working.set_outcome(task_id, "completed")
    await service.working.complete_working_memory(task_id)

    results = await service.consolidate(limit=10)
    assert len(results) == 1
    assert results[0].status == ConsolidationStatus.COMPLETED
    assert results[0].episodes_created == 1
    assert results[0].facts_stored == 1


@pytest.mark.asyncio
async def test_persistence_save_load(tmp_path: Path):
    file_path = tmp_path / "memory_state.json"
    service1 = MemoryService(persistence_path=file_path)

    task_id = uuid.uuid4()
    await service1.working.create_working_memory(task_id=task_id, goal="Persist memory test")
    await service1.add("SQLite supports WAL mode", memory_type=MemoryType.SEMANTIC, tags=["sqlite"])

    await service1.save_to_disk()
    assert file_path.exists()

    service2 = MemoryService(persistence_path=file_path)
    await service2.load_from_disk()

    wm_items = await service2.working.list_working_memories()
    assert len(wm_items) == 1
    assert wm_items[0].goal == "Persist memory test"

    facts = await service2.semantic.list_facts()
    assert len(facts) == 1
    assert facts[0].content == "SQLite supports WAL mode"


@pytest.mark.asyncio
async def test_service_cli_facade():
    service = MemoryService()

    mem = await service.add("Test memory content", memory_type=MemoryType.SEMANTIC, tags=["test"])
    assert mem.content == "Test memory content"

    search_res = await service.search("Test memory")
    assert len(search_res) == 1
    assert search_res[0].memory.id == mem.id

    listed = await service.list(tags=["test"])
    assert len(listed) == 1

    deleted = await service.delete(mem.id)
    assert deleted is True

    with pytest.raises(MemoryNotFoundError):
        await service.delete(mem.id)
