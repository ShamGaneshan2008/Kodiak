from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from kodiak.orchestration.workflow_engine import (
    FailurePolicy,
    InMemoryWorkflowStateStore,
    RetryPolicy,
    WorkflowContext,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowNode,
    WorkflowNodeStatus,
    WorkflowNodeType,
    WorkflowState,
    WorkflowStatus,
)


class ExecutionTask(BaseModel):
    id: Any = Field(default_factory=uuid4)
    name: str
    agent_type: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[Any] = Field(default_factory=list)


class EventSink:
    def __init__(self) -> None:
        self.events: list[WorkflowEvent] = []

    async def publish(self, event: WorkflowEvent) -> None:
        self.events.append(event)


class Hook:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, input_data: Any) -> None:
        self.calls.append(input_data)


@pytest.mark.asyncio
async def test_workflow_engine_runs_sequential_and_parallel_nodes() -> None:
    order: list[str] = []

    async def execute(_: WorkflowContext, node: WorkflowNode) -> dict[str, str]:
        order.append(node.id)
        await asyncio.sleep(0)
        return {"node": node.id}

    workflow = WorkflowState(
        name="build feature",
        nodes=[
            WorkflowNode(id="a", name="A", executor="exec"),
            WorkflowNode(id="b", name="B", executor="exec", dependencies=["a"]),
            WorkflowNode(id="c", name="C", executor="exec", dependencies=["a"]),
        ],
    )

    result = await WorkflowEngine(executors={"exec": execute}).run(workflow)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.progress_pct == 100.0
    assert order[0] == "a"
    assert set(order[1:]) == {"b", "c"}
    assert result.outputs["b"] == {"node": "b"}


@pytest.mark.asyncio
async def test_workflow_engine_supports_conditional_branching() -> None:
    async def decide(_: WorkflowContext, __: WorkflowNode) -> dict[str, str]:
        return {"branch": "left"}

    async def execute(_: WorkflowContext, node: WorkflowNode) -> dict[str, str]:
        return {"node": node.id}

    workflow = WorkflowState(
        name="branch",
        nodes=[
            WorkflowNode(
                id="decision",
                name="Decision",
                node_type=WorkflowNodeType.CONDITION,
                executor="decide",
                branch_key="branch",
                branch_nodes={"left": ["left"], "right": ["right"]},
            ),
            WorkflowNode(id="left", name="Left", executor="exec", dependencies=["decision"]),
            WorkflowNode(id="right", name="Right", executor="exec", dependencies=["decision"]),
            WorkflowNode(
                id="right-child",
                name="Right child",
                executor="exec",
                dependencies=["right"],
            ),
        ],
    )

    result = await WorkflowEngine(executors={"decide": decide, "exec": execute}).run(workflow)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.node_map["left"].status == WorkflowNodeStatus.COMPLETED
    assert result.node_map["right"].status == WorkflowNodeStatus.SKIPPED
    assert result.node_map["right-child"].status == WorkflowNodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_workflow_engine_retries_and_runs_recovery_node() -> None:
    attempts = 0

    async def flaky(_: WorkflowContext, __: WorkflowNode) -> dict[str, str]:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    async def recover(_: WorkflowContext, __: WorkflowNode) -> dict[str, str]:
        return {"recovered": "yes"}

    workflow = WorkflowState(
        name="recover",
        nodes=[
            WorkflowNode(
                id="flaky",
                name="Flaky",
                executor="flaky",
                retry_policy=RetryPolicy(max_attempts=2),
                failure_policy=FailurePolicy(recovery_node_id="recover"),
            ),
            WorkflowNode(
                id="recover",
                name="Recover",
                node_type=WorkflowNodeType.RECOVERY,
                executor="recover",
                dependencies=["flaky"],
            ),
        ],
    )

    result = await WorkflowEngine(executors={"flaky": flaky, "recover": recover}).run(workflow)

    assert result.status == WorkflowStatus.COMPLETED
    assert attempts == 2
    assert result.node_map["flaky"].status == WorkflowNodeStatus.FAILED
    assert result.node_map["recover"].status == WorkflowNodeStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_engine_handles_timeout_and_failure() -> None:
    async def slow(_: WorkflowContext, __: WorkflowNode) -> None:
        await asyncio.sleep(0.05)

    workflow = WorkflowState(
        name="timeout",
        nodes=[
            WorkflowNode(
                id="slow",
                name="Slow",
                executor="slow",
                timeout_seconds=0.001,
            )
        ],
    )

    result = await WorkflowEngine(executors={"slow": slow}).run(workflow)

    assert result.status == WorkflowStatus.FAILED
    assert result.node_map["slow"].status == WorkflowNodeStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_workflow_engine_persists_events_hooks_and_resumes() -> None:
    store = InMemoryWorkflowStateStore()
    sink = EventSink()
    hook = Hook()

    async def execute(_: WorkflowContext, node: WorkflowNode) -> dict[str, str]:
        return {"node": node.id}

    workflow = WorkflowState(
        name="resume",
        nodes=[
            WorkflowNode(
                id="done",
                name="Done",
                executor="exec",
                status=WorkflowNodeStatus.COMPLETED,
            ),
            WorkflowNode(id="pending", name="Pending", executor="exec", dependencies=["done"]),
        ],
    )
    await store.save(workflow)

    result = await WorkflowEngine(
        executors={"exec": execute},
        state_store=store,
        event_sink=sink,
        plugin_hooks=[hook],
    ).resume(workflow.workflow_id)

    persisted = await store.load(workflow.workflow_id)
    assert result.status == WorkflowStatus.COMPLETED
    assert persisted is not None
    assert persisted.status == WorkflowStatus.COMPLETED
    assert result.resume_count == 1
    assert sink.events
    assert hook.calls
    assert result.logs


@pytest.mark.asyncio
async def test_workflow_engine_cancels_pending_workflow() -> None:
    started = asyncio.Event()

    async def slow(_: WorkflowContext, __: WorkflowNode) -> None:
        started.set()
        await asyncio.sleep(0.05)

    store = InMemoryWorkflowStateStore()
    engine = WorkflowEngine(executors={"slow": slow}, state_store=store)
    workflow = WorkflowState(
        name="cancel",
        nodes=[
            WorkflowNode(id="first", name="First", executor="slow"),
            WorkflowNode(id="second", name="Second", executor="slow", dependencies=["first"]),
        ],
    )
    task = asyncio.create_task(engine.run(workflow))
    await started.wait()
    await engine.cancel(workflow.workflow_id)
    result = await task

    assert result.status == WorkflowStatus.CANCELLED
    assert result.node_map["second"].status == WorkflowNodeStatus.CANCELLED


def test_workflow_state_can_be_created_from_execution_tasks() -> None:
    first = ExecutionTask(name="retrieve_context", agent_type="research")
    second = ExecutionTask(
        name="write_code",
        agent_type="coder",
        dependencies=[first.id],
        input_data={"task": "implement"},
    )

    workflow = WorkflowState.from_execution_tasks(
        name="planned workflow",
        tasks=[first, second],
    )

    assert workflow.node_map[str(first.id)].executor == "research"
    assert workflow.node_map[str(second.id)].dependencies == [str(first.id)]
    assert workflow.node_map[str(second.id)].input_data == {"task": "implement"}
