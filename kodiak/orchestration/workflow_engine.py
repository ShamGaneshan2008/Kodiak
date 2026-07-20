"""Async workflow engine for coordinated Kodiak task execution.

The engine owns workflow-level execution semantics: dependency resolution,
parallel scheduling, conditional gates, retries, recovery, cancellation,
resume, persistence, events, plugin hooks, and structured logs. It deliberately
does not perform repository retrieval or code generation itself; those remain
owned by Kodiak's existing agents and RAG subsystems.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)

NodeExecutor = Callable[["WorkflowContext", "WorkflowNode"], Awaitable[Any] | Any]
NodeCondition = Callable[["WorkflowContext", "WorkflowNode"], Awaitable[bool] | bool]


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowNodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class WorkflowNodeType(StrEnum):
    TASK = "task"
    CONDITION = "condition"
    PLUGIN = "plugin"
    AGENT = "agent"
    TOOL = "tool"
    RECOVERY = "recovery"


class WorkflowEventType(StrEnum):
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    WORKFLOW_RESUMED = "workflow.resumed"
    NODE_STARTED = "workflow.node.started"
    NODE_COMPLETED = "workflow.node.completed"
    NODE_FAILED = "workflow.node.failed"
    NODE_RETRYING = "workflow.node.retrying"
    NODE_SKIPPED = "workflow.node.skipped"
    NODE_TIMED_OUT = "workflow.node.timed_out"


class RetryPolicy(BaseModel):
    """Retry policy for one workflow node."""

    max_attempts: int = Field(1, ge=1)
    backoff_seconds: float = Field(0.0, ge=0.0)
    backoff_multiplier: float = Field(1.0, ge=1.0)

    def delay_for_attempt(self, retry_count: int) -> float:
        if retry_count <= 0:
            return 0.0
        return self.backoff_seconds * (self.backoff_multiplier ** (retry_count - 1))


class FailurePolicy(BaseModel):
    """Failure handling behavior for a node after retries are exhausted."""

    continue_on_failure: bool = False
    recovery_node_id: str | None = None


class WorkflowLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: str = "info"
    event: str
    workflow_id: str
    node_id: str | None = None
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvent(BaseModel):
    event_type: str
    workflow_id: str
    node_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class WorkflowNode(BaseModel):
    """Typed execution node in a Kodiak workflow graph."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    node_type: WorkflowNodeType = WorkflowNodeType.TASK
    executor: str
    dependencies: list[str] = Field(default_factory=list)
    status: WorkflowNodeStatus = WorkflowNodeStatus.PENDING
    retry_count: int = 0
    outputs: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_data: dict[str, Any] = Field(default_factory=dict)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)
    timeout_seconds: float | None = Field(None, gt=0)
    condition: str | None = Field(
        None,
        description="Named condition that must pass before this node may execute.",
    )
    branch_key: str | None = Field(
        None,
        description="Optional output key used by branch_nodes after completion.",
    )
    branch_nodes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Branch value to node ids that should remain active.",
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            WorkflowNodeStatus.COMPLETED,
            WorkflowNodeStatus.FAILED,
            WorkflowNodeStatus.SKIPPED,
            WorkflowNodeStatus.CANCELLED,
            WorkflowNodeStatus.TIMED_OUT,
        }

    def start(self) -> None:
        self.status = WorkflowNodeStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.finished_at = None
        self.error = None

    def complete(self, output: Any) -> None:
        self.status = WorkflowNodeStatus.COMPLETED
        self.outputs = {"value": output} if not isinstance(output, Mapping) else dict(output)
        self.finished_at = datetime.now(timezone.utc)
        self.execution_time_ms = self._duration_ms()

    def fail(self, error: str, status: WorkflowNodeStatus = WorkflowNodeStatus.FAILED) -> None:
        self.status = status
        self.error = error
        self.finished_at = datetime.now(timezone.utc)
        self.execution_time_ms = self._duration_ms()

    def skip(self, reason: str) -> None:
        self.status = WorkflowNodeStatus.SKIPPED
        self.error = reason
        self.finished_at = datetime.now(timezone.utc)
        self.execution_time_ms = self._duration_ms()

    def cancel(self) -> None:
        self.status = WorkflowNodeStatus.CANCELLED
        self.finished_at = datetime.now(timezone.utc)
        self.execution_time_ms = self._duration_ms()

    def reset_for_resume(self) -> None:
        if self.status == WorkflowNodeStatus.RUNNING:
            self.status = WorkflowNodeStatus.PENDING
            self.started_at = None

    def _duration_ms(self) -> int | None:
        if self.started_at and self.finished_at:
            return int((self.finished_at - self.started_at).total_seconds() * 1000)
        return None


class WorkflowState(BaseModel):
    """Serializable execution state for a workflow run."""

    workflow_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    nodes: list[WorkflowNode]
    metadata: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    logs: list[WorkflowLogEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    cancelled_at: datetime | None = None
    resume_count: int = 0

    @classmethod
    def from_execution_tasks(
        cls,
        *,
        name: str,
        tasks: Sequence[Any],
        description: str = "",
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkflowState":
        """Create a workflow from existing orchestration task objects.

        This adapter keeps the Workflow Engine aligned with TaskPlanner and
        scheduler-style models without requiring those modules to change.
        """
        task_ids = {str(getattr(task, "id")) for task in tasks}
        nodes = [
            WorkflowNode(
                id=str(getattr(task, "id")),
                name=str(getattr(task, "name")),
                description=str(getattr(task, "description", "")),
                node_type=WorkflowNodeType.AGENT,
                executor=str(getattr(task, "agent_type")),
                dependencies=[
                    str(dependency)
                    for dependency in getattr(task, "dependencies", [])
                    if str(dependency) in task_ids
                ],
                input_data=dict(getattr(task, "input_data", {}) or {}),
                metadata={
                    "source": "execution_task",
                    "plan_step_id": getattr(task, "plan_step_id", None),
                    "tool_names": list(getattr(task, "tool_names", []) or []),
                    **dict(getattr(task, "metadata", {}) or {}),
                },
            )
            for task in tasks
        ]
        return cls(
            workflow_id=workflow_id or uuid.uuid4().hex,
            name=name,
            description=description,
            nodes=nodes,
            metadata=metadata or {},
        )

    @field_validator("nodes")
    @classmethod
    def node_ids_are_unique(cls, nodes: list[WorkflowNode]) -> list[WorkflowNode]:
        ids = [node.id for node in nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("workflow node ids must be unique")
        return nodes

    @property
    def node_map(self) -> dict[str, WorkflowNode]:
        return {node.id: node for node in self.nodes}

    @property
    def progress_pct(self) -> float:
        if not self.nodes:
            return 100.0
        done = sum(1 for node in self.nodes if node.is_terminal)
        return round(done / len(self.nodes) * 100, 1)

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }

    def summary(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in WorkflowNodeStatus}
        for node in self.nodes:
            counts[node.status.value] += 1
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status.value,
            "progress_pct": self.progress_pct,
            "nodes": counts,
            "error": self.error,
        }


class WorkflowContext(BaseModel):
    """Context passed to executors and plugin hooks."""

    workflow: WorkflowState
    node_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    shared: dict[str, Any] = Field(default_factory=dict)

    def dependency_outputs(self, node: WorkflowNode) -> dict[str, dict[str, Any]]:
        return {
            dependency: self.node_outputs.get(dependency, {})
            for dependency in node.dependencies
        }


@runtime_checkable
class WorkflowStateStore(Protocol):
    async def save(self, state: WorkflowState) -> None:
        """Persist the latest workflow state."""

    async def load(self, workflow_id: str) -> WorkflowState | None:
        """Load persisted workflow state by id."""


@runtime_checkable
class WorkflowEventSink(Protocol):
    async def publish(self, event: Any) -> Any:
        """Publish a workflow event."""


@runtime_checkable
class WorkflowPluginHook(Protocol):
    async def execute(self, input_data: Any) -> Any:
        """Execute a plugin hook with workflow context."""


class InMemoryWorkflowStateStore:
    """Default state store for tests and single-process deployments."""

    def __init__(self) -> None:
        self._states: dict[str, WorkflowState] = {}
        self._lock = asyncio.Lock()

    async def save(self, state: WorkflowState) -> None:
        async with self._lock:
            self._states[state.workflow_id] = state.model_copy(deep=True)

    async def load(self, workflow_id: str) -> WorkflowState | None:
        async with self._lock:
            state = self._states.get(workflow_id)
            return state.model_copy(deep=True) if state is not None else None


class WorkflowEngine:
    """Execute typed workflow nodes with Kodiak orchestration semantics."""

    def __init__(
        self,
        *,
        executors: Mapping[str, NodeExecutor] | None = None,
        conditions: Mapping[str, NodeCondition] | None = None,
        state_store: WorkflowStateStore | None = None,
        event_sink: WorkflowEventSink | None = None,
        plugin_hooks: Sequence[WorkflowPluginHook] = (),
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")
        self._executors = dict(executors or {})
        self._conditions = dict(conditions or {})
        self._state_store = state_store or InMemoryWorkflowStateStore()
        self._event_sink = event_sink
        self._plugin_hooks = tuple(plugin_hooks)
        self._max_concurrency = max_concurrency
        self._cancelled: set[str] = set()

    def register_executor(self, name: str, executor: NodeExecutor) -> None:
        self._executors[name] = executor
        logger.info("workflow_executor_registered", executor=name)

    def register_condition(self, name: str, condition: NodeCondition) -> None:
        self._conditions[name] = condition
        logger.info("workflow_condition_registered", condition=name)

    async def run(
        self,
        workflow: WorkflowState,
        *,
        shared_context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Run a workflow from its current persisted state."""
        self._validate_workflow(workflow)
        self._cancelled.discard(workflow.workflow_id)
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = workflow.started_at or datetime.now(timezone.utc)
        context = WorkflowContext(
            workflow=workflow,
            node_outputs=self._completed_outputs(workflow),
            shared=shared_context or {},
        )

        await self._persist_and_emit(workflow, WorkflowEventType.WORKFLOW_STARTED)
        await self._run_hooks("workflow_started", context)

        try:
            while not workflow.is_terminal:
                if workflow.workflow_id in self._cancelled:
                    await self._cancel(workflow, context)
                    break

                ready = await self._ready_nodes(workflow, context)
                if not ready:
                    self._finish_if_possible(workflow)
                    if workflow.is_terminal:
                        break
                    self._fail_workflow(workflow, "No runnable nodes remain.")
                    break

                batch = ready[: self._max_concurrency]
                await asyncio.gather(
                    *(self._execute_node(workflow, context, node) for node in batch)
                )
                await self._persist(workflow)

            await self._finalize(workflow, context)
            return workflow
        except Exception as exc:
            self._fail_workflow(workflow, str(exc))
            await self._finalize(workflow, context)
            raise

    async def resume(
        self,
        workflow_id: str,
        *,
        shared_context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Resume a workflow from persisted state."""
        state = await self._state_store.load(workflow_id)
        if state is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        if state.status == WorkflowStatus.CANCELLED:
            raise ValueError(f"Workflow is cancelled and cannot be resumed: {workflow_id}")
        for node in state.nodes:
            node.reset_for_resume()
        state.resume_count += 1
        state.status = WorkflowStatus.RUNNING
        state.error = None
        await self._emit(state, WorkflowEventType.WORKFLOW_RESUMED)
        return await self.run(state, shared_context=shared_context)

    async def cancel(self, workflow_id: str) -> None:
        """Request cancellation. Running nodes finish their current await point."""
        self._cancelled.add(workflow_id)
        state = await self._state_store.load(workflow_id)
        if state is not None and not state.is_terminal:
            state.status = WorkflowStatus.CANCELLED
            state.cancelled_at = datetime.now(timezone.utc)
            for node in state.nodes:
                if node.status in {WorkflowNodeStatus.PENDING, WorkflowNodeStatus.RUNNING}:
                    node.cancel()
            await self._persist_and_emit(state, WorkflowEventType.WORKFLOW_CANCELLED)

    async def load_state(self, workflow_id: str) -> WorkflowState | None:
        return await self._state_store.load(workflow_id)

    async def _ready_nodes(
        self,
        workflow: WorkflowState,
        context: WorkflowContext,
    ) -> list[WorkflowNode]:
        nodes = workflow.node_map
        ready: list[WorkflowNode] = []
        for node in workflow.nodes:
            if node.status != WorkflowNodeStatus.PENDING:
                continue
            skipped_dependency = self._skipped_dependency(node, nodes)
            if skipped_dependency:
                node.skip(f"Dependency {skipped_dependency} was skipped or cancelled.")
                self._log(workflow, "workflow.node.skipped", node.id, node.error or "")
                await self._emit(workflow, WorkflowEventType.NODE_SKIPPED, node)
                continue
            if not self._dependencies_satisfied(node, nodes):
                continue
            if await self._should_skip(node, context):
                node.skip("Condition evaluated to false.")
                self._log(workflow, "workflow.node.skipped", node.id, node.error or "")
                await self._emit(workflow, WorkflowEventType.NODE_SKIPPED, node)
                continue
            ready.append(node)
        return ready

    async def _execute_node(
        self,
        workflow: WorkflowState,
        context: WorkflowContext,
        node: WorkflowNode,
    ) -> None:
        executor = self._executors.get(node.executor)
        if executor is None:
            await self._handle_node_failure(
                workflow,
                context,
                node,
                RuntimeError(f"Executor not registered: {node.executor}"),
            )
            return

        node.start()
        started = time.perf_counter()
        self._log(workflow, "workflow.node.started", node.id, "Node execution started.")
        await self._persist_and_emit(workflow, WorkflowEventType.NODE_STARTED, node)
        await self._run_hooks("node_started", context, node)

        try:
            result = executor(context, node)
            if inspect.isawaitable(result):
                if node.timeout_seconds is not None:
                    result = await asyncio.wait_for(result, timeout=node.timeout_seconds)
                else:
                    result = await result
            node.complete(result)
            context.node_outputs[node.id] = node.outputs
            self._apply_branching(workflow, node)
            self._log(
                workflow,
                "workflow.node.completed",
                node.id,
                "Node execution completed.",
                {"execution_time_ms": int((time.perf_counter() - started) * 1000)},
            )
            await self._persist_and_emit(workflow, WorkflowEventType.NODE_COMPLETED, node)
            await self._run_hooks("node_completed", context, node)
        except asyncio.TimeoutError as exc:
            await self._handle_node_failure(
                workflow,
                context,
                node,
                exc,
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001
            await self._handle_node_failure(workflow, context, node, exc)

    async def _handle_node_failure(
        self,
        workflow: WorkflowState,
        context: WorkflowContext,
        node: WorkflowNode,
        exc: BaseException,
        *,
        timed_out: bool = False,
    ) -> None:
        if node.retry_count < node.retry_policy.max_attempts - 1:
            node.retry_count += 1
            node.status = WorkflowNodeStatus.PENDING
            node.error = str(exc)
            delay = node.retry_policy.delay_for_attempt(node.retry_count)
            self._log(
                workflow,
                "workflow.node.retrying",
                node.id,
                f"Retrying node after failure: {exc}",
                {"retry_count": node.retry_count, "delay_seconds": delay},
            )
            await self._persist_and_emit(workflow, WorkflowEventType.NODE_RETRYING, node)
            if delay:
                await asyncio.sleep(delay)
            return

        status = WorkflowNodeStatus.TIMED_OUT if timed_out else WorkflowNodeStatus.FAILED
        node.fail(str(exc) or "Node execution failed.", status=status)
        event_type = (
            WorkflowEventType.NODE_TIMED_OUT
            if timed_out
            else WorkflowEventType.NODE_FAILED
        )
        self._log(workflow, event_type.value, node.id, node.error or "")
        await self._persist_and_emit(workflow, event_type, node)
        await self._run_hooks("node_failed", context, node)

        if node.failure_policy.recovery_node_id:
            recovery = workflow.node_map.get(node.failure_policy.recovery_node_id)
            if recovery and recovery.status == WorkflowNodeStatus.SKIPPED:
                recovery.status = WorkflowNodeStatus.PENDING
            return
        if not node.failure_policy.continue_on_failure:
            self._fail_workflow(workflow, node.error or "Workflow node failed.")

    def _dependencies_satisfied(
        self,
        node: WorkflowNode,
        nodes: Mapping[str, WorkflowNode],
    ) -> bool:
        for dependency_id in node.dependencies:
            dependency = nodes.get(dependency_id)
            if dependency is None:
                return False
            if dependency.status == WorkflowNodeStatus.COMPLETED:
                continue
            if dependency.status == WorkflowNodeStatus.FAILED:
                continue
            if dependency.status == WorkflowNodeStatus.TIMED_OUT:
                continue
            return False
        return True

    def _skipped_dependency(
        self,
        node: WorkflowNode,
        nodes: Mapping[str, WorkflowNode],
    ) -> str | None:
        for dependency_id in node.dependencies:
            dependency = nodes.get(dependency_id)
            if dependency is None:
                continue
            if dependency.status in {
                WorkflowNodeStatus.SKIPPED,
                WorkflowNodeStatus.CANCELLED,
            }:
                return dependency_id
        return None

    async def _should_skip(self, node: WorkflowNode, context: WorkflowContext) -> bool:
        if not node.condition:
            return False
        condition = self._conditions.get(node.condition)
        if condition is None:
            raise RuntimeError(f"Condition not registered: {node.condition}")
        result = condition(context, node)
        if inspect.isawaitable(result):
            result = await result
        return not bool(result)

    def _apply_branching(self, workflow: WorkflowState, node: WorkflowNode) -> None:
        if not node.branch_key or not node.branch_nodes:
            return
        selected = str(node.outputs.get(node.branch_key, ""))
        active = set(node.branch_nodes.get(selected, []))
        inactive = {
            node_id
            for branch, node_ids in node.branch_nodes.items()
            if branch != selected
            for node_id in node_ids
        }
        for node_id in inactive - active:
            candidate = workflow.node_map.get(node_id)
            if candidate and candidate.status == WorkflowNodeStatus.PENDING:
                candidate.skip(f"Branch '{selected}' selected.")
                self._log(workflow, "workflow.node.skipped", node_id, candidate.error or "")

    def _finish_if_possible(self, workflow: WorkflowState) -> None:
        if any(node.status == WorkflowNodeStatus.RUNNING for node in workflow.nodes):
            return
        if all(node.is_terminal for node in workflow.nodes):
            failed = [
                node
                for node in workflow.nodes
                if node.status in {WorkflowNodeStatus.FAILED, WorkflowNodeStatus.TIMED_OUT}
                and not node.failure_policy.continue_on_failure
                and not node.failure_policy.recovery_node_id
            ]
            if failed:
                self._fail_workflow(workflow, failed[0].error or "Workflow node failed.")
                return
            workflow.status = WorkflowStatus.COMPLETED
            workflow.finished_at = datetime.now(timezone.utc)
            workflow.outputs = {
                node.id: node.outputs
                for node in workflow.nodes
                if node.status == WorkflowNodeStatus.COMPLETED
            }

    async def _cancel(self, workflow: WorkflowState, context: WorkflowContext) -> None:
        workflow.status = WorkflowStatus.CANCELLED
        workflow.cancelled_at = datetime.now(timezone.utc)
        workflow.finished_at = workflow.cancelled_at
        for node in workflow.nodes:
            if node.status in {WorkflowNodeStatus.PENDING, WorkflowNodeStatus.RUNNING}:
                node.cancel()
        self._log(workflow, "workflow.cancelled", None, "Workflow cancelled.")
        await self._persist_and_emit(workflow, WorkflowEventType.WORKFLOW_CANCELLED)
        await self._run_hooks("workflow_cancelled", context)

    async def _finalize(self, workflow: WorkflowState, context: WorkflowContext) -> None:
        self._finish_if_possible(workflow)
        if workflow.status == WorkflowStatus.COMPLETED:
            await self._run_hooks("workflow_completed", context)
            await self._persist_and_emit(workflow, WorkflowEventType.WORKFLOW_COMPLETED)
        elif workflow.status == WorkflowStatus.FAILED:
            await self._run_hooks("workflow_failed", context)
            await self._persist_and_emit(workflow, WorkflowEventType.WORKFLOW_FAILED)
        elif workflow.status == WorkflowStatus.CANCELLED:
            await self._persist_and_emit(workflow, WorkflowEventType.WORKFLOW_CANCELLED)
        else:
            await self._persist(workflow)

    def _fail_workflow(self, workflow: WorkflowState, error: str) -> None:
        workflow.status = WorkflowStatus.FAILED
        workflow.error = error
        workflow.finished_at = datetime.now(timezone.utc)

    async def _run_hooks(
        self,
        hook_name: str,
        context: WorkflowContext,
        node: WorkflowNode | None = None,
    ) -> None:
        if not self._plugin_hooks:
            return
        payload = {
            "hook": hook_name,
            "workflow": context.workflow.summary(),
            "node": node.model_dump(mode="json") if node else None,
            "shared": context.shared,
        }
        for hook in self._plugin_hooks:
            try:
                await hook.execute(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "workflow_plugin_hook_failed",
                    hook=hook_name,
                    error=str(exc),
                )

    async def _persist_and_emit(
        self,
        workflow: WorkflowState,
        event_type: WorkflowEventType,
        node: WorkflowNode | None = None,
    ) -> None:
        await self._persist(workflow)
        await self._emit(workflow, event_type, node)

    async def _persist(self, workflow: WorkflowState) -> None:
        await self._state_store.save(workflow)

    async def _emit(
        self,
        workflow: WorkflowState,
        event_type: WorkflowEventType,
        node: WorkflowNode | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        event = WorkflowEvent(
            event_type=event_type.value,
            workflow_id=workflow.workflow_id,
            node_id=node.id if node else None,
            payload={
                "workflow": workflow.summary(),
                "node": node.model_dump(mode="json") if node else None,
            },
        )
        if hasattr(self._event_sink, "emit"):
            await self._event_sink.emit(event)  # type: ignore[attr-defined]
        else:
            await self._event_sink.publish(event)

    def _log(
        self,
        workflow: WorkflowState,
        event: str,
        node_id: str | None,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = WorkflowLogEntry(
            event=event,
            workflow_id=workflow.workflow_id,
            node_id=node_id,
            message=message,
            metadata=metadata or {},
        )
        workflow.logs.append(entry)
        logger.info(
            event.replace(".", "_"),
            workflow_id=workflow.workflow_id,
            node_id=node_id,
            message=message,
            **entry.metadata,
        )

    def _completed_outputs(self, workflow: WorkflowState) -> dict[str, dict[str, Any]]:
        return {
            node.id: node.outputs
            for node in workflow.nodes
            if node.status == WorkflowNodeStatus.COMPLETED
        }

    def _validate_workflow(self, workflow: WorkflowState) -> None:
        node_ids = set(workflow.node_map)
        for node in workflow.nodes:
            missing = set(node.dependencies) - node_ids
            if missing:
                raise ValueError(
                    f"Node {node.id} depends on unknown node(s): {sorted(missing)}"
                )
            if node.failure_policy.recovery_node_id:
                if node.failure_policy.recovery_node_id not in node_ids:
                    raise ValueError(
                        f"Node {node.id} references unknown recovery node "
                        f"{node.failure_policy.recovery_node_id}"
                    )
