"""Autonomous task execution loop connecting Kodiak orchestration subsystems."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from kodiak.agents.manager import AgentManager
from kodiak.db.models.task import Task, TaskPriority
from kodiak.db.models.task import TaskStatus as DbTaskStatus
from kodiak.memory.episodic import EpisodicMemory
from kodiak.memory.models import MemoryType
from kodiak.memory.service import MemoryService
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.models import (
    CancellationToken,
    ExecutionOutcome,
    ExecutionResult,
    RetryPolicy,
)
from kodiak.orchestration.reflection import ReflectionEngine, ReflectionResult, RepairStrategy
from kodiak.orchestration.state import TaskState, TaskStatus, transition_task_status
from kodiak.orchestration.task_planner import ExecutableTask, ExecutionPlan, TaskPlanner
from kodiak.orchestration.tool_router import ToolRouter
from kodiak.orchestration.verification import (
    VerificationEngine,
    VerificationResult,
    VerificationStatus,
)

logger = structlog.get_logger(__name__)

_AGENT_CAPABILITIES: dict[str, list[str]] = {
    "coder": ["code_generation", "file_editing"],
    "tester": ["test_execution"],
    "reviewer": ["code_review"],
    "research": ["research", "context_retrieval"],
    "retrieval": ["context_retrieval", "research"],
    "planner": ["planning"],
    "debugger": ["debugging"],
}


@dataclass(slots=True)
class AutonomousLoopResult:
    task_state: TaskState
    plan: ExecutionPlan | None
    execution_result: ExecutionResult | None
    verification_result: VerificationResult | None
    reflection_results: list[ReflectionResult] = field(default_factory=list)
    selected_agent: str | None = None
    attempts: int = 0
    replans: int = 0
    elapsed_seconds: float = 0.0
    memory_stored: bool = False

    @property
    def success(self) -> bool:
        return self.task_state.status is TaskStatus.COMPLETED


class AutonomousTaskLoop:
    """Coordinate planning, memory, execution, verification, and reflection."""

    def __init__(
        self,
        *,
        task_planner: TaskPlanner,
        memory_service: MemoryService,
        agent_manager: AgentManager,
        execution_engine: ExecutionEngine,
        verifier: VerificationEngine | None = None,
        reflection: ReflectionEngine | None = None,
        tool_router: ToolRouter | None = None,
        max_loop_attempts: int = 3,
        max_replans: int = 2,
        memory_recall_limit: int = 5,
    ) -> None:
        self._planner = task_planner
        self._memory = memory_service
        self._agent_manager = agent_manager
        self._engine = execution_engine
        self._verifier = verifier or VerificationEngine()
        self._reflection = reflection or ReflectionEngine()
        self._tool_router = tool_router
        self._max_loop_attempts = max_loop_attempts
        self._max_replans = max_replans
        self._memory_recall_limit = memory_recall_limit
        self._cancellation_token = CancellationToken()
        self._active_task_state: TaskState | None = None

    def cancel(self) -> None:
        """Request cooperative cancellation of the active loop."""
        self._cancellation_token.cancel()
        if self._active_task_state is not None:
            self._engine.cancel(self._active_task_state.task_id)

    async def run(
        self,
        goal: str,
        *,
        workspace: str | Path | None = None,
        title: str | None = None,
        task_state: TaskState | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> AutonomousLoopResult:
        started = time.monotonic()
        state = task_state or TaskState(
            title=title or goal[:80],
            objective=goal,
            max_retries=self._max_loop_attempts,
        )
        self._active_task_state = state
        if not self._cancellation_token.is_cancelled:
            self._cancellation_token = CancellationToken()

        plan: ExecutionPlan | None = None
        execution_result: ExecutionResult | None = None
        verification_result: VerificationResult | None = None
        reflection_results: list[ReflectionResult] = []
        selected_agent: str | None = None
        replans = 0
        loop_attempt = 0
        memory_context: dict[str, Any] = {}

        log = logger.bind(task_id=state.task_id, execution_id=state.run_id)
        log.info("autonomous_loop_started", goal=goal)

        try:
            while loop_attempt < self._max_loop_attempts:
                loop_attempt += 1
                if self._cancellation_token.is_cancelled:
                    self._set_status(state, TaskStatus.CANCELLED)
                    break

                if plan is None:
                    memory_context = await self._recall_memory(goal, state)
                    plan = await self._plan(
                        goal, workspace, memory_context, extra_context, reflection_results
                    )
                    selected_agent = self._primary_agent_type(plan)

                execution_result = await self._execute_plan(
                    goal=goal,
                    plan=plan,
                    state=state,
                    workspace=workspace,
                    memory_context=memory_context,
                    loop_attempt=loop_attempt,
                )
                selected_agent = (
                    str((execution_result.result or {}).get("agent", selected_agent or ""))
                    or selected_agent
                )

                self._set_status(state, TaskStatus.VERIFYING)
                verification_result = await self._verifier.verify(
                    goal=goal,
                    plan=plan,
                    execution_result=execution_result,
                    task_state=state,
                )
                log.info(
                    "autonomous_loop_verification",
                    status=verification_result.status.value,
                    attempt=loop_attempt,
                )

                if verification_result.status is VerificationStatus.VERIFIED:
                    state.result = str((execution_result.result or {}).get("summary", goal))
                    self._set_status(state, TaskStatus.COMPLETED)
                    break

                self._set_status(state, TaskStatus.REFLECTING)
                reflection = await self._reflection.reflect(
                    self._build_reflection_task(goal=goal, state=state, workspace=workspace),
                    execution_result,
                    verification_result=verification_result,
                    attempt=loop_attempt,
                    max_attempts=self._max_loop_attempts,
                )
                reflection_results.append(reflection)
                log.info(
                    "autonomous_loop_reflection",
                    action=reflection.strategy.value,
                    root_cause=reflection.root_cause,
                )

                if reflection.strategy is RepairStrategy.STOP:
                    state.error = reflection.root_cause
                    self._set_status(state, TaskStatus.FAILED)
                    break

                if reflection.strategy is RepairStrategy.REPLAN:
                    replans += 1
                    state.retry_count += 1
                    self._set_status(state, TaskStatus.REPLANNING)
                    extra_context = {
                        **(extra_context or {}),
                        "reflection_evidence": reflection.evidence,
                        "previous_failure": reflection.root_cause,
                    }
                    plan = None
                    continue

                if reflection.strategy is RepairStrategy.REPAIR:
                    self._set_status(state, TaskStatus.REPAIRING)
                    execution_result = await self._execute_repair(
                        goal=goal,
                        state=state,
                        reflection=reflection,
                        workspace=workspace,
                        memory_context=memory_context,
                    )
                    selected_agent = str(
                        (execution_result.result or {}).get("agent", selected_agent)
                    )
                    self._set_status(state, TaskStatus.VERIFYING)
                    verification_result = await self._verifier.verify(
                        goal=goal,
                        plan=plan,
                        execution_result=execution_result,
                        task_state=state,
                    )
                    if verification_result.status is VerificationStatus.VERIFIED:
                        state.result = str((execution_result.result or {}).get("summary", goal))
                        self._set_status(state, TaskStatus.COMPLETED)
                        break
                    state.retry_count += 1
                    continue

                state.retry_count += 1
                self._set_status(state, TaskStatus.RUNNING)

            else:
                state.error = "Autonomous loop attempt budget exhausted."
                self._set_status(state, TaskStatus.FAILED)

            memory_stored = await self._store_experience(
                goal=goal,
                state=state,
                plan=plan,
                execution_result=execution_result,
                verification_result=verification_result,
            )

            elapsed = time.monotonic() - started
            log.info(
                "autonomous_loop_finished",
                status=state.status.value,
                attempts=loop_attempt,
                elapsed_seconds=round(elapsed, 3),
            )
            return AutonomousLoopResult(
                task_state=state,
                plan=plan,
                execution_result=execution_result,
                verification_result=verification_result,
                reflection_results=reflection_results,
                selected_agent=selected_agent,
                attempts=loop_attempt,
                replans=replans,
                elapsed_seconds=elapsed,
                memory_stored=memory_stored,
            )
        finally:
            self._active_task_state = None

    async def _recall_memory(self, goal: str, state: TaskState) -> dict[str, Any]:
        results = await self._memory.retrieve(
            query=goal,
            task_id=uuid.UUID(state.task_id),
            memory_types=[
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
                MemoryType.PROCEDURAL,
            ],
            limit=self._memory_recall_limit,
        )
        recalled = [
            {
                "id": str(item.memory.id),
                "type": item.memory.type.value,
                "title": item.memory.title,
                "content": item.memory.content,
                "score": item.relevance_score,
            }
            for item in results
        ]
        state.set_memory("recalled_memories", recalled)
        logger.info("autonomous_loop_memory_recalled", count=len(recalled), task_id=state.task_id)
        return {"memories": recalled}

    async def _plan(
        self,
        goal: str,
        workspace: str | Path | None,
        memory_context: dict[str, Any],
        extra_context: dict[str, Any] | None,
        reflection_results: list[ReflectionResult],
    ) -> ExecutionPlan:
        state = self._active_task_state
        if state is not None:
            self._set_status(state, TaskStatus.PLANNING)

        context: dict[str, Any] = {
            "work_dir": str(workspace) if workspace is not None else None,
            **memory_context,
            **(extra_context or {}),
        }
        if reflection_results:
            context["reflection_evidence"] = [
                reflection.to_dict() for reflection in reflection_results
            ]

        plan = await self._planner.plan_execution(goal, context)
        if state is not None:
            state.metadata["plan"] = plan.machine_readable()
            self._set_status(state, TaskStatus.RUNNING)
        return plan

    async def _execute_plan(
        self,
        *,
        goal: str,
        plan: ExecutionPlan,
        state: TaskState,
        workspace: str | Path | None,
        memory_context: dict[str, Any],
        loop_attempt: int,
    ) -> ExecutionResult:
        self._set_status(state, TaskStatus.RUNNING)
        last_result: ExecutionResult | None = None

        for executable in self._ordered_tasks(plan):
            if self._cancellation_token.is_cancelled:
                return ExecutionResult(
                    task_id=state.task_id,
                    outcome=ExecutionOutcome.CANCELLED,
                    attempts=loop_attempt,
                    duration_seconds=0.0,
                    error={"type": "ExecutionCancelledError", "message": "Loop cancelled."},
                    final_status=DbTaskStatus.CANCELLED,
                )

            db_task = self._build_db_task(
                goal=goal,
                executable=executable,
                plan=plan,
                state=state,
                workspace=workspace,
                memory_context=memory_context,
                loop_attempt=loop_attempt,
            )
            last_result = await self._engine.execute(
                db_task,
                retry_policy=RetryPolicy(max_attempts=1),
            )
            state.metadata["last_execution"] = _serialize_execution_result(last_result)
            if not last_result.is_success:
                return last_result
            explicit_status = str((last_result.result or {}).get("verification_status", "")).lower()
            if explicit_status in {
                VerificationStatus.FAILED.value,
                VerificationStatus.INCONCLUSIVE.value,
            }:
                return last_result

        if last_result is None:
            return ExecutionResult(
                task_id=state.task_id,
                outcome=ExecutionOutcome.FAILURE,
                attempts=loop_attempt,
                duration_seconds=0.0,
                error={"type": "PlanningError", "message": "Planner produced no executable tasks."},
                final_status=DbTaskStatus.FAILED,
            )
        return last_result

    async def _execute_repair(
        self,
        *,
        goal: str,
        state: TaskState,
        reflection: ReflectionResult,
        workspace: str | Path | None,
        memory_context: dict[str, Any],
    ) -> ExecutionResult:
        repair_task = ExecutableTask(
            name="repair_execution",
            agent_type="coder",
            input_data={
                "task": goal,
                "mode": "repair",
                "root_cause": reflection.root_cause,
            },
            tool_names=["coder"],
        )
        db_task = self._build_db_task(
            goal=goal,
            executable=repair_task,
            plan=None,
            state=state,
            workspace=workspace,
            memory_context=memory_context,
            loop_attempt=state.retry_count + 1,
            repair=True,
        )
        return await self._engine.execute(db_task, retry_policy=RetryPolicy(max_attempts=1))

    async def _store_experience(
        self,
        *,
        goal: str,
        state: TaskState,
        plan: ExecutionPlan | None,
        execution_result: ExecutionResult | None,
        verification_result: VerificationResult | None,
    ) -> bool:
        outcome = (
            "success"
            if state.status is TaskStatus.COMPLETED
            else "failed"
            if state.status is TaskStatus.FAILED
            else state.status.value
        )
        steps = [task.name for task in plan.tasks] if plan else []
        context = {
            "task_id": state.task_id,
            "verification": (verification_result.to_dict() if verification_result else None),
            "execution": (
                _serialize_execution_result(execution_result) if execution_result else None
            ),
            "reflection_count": len(state.reflections),
        }
        context = {key: value for key, value in context.items() if value is not None}

        episodic = self._memory.episodic
        if isinstance(episodic, EpisodicMemory):
            await episodic.create_episode(
                goal=goal,
                outcome=outcome,
                task_id=uuid.UUID(state.task_id),
                context=context,
                steps=steps,
            )
            logger.info("autonomous_loop_memory_stored", task_id=state.task_id, outcome=outcome)
            return True
        return False

    @staticmethod
    def _build_reflection_task(
        *,
        goal: str,
        state: TaskState,
        workspace: str | Path | None,
    ) -> Task:
        """Build the workflow-level task context required by reflection."""
        return Task(
            id=str(uuid.uuid4()),
            repository_id=str(uuid.uuid4()),
            title=state.title,
            description=goal,
            status=DbTaskStatus.IN_PROGRESS,
            priority=TaskPriority.MEDIUM,
            max_retries=state.max_retries,
            context={
                "goal": goal,
                "workspace": str(workspace) if workspace is not None else None,
                "orchestration_task_id": state.task_id,
            },
        )

    def _build_db_task(
        self,
        *,
        goal: str,
        executable: ExecutableTask,
        plan: ExecutionPlan | None,
        state: TaskState,
        workspace: str | Path | None,
        memory_context: dict[str, Any],
        loop_attempt: int,
        repair: bool = False,
    ) -> Task:
        capabilities = _AGENT_CAPABILITIES.get(executable.agent_type, [executable.agent_type])
        context: dict[str, Any] = {
            "goal": goal,
            "workspace": str(workspace) if workspace is not None else None,
            "agent_type": executable.agent_type,
            "required_capabilities": capabilities,
            "tool_names": executable.tool_names,
            "memory_context": memory_context,
            "loop_attempt": loop_attempt,
            "orchestration_task_id": state.task_id,
            "execution_id": state.run_id,
            "repair": repair,
        }
        if plan is not None:
            context["plan"] = plan.machine_readable()
        if self._tool_router is not None:
            context["tool_router_available"] = True

        return Task(
            id=str(uuid.uuid4()),
            repository_id=str(uuid.uuid4()),
            title=executable.name,
            description=executable.description or goal,
            status=DbTaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            max_retries=0,
            context=context,
            plan=plan.machine_readable() if plan else {},
        )

    @staticmethod
    def _ordered_tasks(plan: ExecutionPlan) -> list[ExecutableTask]:
        by_id = {task.id: task for task in plan.tasks}
        if plan.execution_order:
            return [by_id[task_id] for task_id in plan.execution_order if task_id in by_id]
        return list(plan.tasks)

    @staticmethod
    def _primary_agent_type(plan: ExecutionPlan) -> str | None:
        ordered = AutonomousTaskLoop._ordered_tasks(plan)
        return ordered[-1].agent_type if ordered else None

    @staticmethod
    def _set_status(state: TaskState, new_status: TaskStatus) -> None:
        if state.status is new_status:
            return
        transition_task_status(state.status, new_status)
        state.status = new_status
        if new_status is TaskStatus.RUNNING and state.started_at is None:
            state.started_at = state.started_at or state.created_at
        if new_status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            from datetime import UTC, datetime

            state.finished_at = datetime.now(UTC)


def _serialize_execution_result(result: ExecutionResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "outcome": result.outcome.value,
        "attempts": result.attempts,
        "duration_seconds": result.duration_seconds,
        "result": result.result,
        "error": result.error,
        "final_status": result.final_status.value,
    }


def build_autonomous_loop(
    *,
    agent_manager: AgentManager | None = None,
    agents: list[Any] | None = None,
    tool_router: ToolRouter | None = None,
    max_loop_attempts: int = 3,
) -> AutonomousTaskLoop:
    """Construct a ready-to-run autonomous loop with default dependencies.

    Callers that pass ``agents`` must register them on the returned loop's
    manager before invoking ``run()`` (for example in an async test fixture).
    """
    manager = agent_manager or AgentManager()
    engine = ExecutionEngine(manager, default_timeout_seconds=30.0)
    loop = AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=engine,
        tool_router=tool_router,
        max_loop_attempts=max_loop_attempts,
    )
    loop._pending_agents = agents or []
    return loop


async def initialize_autonomous_loop(loop: AutonomousTaskLoop) -> AutonomousTaskLoop:
    """Register any agents deferred during ``build_autonomous_loop``."""
    pending = getattr(loop, "_pending_agents", None) or []
    for agent in pending:
        await loop._agent_manager.register(agent)
    loop._pending_agents = []
    return loop
