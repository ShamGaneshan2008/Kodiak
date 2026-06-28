import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field

from kodiak.orchestration.state import ExecutionState, TaskState, TaskStatus

logger = structlog.get_logger(__name__)


class ScheduledTask(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    agent_type: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0)
    dependencies: list[uuid.UUID] = Field(default_factory=list)
    max_retries: int = Field(default=2, ge=0)
    retry_count: int = 0


class TaskScheduler:
    def __init__(self, state: ExecutionState) -> None:
        self._state = state
        self._queue: list[ScheduledTask] = []
        self._lock = asyncio.Lock()

    async def add_task(self, task: ScheduledTask) -> None:
        async with self._lock:
            self._queue.append(task)
            self._queue.sort(key=lambda t: t.priority, reverse=True)
            self._state.tasks.append(
                TaskState(
                    id=task.id,
                    name=task.name,
                    agent_type=task.agent_type,
                    dependencies=task.dependencies,
                )
            )
        logger.info("task_scheduled", task_id=str(task.id), name=task.name)

    async def get_next_task(self) -> ScheduledTask | None:
        async with self._lock:
            available: list[ScheduledTask] = []
            for task in self._queue:
                task_state = self._state.get_task(task.id)
                if not task_state or task_state.status != TaskStatus.PENDING:
                    continue
                deps_met = all(
                    self._state.get_task(dep)
                    and self._state.get_task(dep).status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_met:
                    available.append(task)

            if not available:
                return None

            next_task = available[0]
            self._queue.remove(next_task)
            self._state.update_task(next_task.id, status=TaskStatus.RUNNING)
            self._state.current_task_id = next_task.id
            return next_task

    async def complete_task(self, task_id: uuid.UUID) -> None:
        self._state.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )
        logger.info("task_completed", task_id=str(task_id))

    async def fail_task(self, task_id: uuid.UUID, error: str) -> None:
        task_state = self._state.get_task(task_id)
        if not task_state:
            return

        task = next((t for t in self._queue if t.id == task_id), None)
        if task and task_state.retry_count < task.max_retries:
            self._state.update_task(
                task_id,
                status=TaskStatus.PENDING,
                retry_count=task_state.retry_count + 1,
                error=error,
            )
            task.retry_count += 1
            self._queue.append(task)
            logger.info("task_retrying", task_id=str(task_id), attempt=task.retry_count)
        else:
            self._state.update_task(task_id, status=TaskStatus.FAILED, error=error)
            logger.error("task_failed", task_id=str(task_id), error=error)