import uuid

import structlog

from kodiak.orchestration.state import ExecutionState, TaskStatus

logger = structlog.get_logger(__name__)


class ReflectionLoop:
    def __init__(self, state: ExecutionState, max_retries: int = 3) -> None:
        self._state = state
        self._max_retries = max_retries

    async def evaluate(self, task_id: uuid.UUID, output: str, error: str | None = None) -> bool:
        if error:
            logger.info("reflection_detected_failure", task_id=str(task_id), error=error)
            return False
        if not output or len(output.strip()) < 10:
            logger.info("reflection_detected_empty_output", task_id=str(task_id))
            return False
        return True

    async def should_retry(self, task_id: uuid.UUID) -> bool:
        task = self._state.get_task(task_id)
        if not task:
            return False
        if task.status == TaskStatus.COMPLETED:
            return False
        return task.retry_count < self._max_retries

    async def retry_action(
        self, task_id: uuid.UUID, new_strategy: str | None = None
    ) -> dict[str, str]:
        task = self._state.get_task(task_id)
        if not task:
            return {"status": "error", "message": "Task not found"}

        if not await self.should_retry(task_id):
            self._state.update_task(task_id, status=TaskStatus.FAILED, error="Max retries exceeded")
            logger.error("reflection_max_retries_exceeded", task_id=str(task_id))
            return {"status": "failed", "message": "Max retries exceeded"}

        self._state.update_task(
            task_id,
            status=TaskStatus.PENDING,
            retry_count=task.retry_count + 1,
            error=None,
        )

        strategy = new_strategy or "Retry with standard parameters"
        logger.info(
            "reflection_retrying_task",
            task_id=str(task_id),
            attempt=task.retry_count + 1,
            strategy=strategy,
        )
        return {"status": "retrying", "strategy": strategy}
