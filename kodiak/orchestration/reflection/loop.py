"""Self-repair loop coordinating reflection, retry, and replanning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from kodiak.db.models.task import Task, TaskStatus
from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult, RetryPolicy
from kodiak.orchestration.reflection.engine import ReflectionEngine
from kodiak.orchestration.reflection.models import ReflectionOutcome, RepairStrategy

if TYPE_CHECKING:
    from kodiak.orchestration.execution.engine import ExecutionEngine

logger = structlog.get_logger(__name__)


class SelfRepairLoop:
    """Coordinates execution, verification, reflection, retry, and replanning."""

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        reflection_engine: ReflectionEngine | None = None,
        planner: Any | None = None,
        *,
        max_cycles: int = 3,
    ) -> None:
        self._execution_engine = execution_engine
        self._reflection_engine = reflection_engine or ReflectionEngine()
        self._planner = planner
        self._max_cycles = max(1, max_cycles)
        self._logger = logger.bind(component="self_repair_loop")

    async def run(self, task: Task, **execute_kwargs: Any) -> ExecutionResult:
        """Run the self-repair loop until success, stop, or cycle limit."""
        policy = RetryPolicy.from_task(task)
        max_attempts = policy.max_attempts
        last_result: ExecutionResult | None = None

        for cycle in range(1, self._max_cycles + 1):
            result = await self._execution_engine.execute(task, **execute_kwargs)
            last_result = result

            if result.is_success and not ReflectionEngine.should_reflect(task, result):
                return result

            reflection = await self._reflection_engine.reflect(
                task,
                result,
                attempt=result.attempts,
                max_attempts=max_attempts,
            )
            result = self._attach_reflection(result, reflection.to_dict())

            if reflection.outcome is ReflectionOutcome.SUCCESS:
                return result

            if reflection.strategy is RepairStrategy.STOP:
                return result

            if reflection.strategy is RepairStrategy.REPLAN:
                replanned = await self._maybe_replan(task, result, reflection.to_dict())
                if replanned:
                    self._inject_reflection(task, reflection.to_dict())
                    continue
                return result

            if reflection.strategy is RepairStrategy.RETRY:
                if cycle >= self._max_cycles or result.attempts >= max_attempts:
                    return self._mark_max_retries(result, reflection.to_dict())
                self._inject_reflection(task, reflection.to_dict())
                task.status = TaskStatus.PENDING
                continue

            return result

        if last_result is not None:
            return last_result
        raise RuntimeError("Self-repair loop completed without producing a result.")

    async def _maybe_replan(
        self,
        task: Task,
        result: ExecutionResult,
        reflection_payload: dict[str, Any],
    ) -> bool:
        if self._planner is None:
            return False
        current_plan = task.context.get("execution_plan")
        if current_plan is None:
            return False
        try:
            replan_input = {
                "task_id": result.task_id,
                "error": result.error or {},
                "reflection": reflection_payload,
            }
            updated_plan = await self._planner.replan(current_plan, replan_input)
            task.context["execution_plan"] = updated_plan
            task.context["replanned"] = True
            self._logger.info("self_repair_replan_applied", task_id=result.task_id)
            return True
        except Exception:
            self._logger.exception("self_repair_replan_failed", task_id=result.task_id)
            return False

    @staticmethod
    def _inject_reflection(task: Task, reflection_payload: dict[str, Any]) -> None:
        history = list(task.context.get("reflection_history", []))
        history.append(reflection_payload)
        task.context["reflection_history"] = history
        task.context["reflection"] = reflection_payload
        corrections = task.context.setdefault("correction_context", {})
        corrections.update(
            {
                "root_cause": reflection_payload.get("root_cause"),
                "suggested_correction": reflection_payload.get("suggested_correction"),
                "category": reflection_payload.get("category"),
                "strategy": reflection_payload.get("strategy"),
            }
        )

    @staticmethod
    def _attach_reflection(
        result: ExecutionResult,
        reflection_payload: dict[str, Any],
    ) -> ExecutionResult:
        merged_result = dict(result.result)
        merged_result["reflection"] = reflection_payload
        return ExecutionResult(
            task_id=result.task_id,
            outcome=result.outcome,
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            result=merged_result,
            error=result.error,
            final_status=result.final_status,
            verification=result.verification,
            reflection=reflection_payload,
        )

    @staticmethod
    def _mark_max_retries(
        result: ExecutionResult,
        reflection_payload: dict[str, Any],
    ) -> ExecutionResult:
        reflection_payload = {
            **reflection_payload,
            "outcome": ReflectionOutcome.MAX_RETRIES_REACHED.value,
            "strategy": RepairStrategy.STOP.value,
        }
        return ExecutionResult(
            task_id=result.task_id,
            outcome=ExecutionOutcome.RETRY_EXHAUSTED,
            attempts=result.attempts,
            duration_seconds=result.duration_seconds,
            result={**result.result, "reflection": reflection_payload},
            error=result.error,
            final_status=TaskStatus.FAILED,
            verification=result.verification,
            reflection=reflection_payload,
        )


__all__ = ["SelfRepairLoop"]
