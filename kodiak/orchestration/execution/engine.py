"""
kodiak/execution/engine.py

The Execution Engine sits between the Supervisor and the Agent Manager:

    Supervisor -> ExecutionEngine -> AgentManager -> ... -> ExecutionResult

It owns the runtime lifecycle of a single Task execution: attempt
sequencing, retries, per-attempt timeouts, cancellation, structured
logging, event hooks, progress reporting, and failure recovery. It knows
nothing about planning, agents, or tools — those belong to the Agent
Manager, injected here as a dependency.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections import defaultdict
from types import TracebackType
from typing import TYPE_CHECKING, Any

import structlog

from kodiak.db.models.task import Task, TaskStatus
from kodiak.orchestration.execution.exceptions import (
    ExecutionCancelledError,
    ExecutionTimeoutError,
    NonRetryableExecutionError,
)
from kodiak.orchestration.execution.interfaces import AgentManager, TaskRepository
from kodiak.orchestration.execution.models import (
    CancellationToken,
    ExecutionContext,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionHook,
    ExecutionOutcome,
    ExecutionResult,
    RetryPolicy,
    outcome_to_task_status,
)

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from kodiak.memory.integration import MemoryIntegration
    from kodiak.orchestration.reflection import ReflectionEngine
    from kodiak.orchestration.verification import VerificationEngine


class _NullSlot:
    """No-op async context manager used when no concurrency limit is configured."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class ExecutionEngine:
    """Drives a single Task from dispatch through terminal outcome.

    The engine is deliberately single-task-scoped: `execute()` runs one
    task end-to-end and returns its terminal `ExecutionResult`. Concurrency
    across tasks is the Supervisor's concern — it can invoke `execute()`
    many times concurrently (optionally bounded by `concurrency_limit`
    below), which keeps this class ready for a future parallel-execution
    mode with no change to its internal logic.

    Timeout granularity is per-attempt rather than per-task: each retry
    gets a fresh timeout budget, since a fresh attempt is a fresh chance to
    succeed within its own time box. A Supervisor wanting an overall
    wall-clock cap across all retries can layer `asyncio.timeout()` around
    its own call to `execute()`.

    Args:
        agent_manager: Executes the actual work for a task attempt.
        task_repository: Optional persistence port for lifecycle updates.
            If omitted, the engine still runs correctly but the caller is
            responsible for persisting the mutated `Task`/`ExecutionResult`.
        default_retry_policy: Fallback retry policy used when `execute()`
            isn't given an explicit override. If also omitted, a policy is
            derived from `task.max_retries`.
        default_timeout_seconds: Per-attempt timeout applied when
            `execute()` isn't given an explicit override.
        concurrency_limit: Optional cap on in-flight Agent Manager attempts
            across all `execute()` calls sharing this engine instance.
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        task_repository: TaskRepository | None = None,
        default_retry_policy: RetryPolicy | None = None,
        default_timeout_seconds: float = 600.0,
        concurrency_limit: int | None = None,
        verification_engine: VerificationEngine | None = None,
        reflection_engine: ReflectionEngine | None = None,
        memory_integration: MemoryIntegration | None = None,
    ) -> None:
        self._agent_manager = agent_manager
        self._task_repository = task_repository
        self._default_retry_policy = default_retry_policy
        self._default_timeout_seconds = default_timeout_seconds
        self._semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit else None
        self._hooks: dict[ExecutionEventType, list[ExecutionHook]] = defaultdict(list)
        self._active_tokens: dict[str, CancellationToken] = {}
        self._verification_engine = verification_engine
        self._reflection_engine = reflection_engine
        self._memory_integration = memory_integration

    def on(self, event_type: ExecutionEventType, hook: ExecutionHook) -> None:
        """Register an async hook invoked whenever `event_type` fires.

        Hook exceptions are logged and swallowed — a broken observer must
        never fail an execution.
        """
        self._hooks[event_type].append(hook)

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of an in-flight execution for `task_id`.

        Returns:
            True if a running execution was found and signalled, False if
            no execution for `task_id` is currently active on this engine
            instance.
        """
        token = self._active_tokens.get(task_id)
        if token is None:
            return False
        token.cancel()
        return True

    async def execute(
        self,
        task: Task,
        *,
        timeout_seconds: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> ExecutionResult:
        """Run `task` to completion, handling retries, timeouts, and cancellation.

        Args:
            task: The task to execute. Its `status`, `retry_count`,
                `result`, `error`, `total_tokens`, and `total_cost_usd`
                fields are mutated in place to reflect lifecycle progress.
            timeout_seconds: Per-attempt timeout override.
            retry_policy: Retry policy override. Defaults to
                `self._default_retry_policy`, then to a policy derived
                from `task.max_retries`.

        Returns:
            The terminal `ExecutionResult`. This method does not raise for
            ordinary execution failures (timeouts, retries exhausted,
            agent errors) — those are represented in the result's
            `outcome` and `error` fields. It only raises for programming
            errors such as a misbehaving hook that isn't already isolated,
            or `asyncio.CancelledError` if the *caller's* own task is
            cancelled externally (distinct from `self.cancel()`).
        """
        task_id = str(task.id)
        policy = retry_policy or self._default_retry_policy or RetryPolicy.from_task(task)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        token = CancellationToken()
        self._active_tokens[task_id] = token

        context = ExecutionContext(task=task, cancellation_token=token)
        log = logger.bind(task_id=task_id, correlation_id=context.correlation_id)
        started_at = time.monotonic()

        await self._set_status(task, TaskStatus.IN_PROGRESS, log)
        await self._emit(ExecutionEventType.TASK_STARTED, context, log, message="execution started")

        last_error_payload: dict[str, Any] | None = None

        try:
            for attempt in range(1, policy.max_attempts + 1):
                context.attempt = attempt
                attempt_log = log.bind(attempt=attempt, max_attempts=policy.max_attempts)

                if token.is_cancelled:
                    return await self._complete_execution(
                        task,
                        await self._finalize(
                            task,
                            context,
                            ExecutionOutcome.CANCELLED,
                            started_at,
                            attempt_log,
                            error=self._cancelled_payload("cancelled before attempt"),
                        ),
                    )

                await self._emit(ExecutionEventType.ATTEMPT_STARTED, context, attempt_log)

                try:
                    agent_result = await self._run_attempt(
                        context, timeout_seconds=timeout, token=token
                    )

                except ExecutionCancelledError:
                    attempt_log.warning("execution.cancelled")
                    return await self._complete_execution(
                        task,
                        await self._finalize(
                            task,
                            context,
                            ExecutionOutcome.CANCELLED,
                            started_at,
                            attempt_log,
                            error=self._cancelled_payload("cancelled during attempt"),
                        ),
                    )

                except ExecutionTimeoutError as exc:
                    last_error_payload = self._error_payload(exc)
                    attempt_log.warning("execution.attempt_timed_out", timeout_seconds=timeout)
                    await self._emit(
                        ExecutionEventType.TIMEOUT,
                        context,
                        attempt_log,
                        message=f"attempt {attempt} timed out after {timeout}s",
                    )
                    stop_result = await self._resolve_retry_decision(
                        task,
                        context,
                        policy,
                        attempt,
                        started_at,
                        attempt_log,
                        exhausted_outcome=ExecutionOutcome.TIMEOUT,
                        error_payload=last_error_payload,
                    )
                    if stop_result is not None:
                        return await self._complete_execution(task, stop_result)
                    continue

                except NonRetryableExecutionError as exc:
                    last_error_payload = self._error_payload(exc.cause)
                    last_error_payload["non_retryable"] = True
                    attempt_log.error("execution.non_retryable_failure", error=str(exc.cause))
                    await self._emit(
                        ExecutionEventType.ATTEMPT_FAILED,
                        context,
                        attempt_log,
                        message=str(exc.cause),
                    )
                    return await self._complete_execution(
                        task,
                        await self._finalize(
                            task,
                            context,
                            ExecutionOutcome.FAILURE,
                            started_at,
                            attempt_log,
                            error=last_error_payload,
                        ),
                    )

                except Exception as exc:  # noqa: BLE001 - classified via policy below
                    last_error_payload = self._error_payload(exc)
                    attempt_log.error("execution.attempt_failed", error=str(exc), exc_info=True)
                    await self._emit(
                        ExecutionEventType.ATTEMPT_FAILED,
                        context,
                        attempt_log,
                        message=str(exc),
                    )
                    stop_now = await self._maybe_stop_after_reflection(
                        task,
                        context,
                        policy,
                        attempt,
                        started_at,
                        attempt_log,
                        error_payload=last_error_payload,
                    )
                    if stop_now is not None:
                        return await self._complete_execution(task, stop_now)
                    if not policy.is_retryable(exc):
                        return await self._complete_execution(
                            task,
                            await self._finalize(
                                task,
                                context,
                                ExecutionOutcome.FAILURE,
                                started_at,
                                attempt_log,
                                error=last_error_payload,
                            ),
                        )
                    stop_result = await self._resolve_retry_decision(
                        task,
                        context,
                        policy,
                        attempt,
                        started_at,
                        attempt_log,
                        exhausted_outcome=ExecutionOutcome.RETRY_EXHAUSTED,
                        error_payload=last_error_payload,
                    )
                    if stop_result is not None:
                        return await self._complete_execution(task, stop_result)
                    continue

                else:
                    await self._emit(ExecutionEventType.ATTEMPT_SUCCEEDED, context, attempt_log)
                    execution_result = await self._finalize(
                        task,
                        context,
                        ExecutionOutcome.SUCCESS,
                        started_at,
                        attempt_log,
                        result=agent_result.output,
                        tokens_used=agent_result.tokens_used,
                        cost_usd=agent_result.cost_usd,
                    )
                    verified_result = await self._apply_verification(
                        task,
                        context,
                        execution_result,
                        attempt_log,
                        policy=policy,
                        attempt=attempt,
                    )
                    if verified_result is None:
                        continue
                    return await self._complete_execution(task, verified_result)

            # Loop exhausted without an explicit return: out of attempts.
            return await self._complete_execution(
                task,
                await self._finalize(
                    task,
                    context,
                    ExecutionOutcome.RETRY_EXHAUSTED,
                    started_at,
                    log,
                    error=last_error_payload,
                ),
            )
        finally:
            self._active_tokens.pop(task_id, None)

    async def _run_attempt(
        self,
        context: ExecutionContext,
        timeout_seconds: float,
        token: CancellationToken,
    ) -> Any:
        """Run one Agent Manager attempt, racing it against timeout and cancellation.

        Raises:
            ExecutionTimeoutError: If the attempt doesn't finish within `timeout_seconds`.
            ExecutionCancelledError: If `token` is cancelled before the attempt finishes.
            Exception: Whatever the Agent Manager itself raises.
        """
        slot = self._semaphore if self._semaphore is not None else _NullSlot()
        async with slot:
            agent_task = asyncio.ensure_future(self._agent_manager.run(context))
            cancel_wait = asyncio.ensure_future(token.wait())
            try:
                done, pending = await asyncio.wait(
                    {agent_task, cancel_wait},
                    timeout=timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    agent_task.cancel()
                    await asyncio.gather(agent_task, return_exceptions=True)
                    raise ExecutionTimeoutError(f"attempt exceeded {timeout_seconds}s timeout")

                if cancel_wait in done and agent_task not in done:
                    agent_task.cancel()
                    await asyncio.gather(agent_task, return_exceptions=True)
                    raise ExecutionCancelledError("execution cancelled during attempt")

                for future in pending:
                    future.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                return agent_task.result()
            finally:
                if not cancel_wait.done():
                    cancel_wait.cancel()

    async def _resolve_retry_decision(
        self,
        task: Task,
        context: ExecutionContext,
        policy: RetryPolicy,
        attempt: int,
        started_at: float,
        log: Any,
        *,
        exhausted_outcome: ExecutionOutcome,
        error_payload: dict[str, Any] | None,
    ) -> ExecutionResult | None:
        """Decide whether to retry, and finalize if not.

        Returns:
            None if the caller should proceed to the next attempt, or a
            finalized `ExecutionResult` if execution should stop (retries
            exhausted or cancellation arrived during backoff).
        """
        should_retry = await self._await_retry_backoff(context, policy, attempt, log)
        if should_retry:
            return None
        if context.cancellation_token.is_cancelled:
            return await self._finalize(
                task,
                context,
                ExecutionOutcome.CANCELLED,
                started_at,
                log,
                error=self._cancelled_payload("cancelled during retry backoff"),
            )
        return await self._finalize(
            task,
            context,
            exhausted_outcome,
            started_at,
            log,
            error=error_payload,
        )

    async def _await_retry_backoff(
        self,
        context: ExecutionContext,
        policy: RetryPolicy,
        attempt: int,
        log: Any,
    ) -> bool:
        """Wait out the backoff delay before the next attempt.

        Returns:
            True if the delay elapsed and another attempt should run,
            False if attempts are exhausted or cancellation arrived first.
        """
        if attempt >= policy.max_attempts or context.cancellation_token.is_cancelled:
            return False

        delay = policy.delay_for_attempt(attempt)
        await self._emit(
            ExecutionEventType.RETRY_SCHEDULED,
            context,
            log,
            message=f"retrying in {delay:.1f}s",
            data={"delay_seconds": delay},
        )
        log.info("execution.retry_scheduled", delay_seconds=round(delay, 2))

        try:
            await asyncio.wait_for(context.cancellation_token.wait(), timeout=delay)
        except TimeoutError:
            return True
        return False  # cancellation arrived during backoff

    async def _finalize(
        self,
        task: Task,
        context: ExecutionContext,
        outcome: ExecutionOutcome,
        started_at: float,
        log: Any,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        tokens_used: int = 0,
        cost_usd: float | None = None,
        reflection: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Persist terminal state, emit its event, and build the execution result."""
        duration = time.monotonic() - started_at
        final_status = outcome_to_task_status(outcome)

        task.retry_count = max(context.attempt - 1, 0)
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if tokens_used:
            task.total_tokens = (task.total_tokens or 0) + tokens_used
        if cost_usd is not None:
            task.total_cost_usd = (task.total_cost_usd or 0.0) + cost_usd

        await self._set_status(task, final_status, log)

        event_type = (
            ExecutionEventType.TASK_SUCCEEDED
            if outcome is ExecutionOutcome.SUCCESS
            else ExecutionEventType.TASK_FAILED
        )
        await self._emit(
            event_type,
            context,
            log,
            message=outcome.value,
            data={"duration_seconds": duration},
        )

        log.info(
            "execution.finished",
            outcome=outcome.value,
            attempts=context.attempt,
            duration_seconds=round(duration, 3),
        )

        return ExecutionResult(
            task_id=str(task.id),
            outcome=outcome,
            attempts=context.attempt,
            duration_seconds=duration,
            result=result or {},
            error=error,
            final_status=final_status,
            reflection=reflection,
        )

    async def _complete_execution(self, task: Task, result: ExecutionResult) -> ExecutionResult:
        """Record terminal execution in memory when configured."""
        if self._memory_integration is not None:
            try:
                await self._memory_integration.record_execution(task, result)
            except Exception:
                logger.warning(
                    "execution.memory_record_failed",
                    task_id=str(task.id),
                    exc_info=True,
                )
        return result

    async def _apply_verification(
        self,
        task: Task,
        context: ExecutionContext,
        execution_result: ExecutionResult,
        log: Any,
        *,
        policy: RetryPolicy,
        attempt: int,
    ) -> ExecutionResult | None:
        """Run verification and reflection; optionally request another attempt."""
        from kodiak.orchestration.verification import VerificationEngine, VerificationStatus

        if self._verification_engine is None or not VerificationEngine.should_verify(task):
            return execution_result

        verification = await self._verification_engine.verify(
            task,
            execution_result,
            execution_context=context,
        )
        verification_payload = verification.to_dict()
        log.info(
            "execution.verification_completed",
            verification_status=verification.status.value,
            retry_recommended=verification.retry_recommended,
        )

        from kodiak.orchestration.reflection import RepairStrategy

        if verification.status is not VerificationStatus.FAILED:
            merged_result = dict(execution_result.result)
            merged_result["verification"] = verification_payload
            task.result = merged_result
            return ExecutionResult(
                task_id=execution_result.task_id,
                outcome=execution_result.outcome,
                attempts=execution_result.attempts,
                duration_seconds=execution_result.duration_seconds + verification.duration_seconds,
                result=merged_result,
                error=execution_result.error,
                final_status=execution_result.final_status,
                verification=verification_payload,
            )

        error_payload = {
            "type": "VerificationFailed",
            "message": verification.summary or "Task verification failed.",
            "verification": verification_payload,
        }
        failed_result = ExecutionResult(
            task_id=execution_result.task_id,
            outcome=ExecutionOutcome.FAILURE,
            attempts=execution_result.attempts,
            duration_seconds=execution_result.duration_seconds + verification.duration_seconds,
            result=execution_result.result,
            error=error_payload,
            final_status=TaskStatus.FAILED,
            verification=verification_payload,
        )

        reflection_payload = await self._reflect_and_inject(
            task,
            context,
            failed_result,
            verification_result=verification,
            attempt=attempt,
            max_attempts=policy.max_attempts,
            log=log,
        )

        if (
            reflection_payload
            and reflection_payload.get("strategy") == RepairStrategy.RETRY.value
            and attempt < policy.max_attempts
        ):
            task.status = TaskStatus.IN_PROGRESS
            stop_result = await self._resolve_retry_decision(
                task,
                context,
                policy,
                attempt,
                time.monotonic(),
                log,
                exhausted_outcome=ExecutionOutcome.RETRY_EXHAUSTED,
                error_payload=error_payload,
            )
            if stop_result is None:
                return None
            stop_result.reflection = reflection_payload
            return stop_result

        task.error = error_payload
        task.result = execution_result.result
        await self._set_status(task, TaskStatus.FAILED, log)
        await self._emit(
            ExecutionEventType.TASK_FAILED,
            context,
            log,
            message="verification failed",
            data={"verification": verification_payload, "reflection": reflection_payload},
        )
        return ExecutionResult(
            task_id=execution_result.task_id,
            outcome=ExecutionOutcome.FAILURE,
            attempts=execution_result.attempts,
            duration_seconds=failed_result.duration_seconds,
            result=execution_result.result,
            error=error_payload,
            final_status=TaskStatus.FAILED,
            verification=verification_payload,
            reflection=reflection_payload,
        )

    async def _maybe_stop_after_reflection(
        self,
        task: Task,
        context: ExecutionContext,
        policy: RetryPolicy,
        attempt: int,
        started_at: float,
        log: Any,
        *,
        error_payload: dict[str, Any],
    ) -> ExecutionResult | None:
        if self._reflection_engine is None:
            return None

        from kodiak.orchestration.reflection import RepairStrategy

        failed_result = ExecutionResult(
            task_id=str(task.id),
            outcome=ExecutionOutcome.FAILURE,
            attempts=attempt,
            duration_seconds=time.monotonic() - started_at,
            error=error_payload,
            final_status=TaskStatus.FAILED,
        )
        reflection_payload = await self._reflect_and_inject(
            task,
            context,
            failed_result,
            attempt=attempt,
            max_attempts=policy.max_attempts,
            log=log,
        )
        if reflection_payload is None:
            return None
        if reflection_payload.get("strategy") == RepairStrategy.STOP.value:
            return await self._finalize(
                task,
                context,
                ExecutionOutcome.FAILURE,
                started_at,
                log,
                error=error_payload,
                reflection=reflection_payload,
            )
        return None

    async def _reflect_and_inject(
        self,
        task: Task,
        context: ExecutionContext,
        execution_result: ExecutionResult,
        *,
        verification_result: Any | None = None,
        attempt: int,
        max_attempts: int,
        log: Any,
    ) -> dict[str, Any] | None:
        if self._reflection_engine is None:
            return None

        from kodiak.orchestration.reflection import ReflectionEngine

        if not ReflectionEngine.should_reflect(task, execution_result):
            return None

        reflection = await self._reflection_engine.reflect(
            task,
            execution_result,
            execution_context=context,
            verification_result=verification_result,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        payload = reflection.to_dict()
        self._inject_reflection(task, payload)
        log.info(
            "execution.reflection_completed",
            reflection_outcome=payload.get("outcome"),
            strategy=payload.get("strategy"),
            category=payload.get("category"),
        )
        return payload

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

    async def _set_status(self, task: Task, status: TaskStatus, log: Any) -> None:
        """Update task status in memory and, if configured, persist the transition.

        Persistence failures are logged, not raised — a repository outage
        must not prevent the engine from reporting the correct in-memory
        outcome back to the Supervisor, which can retry the persistence
        step itself using the returned `ExecutionResult`.
        """
        task.status = status
        if self._task_repository is None:
            return
        try:
            await self._task_repository.update_status(str(task.id), status)
        except Exception:
            log.error("execution.status_persist_failed", status=status.value, exc_info=True)

    async def _emit(
        self,
        event_type: ExecutionEventType,
        context: ExecutionContext,
        log: Any,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        hooks = self._hooks.get(event_type)
        if not hooks:
            return
        event = ExecutionEvent(
            type=event_type,
            task_id=str(context.task.id),
            attempt=context.attempt,
            message=message,
            data=data or {},
        )
        for hook in hooks:
            try:
                await hook(event)
            except Exception:
                log.warning("execution.hook_failed", event_type=event_type.value, exc_info=True)

    @staticmethod
    def _error_payload(exc: BaseException) -> dict[str, Any]:
        return {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }

    @staticmethod
    def _cancelled_payload(message: str) -> dict[str, Any]:
        return {"type": "ExecutionCancelledError", "message": message}
