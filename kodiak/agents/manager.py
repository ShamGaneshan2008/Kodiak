
"""Canonical Agent Manager for Kodiak.

This module owns agent registration, discovery, selection, and dispatch.
It depends only on the :class:`Agent` protocol defined here and never
imports concrete agent implementations or infrastructure directly.

Retry policy is intentionally not implemented in this module. The
ExecutionEngine owns retry decisions and backoff; the manager performs a
single execution attempt per call to :meth:`AgentManager.execute` and
reports whether the outcome is retryable via
:attr:`AgentManagerResult.is_retryable`, leaving the decision to re-invoke
entirely to the caller.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "Agent",
    "AgentTask",
    "AgentManager",
    "AgentManagerResult",
    "AgentManagerError",
    "AgentAlreadyRegisteredError",
    "AgentNotFoundError",
    "NoSuitableAgentError",
]


@runtime_checkable
class AgentTask(Protocol):
    """Minimal shape required of any task dispatched to an agent.

    Tasks may optionally expose ``required_capabilities`` (an iterable of
    strings) to influence agent selection in :meth:`AgentManager.select_agent`.
    """

    task_id: str
    task_type: str


@runtime_checkable
class Agent(Protocol):
    """Protocol that all agents registered with the manager must satisfy.

    The manager depends only on this protocol, never on concrete agent
    classes or infrastructure.
    """

    name: str
    capabilities: frozenset[str]

    async def execute(self, task: AgentTask) -> Any:
        """Execute a task and return its result."""
        ...


class AgentManagerError(Exception):
    """Base exception for all agent manager errors."""


class AgentAlreadyRegisteredError(AgentManagerError):
    """Raised when registering an agent whose name is already taken."""


class AgentNotFoundError(AgentManagerError):
    """Raised when an operation references an unregistered agent."""


class NoSuitableAgentError(AgentManagerError):
    """Raised when no registered agent can handle a given task."""


@dataclass(frozen=True, slots=True)
class AgentManagerResult:
    """Outcome of a single agent execution attempt.

    Attributes:
        success: Whether the agent completed the task without error.
        agent_name: Name of the agent that handled the task.
        task_id: Identifier of the task that was executed.
        output: The value returned by the agent on success, if any.
        error: The exception raised by the agent on failure, if any.
        is_retryable: Whether the caller should consider retrying this
            task on failure. Ignored when ``success`` is ``True``.
        duration_seconds: Wall-clock time spent executing the agent.
        attempt: The attempt number this result corresponds to, as
            supplied by the caller.
    """

    success: bool
    agent_name: str
    task_id: str
    output: Any | None = None
    error: BaseException | None = None
    is_retryable: bool = True
    duration_seconds: float = 0.0
    attempt: int = 1


@dataclass(slots=True)
class _AgentMetrics:
    """Mutable execution metrics tracked per registered agent."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration_seconds: float = 0.0

    @property
    def average_duration_seconds(self) -> float:
        """Mean execution duration in seconds, or 0.0 with no executions."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_seconds / self.total_executions

    @property
    def success_rate(self) -> float:
        """Fraction of executions that succeeded, or 0.0 with no executions."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions


@dataclass(slots=True)
class _AgentEntry:
    """Internal registry entry pairing an agent with its live metrics."""

    agent: Agent
    metrics: _AgentMetrics = field(default_factory=_AgentMetrics)
    in_flight: int = 0


class AgentManager:
    """Registers, selects, and dispatches agents on behalf of callers.

    The manager is safe for concurrent use: registry mutations are
    serialized with an internal lock, and concurrent executions are bounded
    by a semaphore.

    Args:
        max_concurrent_executions: Upper bound on the number of agent
            executions that may run concurrently across all agents.
        clock: Callable returning the current monotonic time, injectable
            for testing.
    """

    def __init__(
        self,
        *,
        max_concurrent_executions: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entries: dict[str, _AgentEntry] = {}
        self._registry_lock = asyncio.Lock()
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._clock = clock
        self._logger = logger.bind(component="agent_manager")

    async def register(self, agent: Agent) -> None:
        """Register an agent, making it available for selection and execution.

        Args:
            agent: The agent instance to register. Must expose a name that
                is unique among currently registered agents.

        Raises:
            AgentAlreadyRegisteredError: If an agent with the same name is
                already registered.
        """
        async with self._registry_lock:
            if agent.name in self._entries:
                self._logger.warning("agent_registration_conflict", agent_name=agent.name)
                raise AgentAlreadyRegisteredError(
                    f"Agent {agent.name!r} is already registered."
                )
            self._entries[agent.name] = _AgentEntry(agent=agent)
            self._logger.info(
                "agent_registered",
                agent_name=agent.name,
                capabilities=sorted(agent.capabilities),
            )

    async def unregister(self, agent_name: str) -> None:
        """Unregister a previously registered agent.

        Args:
            agent_name: Name of the agent to remove.

        Raises:
            AgentNotFoundError: If no agent with that name is registered.
        """
        async with self._registry_lock:
            if agent_name not in self._entries:
                raise AgentNotFoundError(f"Agent {agent_name!r} is not registered.")
            del self._entries[agent_name]
            self._logger.info("agent_unregistered", agent_name=agent_name)

    async def get(self, agent_name: str) -> Agent:
        """Retrieve a registered agent by name.

        Args:
            agent_name: Name of the agent to retrieve.

        Returns:
            The registered agent instance.

        Raises:
            AgentNotFoundError: If no agent with that name is registered.
        """
        async with self._registry_lock:
            entry = self._entries.get(agent_name)
            if entry is None:
                raise AgentNotFoundError(f"Agent {agent_name!r} is not registered.")
            return entry.agent

    async def list_agents(self) -> list[Agent]:
        """List all currently registered agents.

        Returns:
            A snapshot list of registered agent instances.
        """
        async with self._registry_lock:
            return [entry.agent for entry in self._entries.values()]

    async def select_agent(self, task: AgentTask) -> Agent:
        """Select the best registered agent for a given task.

        Agents are scored by the overlap between their declared
        capabilities and the task's ``required_capabilities`` (if any),
        with ties broken in favor of the least-loaded agent.

        Args:
            task: The task to find an agent for.

        Returns:
            The selected agent instance.

        Raises:
            NoSuitableAgentError: If no registered agent can handle the task.
        """
        required_capabilities = frozenset(getattr(task, "required_capabilities", ()) or ())

        async with self._registry_lock:
            candidates = list(self._entries.items())

        if not candidates:
            raise NoSuitableAgentError(
                f"No agents are registered to handle task {task.task_id!r}."
            )

        scored: list[tuple[int, int, Agent]] = []
        for _name, entry in candidates:
            overlap = len(entry.agent.capabilities & required_capabilities)
            if required_capabilities and overlap == 0:
                continue
            scored.append((overlap, -entry.in_flight, entry.agent))

        if not scored:
            raise NoSuitableAgentError(
                f"No registered agent declares the capabilities required by "
                f"task {task.task_id!r}: {sorted(required_capabilities)}."
            )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_agent = scored[0][2]
        self._logger.debug(
            "agent_selected",
            task_id=task.task_id,
            agent_name=selected_agent.name,
            required_capabilities=sorted(required_capabilities),
        )
        return selected_agent

    async def execute(
        self,
        task: AgentTask,
        *,
        agent_name: str | None = None,
        attempt: int = 1,
    ) -> AgentManagerResult:
        """Execute a task on a selected agent and report the outcome.

        This performs exactly one execution attempt. Retry decisions and
        backoff are owned by the caller (typically the ExecutionEngine),
        which should inspect ``is_retryable`` on the returned result and,
        if appropriate, call this method again with an incremented
        ``attempt``.

        Args:
            task: The task to execute.
            agent_name: Optional name of a specific agent to use, bypassing
                selection. If omitted, the best agent is chosen via
                :meth:`select_agent`.
            attempt: The attempt number, supplied by the caller for
                tracking and logging purposes.

        Returns:
            An :class:`AgentManagerResult` describing the outcome.

        Raises:
            AgentNotFoundError: If ``agent_name`` is given but not registered.
            NoSuitableAgentError: If no agent can be selected for the task.
        """
        agent = await self.get(agent_name) if agent_name else await self.select_agent(task)

        async with self._registry_lock:
            entry = self._entries[agent.name]
            entry.in_flight += 1

        start = self._clock()
        log = self._logger.bind(task_id=task.task_id, agent_name=agent.name, attempt=attempt)
        log.info("agent_execution_started")

        try:
            async with self._execution_semaphore:
                try:
                    output = await agent.execute(task)
                except asyncio.CancelledError:
                    duration = self._clock() - start
                    await self._record_outcome(agent.name, success=False, duration=duration)
                    log.warning("agent_execution_cancelled", duration_seconds=duration)
                    raise
                except Exception as exc:  # noqa: BLE001 - outcome is reported, not swallowed
                    duration = self._clock() - start
                    await self._record_outcome(agent.name, success=False, duration=duration)
                    is_retryable = bool(getattr(exc, "retryable", True))
                    log.error(
                        "agent_execution_failed",
                        duration_seconds=duration,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        is_retryable=is_retryable,
                    )
                    return AgentManagerResult(
                        success=False,
                        agent_name=agent.name,
                        task_id=task.task_id,
                        error=exc,
                        is_retryable=is_retryable,
                        duration_seconds=duration,
                        attempt=attempt,
                    )
                else:
                    duration = self._clock() - start
                    await self._record_outcome(agent.name, success=True, duration=duration)
                    log.info("agent_execution_succeeded", duration_seconds=duration)
                    return AgentManagerResult(
                        success=True,
                        agent_name=agent.name,
                        task_id=task.task_id,
                        output=output,
                        is_retryable=False,
                        duration_seconds=duration,
                        attempt=attempt,
                    )
        finally:
            async with self._registry_lock:
                entry.in_flight -= 1

    async def _record_outcome(self, agent_name: str, *, success: bool, duration: float) -> None:
        """Update stored execution metrics for an agent following an attempt.

        Args:
            agent_name: Name of the agent the outcome belongs to.
            success: Whether the execution attempt succeeded.
            duration: Wall-clock duration of the attempt, in seconds.
        """
        async with self._registry_lock:
            metrics = self._entries[agent_name].metrics
            metrics.total_executions += 1
            metrics.total_duration_seconds += duration
            if success:
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1

    async def health_check(self) -> dict[str, bool]:
        """Check the health of all registered agents.

        Agents that expose an async ``health_check() -> bool`` method are
        queried directly; agents that do not are assumed healthy.

        Returns:
            A mapping of agent name to health status.
        """
        async with self._registry_lock:
            entries = dict(self._entries)

        async def _check(name: str, agent: Agent) -> tuple[str, bool]:
            check = getattr(agent, "health_check", None)
            if check is None:
                return name, True
            try:
                healthy = bool(await check())
            except Exception as exc:  # noqa: BLE001 - health check failures are non-fatal
                self._logger.warning("agent_health_check_failed", agent_name=name, error=str(exc))
                return name, False
            return name, healthy

        results = await asyncio.gather(
            *(_check(name, entry.agent) for name, entry in entries.items())
        )
        return dict(results)

    async def metrics(self) -> dict[str, dict[str, float | int]]:
        """Return a snapshot of execution metrics for all registered agents.

        Returns:
            A mapping of agent name to a dictionary containing
            ``total_executions``, ``successful_executions``,
            ``failed_executions``, ``success_rate``,
            ``average_duration_seconds``, and ``in_flight``.
        """
        async with self._registry_lock:
            return {
                name: {
                    "total_executions": entry.metrics.total_executions,
                    "successful_executions": entry.metrics.successful_executions,
                    "failed_executions": entry.metrics.failed_executions,
                    "success_rate": entry.metrics.success_rate,
                    "average_duration_seconds": entry.metrics.average_duration_seconds,
                    "in_flight": entry.in_flight,
                }
                for name, entry in self._entries.items()
            }
