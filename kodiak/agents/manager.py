"""Canonical Agent Manager with Intelligent Selection for Kodiak.

This module owns agent registration, discovery, selection, and dispatch.
It implements the central decision layer between AgentManager and registered agents,
providing intelligent routing with capability-based, priority-based, and health-aware selection.

Architecture:

Supervisor
        │
        ▼
ExecutionEngine
        │
        ▼
AgentManager (this module)
        │
        ▼
Intelligent Selection Engine
        │
        ▼
Registered Agents
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog

from kodiak.agents import selector as agent_selector
from kodiak.agents.registry import (
    AgentAlreadyRegisteredError as RegistryAgentAlreadyRegisteredError,
)
from kodiak.agents.registry import (
    AgentNotFoundError as RegistryAgentNotFoundError,
)
from kodiak.agents.registry import AgentRegistry
from kodiak.config.metrics import (
    ACTIVE_AGENT_TASKS,
    AGENT_SELECTIONS_TOTAL,
    AGENT_TASK_DURATION_SECONDS,
    AGENT_TASKS_TOTAL,
)
from kodiak.db.models.task import Task, TaskPriority
from kodiak.orchestration.execution.interfaces import AgentManager as AgentManagerProtocol
from kodiak.orchestration.execution.models import AgentManagerResult, ExecutionContext

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
    "AgentSelectionStrategy",
    "AgentHealthStatus",
    "AgentScore",
    "SelectionContext",
    "SelectionResult",
    "AgentUnavailableError",
    "SelectionTimeoutError",
]


AgentHealthStatus = agent_selector.AgentHealthStatus
AgentSelectionStrategy = agent_selector.AgentSelectionStrategy
AgentScore = agent_selector.AgentScore
SelectionContext = agent_selector.SelectionContext
SelectionResult = agent_selector.SelectionResult


@runtime_checkable
class AgentTask(Protocol):
    """Minimal shape required of any task dispatched to an agent.

    Tasks may optionally expose ``required_capabilities`` (an iterable of
    strings) to influence agent selection.
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

    async def health_check(self) -> bool:
        """Check if the agent is healthy. Optional."""
        ...


class AgentManagerError(Exception):
    """Base exception for all agent manager errors."""


class AgentAlreadyRegisteredError(AgentManagerError):
    """Raised when registering an agent whose name is already taken."""


class AgentNotFoundError(AgentManagerError):
    """Raised when an operation references an unregistered agent."""


class NoSuitableAgentError(AgentManagerError):
    """Raised when no registered agent can handle a given task."""


class AgentUnavailableError(AgentManagerError):
    """Raised when an agent is temporarily unavailable."""


class SelectionTimeoutError(AgentManagerError):
    """Raised when agent selection takes too long."""


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    """Result of an agent execution attempt.

    Contains the output, token usage, and duration metrics from agent execution.
    """

    output: dict[str, Any]
    duration_seconds: float
    tokens_used: int = 0


@dataclass(frozen=True, slots=True)
class AgentExecutionMetrics:
    """Immutable snapshot of execution metrics for an agent."""

    total_executions: int
    successful_executions: int
    failed_executions: int
    total_duration_seconds: float

    @property
    def success_rate(self) -> float:
        """Fraction of executions that succeeded, or 0.0 with no executions."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions


@dataclass(slots=True)
class _AgentMetrics:
    """Mutable execution metrics tracked per registered agent."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Mean execution duration in seconds, or 0.0 with no executions."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions


@dataclass(slots=True)
class _AgentEntry:
    """Internal registry entry pairing an agent with its live metrics."""

    agent: Agent
    metrics: _AgentMetrics = field(default_factory=_AgentMetrics)
    in_flight: int = 0

    # Health and availability
    enabled: bool = True
    last_health_check: float = 0.0
    health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN


class AgentManager(AgentManagerProtocol):
    """Registers, selects, and dispatches agents on behalf of the ExecutionEngine.

    Implements the AgentManager protocol with intelligent selection engine.
    The manager is safe for concurrent use: registry mutations are
    serialized with an internal lock, and concurrent executions are bounded
    by a semaphore.

    Args:
        max_concurrent_executions: Upper bound on the number of agent
            executions that may run concurrently across all agents.
        clock: Callable returning the current monotonic time, injectable
            for testing.
        selection_timeout_seconds: Maximum time for agent selection.
    """

    def __init__(
        self,
        *,
        max_concurrent_executions: int = 10,
        clock: Callable[[], float] = time.monotonic,
        selection_timeout_seconds: float = 5.0,
        registry: AgentRegistry | None = None,
        selector: agent_selector.AgentSelector | None = None,
    ) -> None:
        self._entries: dict[str, _AgentEntry] = {}
        self._registry_lock = asyncio.Lock()
        self._registry = registry or AgentRegistry()
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._clock = clock
        self._logger = logger.bind(component="agent_manager")
        self._selector = selector or agent_selector.AgentSelector()
        self._selection_timeout_seconds = selection_timeout_seconds
        self._last_selection: agent_selector.SelectionResult | None = None

    async def register(self, agent: Agent) -> None:
        """Register an agent, making it available for selection and execution.

        Args:
            agent: The agent instance to register. Must expose a name that
                is unique among currently registered agents.

        Raises:
            AgentAlreadyRegisteredError: If an agent with the same name is
                already registered.
        """
        agent_name = self._agent_name(agent)
        agent_id = self._agent_id(agent)
        capabilities = self._agent_capabilities(agent)
        async with self._registry_lock:
            if agent_id in self._entries:
                self._logger.warning("agent_registration_conflict", agent_name=agent_id)
                raise AgentAlreadyRegisteredError(
                    f"Agent {agent_id!r} is already registered."
                )
            try:
                await self._registry.register(
                    agent_id,
                    instance=agent,
                    name=agent_name,
                    capabilities=sorted(capabilities),
                )
            except RegistryAgentAlreadyRegisteredError as exc:
                raise AgentAlreadyRegisteredError(str(exc)) from exc
            self._entries[agent_id] = _AgentEntry(agent=agent)
            self._logger.info(
                "agent_registered",
                agent_name=agent_id,
                capabilities=sorted(capabilities),
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
            await self._registry.unregister(agent_name)
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
        try:
            return await self._registry.get(agent_name)  # type: ignore[return-value]
        except RegistryAgentNotFoundError as exc:
            raise AgentNotFoundError(f"Agent {agent_name!r} is not registered.") from exc

    async def list_agents(self) -> list[Agent]:
        """List all currently registered agents.

        Returns:
            A snapshot list of registered agent instances.
        """
        async with self._registry_lock:
            names = sorted(self._entries)
        return [await self.get(name) for name in names]

    async def select_agent(self, task: AgentTask) -> Agent:
        """Select the best registered agent for a given task.

        Uses the intelligent selection engine with multi-criteria scoring.

        Args:
            task: The task to find an agent for.

        Returns:
            The selected agent instance.

        Raises:
            NoSuitableAgentError: If no registered agent can handle the task.
        """
        context = self._selection_context(task)
        candidates = await self._selection_candidates()

        if not candidates:
            raise NoSuitableAgentError(
                f"No agents are registered to handle task {task.task_id!r}."
            )

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._selector.select, context, candidates),
                timeout=self._selection_timeout_seconds,
            )
            agent_name = result.selected_agent_id
            self._last_selection = result

            self._logger.debug(
                "agent_selected",
                task_id=task.task_id,
                agent_name=agent_name,
                strategy=result.selection_strategy.value,
                score=result.selected_score,
                reason=result.reason,
            )
            AGENT_SELECTIONS_TOTAL.labels(
                strategy=result.selection_strategy.value,
                fallback="false",
            ).inc()

            return await self.get(agent_name)
        except TimeoutError:
            raise NoSuitableAgentError(
                f"Agent selection timed out for task {task.task_id!r}."
            ) from None
        except agent_selector.NoSuitableAgentError as exc:
            raise NoSuitableAgentError(str(exc)) from exc

    async def execute(
        self,
        task: AgentTask,
        *,
        agent_name: str | None = None,
        attempt: int = 1,
    ) -> AgentManagerResult:
        """Execute a task on a selected agent and report the outcome.

        This performs exactly one execution attempt. Retry decisions and
        backoff are owned by the caller (typically the ExecutionEngine).

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
        selected_agent_name = self._agent_id(agent)

        async with self._registry_lock:
            entry = self._entries[selected_agent_name]
            entry.in_flight += 1
            ACTIVE_AGENT_TASKS.labels(agent_type=selected_agent_name).inc()

        start = self._clock()
        log = self._logger.bind(
            task_id=task.task_id,
            agent_name=selected_agent_name,
            attempt=attempt,
        )
        log.info("agent_execution_started")

        try:
            async with self._execution_semaphore:
                try:
                    output = await agent.execute(task)
                except asyncio.CancelledError:
                    duration = self._clock() - start
                    await self._record_outcome(
                        selected_agent_name,
                        success=False,
                        duration=duration,
                    )
                    log.warning("agent_execution_cancelled", duration_seconds=duration)
                    raise
                except Exception as exc:  # noqa: BLE001 - outcome is reported, not swallowed
                    duration = self._clock() - start
                    await self._record_outcome(
                        selected_agent_name,
                        success=False,
                        duration=duration,
                    )
                    is_retryable = bool(getattr(exc, "retryable", True))
                    log.error(
                        "agent_execution_failed",
                        duration_seconds=duration,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        is_retryable=is_retryable,
                    )
                    if is_retryable:
                        raise
                    raise RuntimeError(str(exc)) from exc
                else:
                    duration = self._clock() - start
                    await self._record_outcome(selected_agent_name, success=True, duration=duration)
                    log.info("agent_execution_succeeded", duration_seconds=duration)
                    payload = output if isinstance(output, dict) else {"output": output}
                    return AgentManagerResult(
                        output=payload,
                        tokens_used=self._tokens_used(output),
                        cost_usd=getattr(output, "cost_usd", None),
                    )
        finally:
            async with self._registry_lock:
                entry.in_flight -= 1
                ACTIVE_AGENT_TASKS.labels(agent_type=selected_agent_name).dec()

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
                AGENT_TASKS_TOTAL.labels(agent_type=agent_name, status="success").inc()
            else:
                metrics.failed_executions += 1
                AGENT_TASKS_TOTAL.labels(agent_type=agent_name, status="failure").inc()
            AGENT_TASK_DURATION_SECONDS.labels(agent_type=agent_name).observe(duration)

    async def enable_agent(self, agent_name: str) -> bool:
        """Enable a previously disabled agent.

        Args:
            agent_name: Name of the agent to enable.

        Returns:
            True if agent was enabled, False if not found.

        Raises:
            AgentNotFoundError: If agent is not registered.
        """
        async with self._registry_lock:
            entry = self._entries.get(agent_name)
            if entry is None:
                raise AgentNotFoundError(f"Agent {agent_name!r} is not registered.")
            if entry.enabled:
                return False
            entry.enabled = True
            self._logger.info("agent_enabled", agent_name=agent_name)
            return True

    async def disable_agent(self, agent_name: str) -> bool:
        """Disable an agent from selection.

        Args:
            agent_name: Name of the agent to disable.

        Returns:
            True if agent was disabled, False if not found or already disabled.

        Raises:
            AgentNotFoundError: If agent is not registered.
        """
        async with self._registry_lock:
            entry = self._entries.get(agent_name)
            if entry is None:
                raise AgentNotFoundError(f"Agent {agent_name!r} is not registered.")
            if not entry.enabled:
                return False
            entry.enabled = False
            self._logger.info("agent_disabled", agent_name=agent_name)
            return True

    async def set_agent_health(self, agent_name: str, healthy: bool) -> None:
        """Manually set agent health status.

        Args:
            agent_name: Name of the agent.
            healthy: Whether the agent is healthy.

        Raises:
            AgentNotFoundError: If agent is not registered.
        """
        async with self._registry_lock:
            entry = self._entries.get(agent_name)
            if entry is None:
                raise AgentNotFoundError(f"Agent {agent_name!r} is not registered.")
            entry.health_status = (
                AgentHealthStatus.HEALTHY if healthy else AgentHealthStatus.UNHEALTHY
            )
            entry.last_health_check = self._clock()
            self._logger.info(
                "agent_health_updated",
                agent_name=agent_name,
                healthy=healthy,
            )

    async def health_check(self) -> dict[str, bool]:
        """Check the health of all registered agents.

        Agents that expose an async ``health_check() -> bool`` method are
        queried directly; agents that do not are assumed healthy.

        Returns:
            A mapping of agent name to health status.
        """
        async with self._registry_lock:
            entries = dict(self._entries)

        async def _check(name: str, entry: _AgentEntry) -> tuple[str, bool]:
            agent = entry.agent
            check = getattr(agent, "health_check", None)
            if check is None:
                return name, True
            try:
                healthy = bool(await check())
                async with self._registry_lock:
                    entry.health_status = (
                        AgentHealthStatus.HEALTHY if healthy else AgentHealthStatus.UNHEALTHY
                    )
                    entry.last_health_check = self._clock()
                return name, healthy
            except Exception as exc:  # noqa: BLE001 - health check failures are non-fatal
                async with self._registry_lock:
                    entry.health_status = AgentHealthStatus.UNHEALTHY
                    entry.last_health_check = self._clock()
                self._logger.warning("agent_health_check_failed", agent_name=name, error=str(exc))
                return name, False

        results = await asyncio.gather(
            *(_check(name, entry) for name, entry in entries.items())
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
                    "average_duration_seconds": entry.metrics.total_duration_seconds
                    / max(entry.metrics.total_executions, 1),
                    "in_flight": entry.in_flight,
                    "is_enabled": entry.enabled,
                    "health_status": entry.health_status.value,
                }
                for name, entry in self._entries.items()
            }

    async def get_selection_scores(
        self,
        task: AgentTask,
    ) -> list[agent_selector.AgentScore]:
        """Get ranking scores for all agents for a task (without executing).

        Useful for debugging and observability.

        Args:
            task: The task to evaluate.

        Returns:
            List of AgentScore objects for all registered agents.
        """
        context = self._selection_context(task)
        candidates = await self._selection_candidates()
        try:
            return list(self._selector.select(context, candidates).candidate_ranking)
        except agent_selector.NoSuitableAgentError:
            return [
                self._selector.score(context, candidate)
                for candidate in sorted(candidates, key=lambda item: item.agent_id)
            ]

    async def run(self, context: ExecutionContext) -> AgentManagerResult:
        """Execute one attempt for `context.task`, implementing the AgentManager protocol.

        This is the main entry point used by ExecutionEngine. It wraps the
        internal execute() method to implement the AgentManager protocol.

        Args:
            context: The execution context containing the task and cancellation token.

        Returns:
            An AgentManagerResult with the execution outcome.

        Raises:
            Exception: Any failure. Retryability is determined by the engine's
                RetryPolicy unless the implementation raises
                `NonRetryableExecutionError`.
        """
        class AgentTaskImpl:
            def __init__(self, task: Task, attempt: int) -> None:
                self.task_id = str(task.id)
                self.task_type = task.title
                self._task = task
                self.attempt = attempt
                context_caps = {}
                if isinstance(task.context, dict):
                    context_caps = task.context
                self.required_capabilities = frozenset(
                    context_caps.get("required_capabilities", ())
                )

            @property
            def priority(self) -> TaskPriority:
                return self._task.priority

            @property
            def allow_fallback(self) -> bool:
                return True

            @property
            def health_check_required(self) -> bool:
                return False

        task_impl = AgentTaskImpl(context.task, context.attempt)
        return await self.execute(task_impl, attempt=context.attempt)

    async def get_selection_explanation(self) -> agent_selector.SelectionResult | None:
        """Return the most recent selection decision, if one has been made."""
        return self._last_selection

    async def _selection_candidates(self) -> list[agent_selector.AgentCandidate]:
        """Build selector candidates from registry metadata and runtime metrics."""
        metadata_by_id = await self._registry.snapshot()
        async with self._registry_lock:
            entries = dict(self._entries)

        candidates: list[agent_selector.AgentCandidate] = []
        for agent_id, metadata in sorted(metadata_by_id.items()):
            entry = entries.get(agent_id)
            if entry is None:
                continue
            candidates.append(
                agent_selector.AgentCandidate(
                    agent_id=agent_id,
                    capabilities=frozenset(metadata.capabilities),
                    priority=self._agent_priority(entry.agent),
                    enabled=entry.enabled,
                    health_status=agent_selector.AgentHealthStatus(entry.health_status.value),
                    in_flight=entry.in_flight,
                    success_rate=(
                        entry.metrics.success_rate
                        if entry.metrics.total_executions
                        else None
                    ),
                    metadata={
                        "name": metadata.name,
                        "version": metadata.version,
                        "tags": metadata.tags,
                    },
                )
            )
        return candidates

    @staticmethod
    def _selection_context(task: AgentTask) -> agent_selector.SelectionContext:
        """Build selector context from the task-like object."""
        return agent_selector.SelectionContext(
            required_capabilities=frozenset(getattr(task, "required_capabilities", ()) or ()),
            priority=getattr(task, "priority", TaskPriority.MEDIUM),
            task_type=getattr(task, "task_type", None),
            task_id=getattr(task, "task_id", None),
            metadata=dict(getattr(task, "metadata", {}) or {}),
        )

    @staticmethod
    def _agent_name(agent: Agent) -> str:
        """Return a display name for an agent."""
        return str(
            getattr(
                agent,
                "name",
                getattr(getattr(agent, "role", None), "value", agent.__class__.__name__.lower()),
            )
        )

    @classmethod
    def _agent_id(cls, agent: Agent) -> str:
        """Return the stable registry id for an agent, adding one if needed."""
        agent_id = str(getattr(agent, "agent_id", cls._agent_name(agent)))
        if not hasattr(agent, "agent_id"):
            agent.agent_id = agent_id
        return agent_id

    @staticmethod
    def _agent_capabilities(agent: Agent) -> frozenset[str]:
        """Return normalized agent capabilities."""
        capabilities = getattr(agent, "capabilities", None)
        if capabilities is None:
            role = getattr(getattr(agent, "role", None), "value", None)
            capabilities = (role,) if role else ()
        return frozenset(str(capability) for capability in capabilities)

    @staticmethod
    def _agent_priority(agent: Agent) -> int:
        """Return a deterministic priority for ranking."""
        priority = getattr(agent, "priority", 0)
        try:
            return int(priority)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _tokens_used(output: Any) -> int:
        """Extract token usage from common agent output shapes."""
        if hasattr(output, "total_tokens"):
            return int(output.total_tokens)
        if isinstance(output, dict):
            token_usage = output.get("token_usage")
            if isinstance(token_usage, dict):
                return sum(int(value) for value in token_usage.values())
        return 0

    def get_statistics(self) -> dict[str, Any]:
        """Return aggregate statistics about all registered agents.

        Returns:
            A dictionary containing agent counts and capability overview.
        """
        return {
            "total_agents": len(self._entries),
            "enabled_agents": sum(1 for e in self._entries.values() if e.enabled),
            "capabilities": sorted(
                {cap for entry in self._entries.values() for cap in entry.agent.capabilities}
            ),
            "health_distribution": {},
        }

    async def get_detailed_statistics(self) -> dict[str, Any]:
        """Return detailed statistics about all registered agents.

        Returns:
            A dictionary with detailed agent statistics.
        """
        async with self._registry_lock:
            entries = dict(self._entries)

        health_dist: dict[str, int] = {}
        for entry in entries.values():
            status = entry.health_status.value
            health_dist[status] = health_dist.get(status, 0) + 1

        return {
            "total_agents": len(entries),
            "enabled_agents": sum(1 for e in entries.values() if e.enabled),
            "disabled_agents": sum(1 for e in entries.values() if not e.enabled),
            "capabilities": sorted(
                {cap for entry in entries.values() for cap in entry.agent.capabilities}
            ),
            "health_distribution": health_dist,
            "agents": {
                name: {
                    "enabled": entry.enabled,
                    "health_status": entry.health_status.value,
                    "in_flight": entry.in_flight,
                    "success_rate": entry.metrics.success_rate,
                    "total_executions": entry.metrics.total_executions,
                    "capabilities": sorted(entry.agent.capabilities),
                }
                for name, entry in entries.items()
            },
        }
