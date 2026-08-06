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
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog

from kodiak.config.metrics import (
    AGENT_SELECTIONS_TOTAL,
    AGENT_TASK_DURATION_SECONDS,
    AGENT_TASKS_TOTAL,
    ACTIVE_AGENT_TASKS,
)
from kodiak.db.models.task import TaskPriority
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


class AgentHealthStatus(StrEnum):
    """Health status of an agent."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AgentSelectionStrategy(StrEnum):
    """Strategy for selecting an agent."""

    CAPABILITY_BASED = "capability_based"
    WEIGHTED_RANKING = "weighted_ranking"
    HEALTH_AWARE = "health_aware"
    FALLBACK = "fallback"
    DETERMINISTIC = "deterministic"


@dataclass(frozen=True)
class AgentScore:
    """Score calculated for an agent during selection."""

    agent_name: str
    total_score: float
    capability_score: float
    health_score: float
    load_score: float
    priority_score: float
    confidence_score: float
    is_available: bool
    health_status: AgentHealthStatus
    raw_metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionContext:
    """Context passed to the selection engine."""

    required_capabilities: frozenset[str]
    priority: TaskPriority
    attempt: int
    fallback_allowed: bool
    health_check_required: bool
    task_id: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    """Result of agent selection."""

    selected_agent_name: str
    is_fallback: bool
    scores: tuple[AgentScore, ...]
    selection_strategy: AgentSelectionStrategy


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


class AgentSelectionEngine:
    """Intelligent agent selection engine with multiple strategies.

    Provides capability-based routing, priority-based routing, confidence
    scoring, weighted ranking, fallback routing, and health-aware routing.

    Args:
        default_timeout_seconds: Maximum time for selection before timeout.
    """

    def __init__(self, default_timeout_seconds: float = 5.0) -> None:
        self._default_timeout_seconds = default_timeout_seconds
        self._logger = logger.bind(component="selection_engine")

    async def select(
        self,
        candidates: list[tuple[str, _AgentEntry]],
        context: SelectionContext,
        *,
        timeout_seconds: float | None = None,
    ) -> SelectionResult:
        """Select the best agent from candidates using intelligent scoring.

        Implements multi-criteria scoring with weighted ranking.

        Args:
            candidates: List of (agent_name, entry) tuples from registry.
            context: Selection context with requirements.
            timeout_seconds: Optional timeout for selection.

        Returns:
            SelectionResult with the chosen agent and score details.

        Raises:
            NoSuitableAgentError: If no agent matches requirements.
            SelectionTimeoutError: If selection takes too long.
        """
        timeout = timeout_seconds or self._default_timeout_seconds

        selected: AgentScore | None = None
        scores: list[AgentScore] = []
        strategy = AgentSelectionStrategy.CAPABILITY_BASED
        is_fallback = False

        try:
            await asyncio.wait_for(
                self._compute_scores(candidates, context, scores),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "agent_selection_timeout",
                timeout_seconds=timeout,
                candidate_count=len(candidates),
            )
            raise SelectionTimeoutError(
                f"Agent selection timed out after {timeout}s"
            ) from None

        # Filter out unavailable agents
        available_scores = [s for s in scores if s.is_available and s.health_status != AgentHealthStatus.UNHEALTHY]

        # If no available agents, try fallback
        if not available_scores:
            if context.fallback_allowed:
                is_fallback = True
                strategy = AgentSelectionStrategy.FALLBACK
                # Include unhealthy agents as fallback
                available_scores = [s for s in scores if s.agent_name]
                if not available_scores:
                    raise NoSuitableAgentError(
                        f"No agents available for task {context.task_id or 'unknown'}"
                    )
            else:
                raise NoSuitableAgentError(
                    f"No healthy agents available for task {context.task_id or 'unknown'}"
                )

        if not available_scores:
            raise NoSuitableAgentError(
                f"No agents match capabilities {sorted(context.required_capabilities)}"
            )

        # Find best score
        available_scores.sort(key=lambda s: s.total_score, reverse=True)
        selected = available_scores[0]

        self._logger.debug(
            "agent_selected",
            agent_name=selected.agent_name,
            strategy=strategy.value,
            total_score=selected.total_score,
            capability_score=selected.capability_score,
        )

        # Record metrics
        AGENT_SELECTIONS_TOTAL.labels(
            strategy=strategy.value,
            fallback=str(is_fallback).lower(),
        ).inc()

        return SelectionResult(
            selected_agent_name=selected.agent_name,
            is_fallback=is_fallback,
            scores=tuple(scores),
            selection_strategy=strategy,
        )

    async def _compute_scores(
        self,
        candidates: list[tuple[str, _AgentEntry]],
        context: SelectionContext,
        scores: list[AgentScore],
    ) -> None:
        """Compute scores for all candidate agents."""
        for agent_name, entry in candidates:
            score = self._score_agent(agent_name, entry, context)
            scores.append(score)

    def _score_agent(
        self,
        agent_name: str,
        entry: _AgentEntry,
        context: SelectionContext,
    ) -> AgentScore:
        """Calculate a composite score for an agent.

        Scoring components (weights sum to 1.0):
        - capability_score: 0.4 - Matches required capabilities
        - health_score: 0.25 - Health status and recent failures
        - load_score: 0.2 - Current in-flight count vs capacity
        - priority_score: 0.1 - Priority alignment
        - confidence_score: 0.05 - Historical success rate
        """
        # Capability score: ratio of matched capabilities
        cap_overlap = len(entry.agent.capabilities & context.required_capabilities)
        if context.required_capabilities:
            capability_score = cap_overlap / len(context.required_capabilities)
        else:
            capability_score = 1.0 if entry.agent.capabilities else 0.5

        # Health score: 1.0 = healthy, 0.0 = unhealthy, 0.5 = unknown
        if not entry.enabled:
            health_status = AgentHealthStatus.UNHEALTHY
            health_score = 0.0
        elif entry.health_status == AgentHealthStatus.HEALTHY:
            health_score = 1.0
        elif entry.health_status == AgentHealthStatus.UNHEALTHY:
            health_score = 0.0
        else:
            health_score = 0.5
            health_status = AgentHealthStatus.UNKNOWN

        # Load score: lower is better (inverse of in-flight ratio)
        capacity = 10  # Default capacity, could be configurable
        load_ratio = entry.in_flight / max(capacity, 1)
        load_score = max(0.0, 1.0 - load_ratio)

        # Priority score: match priority level
        priority_score = self._compute_priority_score(entry.agent, context.priority)

        # Confidence score: success rate
        confidence_score = entry.metrics.success_rate

        # Check availability
        is_available = entry.enabled and health_status != AgentHealthStatus.UNHEALTHY

        # Weighted total
        weights = {
            "capability": 0.4,
            "health": 0.25,
            "load": 0.2,
            "priority": 0.1,
            "confidence": 0.05,
        }

        total_score = (
            capability_score * weights["capability"]
            + health_score * weights["health"]
            + load_score * weights["load"]
            + priority_score * weights["priority"]
            + confidence_score * weights["confidence"]
        )

        # Zero out score if capabilities don't match
        if context.required_capabilities and capability_score == 0:
            total_score = 0.0
            is_available = False

        return AgentScore(
            agent_name=agent_name,
            total_score=total_score,
            capability_score=capability_score,
            health_score=health_score,
            load_score=load_score,
            priority_score=priority_score,
            confidence_score=confidence_score,
            is_available=is_available,
            health_status=health_status,
            raw_metrics={
                "in_flight": entry.in_flight,
                "total_executions": entry.metrics.total_executions,
                "success_rate": entry.metrics.success_rate,
            },
        )

    def _compute_priority_score(self, agent: Agent, task_priority: TaskPriority) -> float:
        """Compute priority alignment score."""
        # Map agent name prefixes to priorities (could be extended)
        priority_mapping = {
            "planner": TaskPriority.HIGH,
            "coder": TaskPriority.HIGH,
            "reviewer": TaskPriority.MEDIUM,
            "tester": TaskPriority.MEDIUM,
            "debugger": TaskPriority.HIGH,
            "memory": TaskPriority.LOW,
            "learning": TaskPriority.LOW,
            "evaluation": TaskPriority.MEDIUM,
        }

        agent_priority = priority_mapping.get(agent.name, TaskPriority.MEDIUM)

        if agent_priority == task_priority:
            return 1.0
        if abs(agent_priority.value - task_priority.value) == 1:
            return 0.7
        return 0.3


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
    ) -> None:
        self._entries: dict[str, _AgentEntry] = {}
        self._registry_lock = asyncio.Lock()
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._clock = clock
        self._logger = logger.bind(component="agent_manager")
        self._selection_engine = AgentSelectionEngine(
            default_timeout_seconds=selection_timeout_seconds
        )

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

        Uses the intelligent selection engine with multi-criteria scoring.

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

        # Build selection context
        context = SelectionContext(
            required_capabilities=required_capabilities,
            priority=getattr(task, "priority", TaskPriority.MEDIUM),
            attempt=getattr(task, "attempt", 1),
            fallback_allowed=getattr(task, "allow_fallback", True),
            health_check_required=getattr(task, "health_check_required", False),
            task_id=task.task_id,
        )

        try:
            result = await self._selection_engine.select(candidates, context)
            agent_name = result.selected_agent_name

            self._logger.debug(
                "agent_selected",
                task_id=task.task_id,
                agent_name=agent_name,
                strategy=result.selection_strategy.value,
                is_fallback=result.is_fallback,
            )

            return await self.get(agent_name)
        except SelectionTimeoutError:
            raise NoSuitableAgentError(
                f"Agent selection timed out for task {task.task_id!r}."
            ) from None

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

        async with self._registry_lock:
            entry = self._entries[agent.name]
            entry.in_flight += 1
            ACTIVE_AGENT_TASKS.labels(agent_type=agent.name).inc()

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
                ACTIVE_AGENT_TASKS.labels(agent_type=agent.name).dec()

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
            entry.health_status = AgentHealthStatus.HEALTHY if healthy else AgentHealthStatus.UNHEALTHY
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
                    entry.health_status = AgentHealthStatus.HEALTHY if healthy else AgentHealthStatus.UNHEALTHY
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
    ) -> list[AgentScore]:
        """Get ranking scores for all agents for a task (without executing).

        Useful for debugging and observability.

        Args:
            task: The task to evaluate.

        Returns:
            List of AgentScore objects for all registered agents.
        """
        required_capabilities = frozenset(getattr(task, "required_capabilities", ()) or ())

        async with self._registry_lock:
            candidates = list(self._entries.items())

        context = SelectionContext(
            required_capabilities=required_capabilities,
            priority=getattr(task, "priority", TaskPriority.MEDIUM),
            attempt=getattr(task, "attempt", 1),
            fallback_allowed=getattr(task, "allow_fallback", True),
            health_check_required=getattr(task, "health_check_required", False),
            task_id=task.task_id,
        )

        scores: list[AgentScore] = []
        await self._selection_engine._compute_scores(candidates, context, scores)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        return scores

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
        from kodiak.docker.errors import AgentExecutionError

        class AgentTaskImpl:
            def __init__(self, task: Task, attempt: int) -> None:
                self.task_id = str(task.id)
                self.task_type = task.title
                self._task = task
                self.attempt = attempt
                self.required_capabilities: frozenset[str] = frozenset()

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
        result = await self.execute(task_impl)

        if not result.success:
            if result.error is not None:
                raise result.error

        return result

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