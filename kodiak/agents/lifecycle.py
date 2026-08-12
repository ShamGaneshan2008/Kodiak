"""Agent lifecycle state, transitions, and management."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from kodiak.agents.adapters import DiscoveredAgentHandle
from kodiak.agents.base import BaseAgent
from kodiak.agents.registry import AgentNotFoundError, AgentRegistry

logger = structlog.get_logger(__name__)


class AgentLifecycleState(StrEnum):
    """Operational lifecycle state of a registered agent."""

    DISCOVERED = "discovered"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    INITIALIZATION_FAILED = "initialization_failed"
    DEGRADED = "degraded"
    FAILED = "failed"


class AgentLifecycleHealth(StrEnum):
    """Health classification derived from lifecycle state and checks."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


_SELECTABLE_STATES = frozenset(
    {
        AgentLifecycleState.READY,
        AgentLifecycleState.DEGRADED,
    }
)

_VALID_TRANSITIONS: dict[AgentLifecycleState, frozenset[AgentLifecycleState]] = {
    AgentLifecycleState.DISCOVERED: frozenset(
        {AgentLifecycleState.INITIALIZING, AgentLifecycleState.STOPPING}
    ),
    AgentLifecycleState.INITIALIZING: frozenset(
        {
            AgentLifecycleState.READY,
            AgentLifecycleState.INITIALIZATION_FAILED,
            AgentLifecycleState.STOPPING,
        }
    ),
    AgentLifecycleState.READY: frozenset(
        {AgentLifecycleState.RUNNING, AgentLifecycleState.STOPPING}
    ),
    AgentLifecycleState.RUNNING: frozenset(
        {
            AgentLifecycleState.READY,
            AgentLifecycleState.DEGRADED,
            AgentLifecycleState.STOPPING,
        }
    ),
    AgentLifecycleState.DEGRADED: frozenset(
        {AgentLifecycleState.READY, AgentLifecycleState.STOPPING}
    ),
    AgentLifecycleState.INITIALIZATION_FAILED: frozenset(
        {AgentLifecycleState.INITIALIZING, AgentLifecycleState.STOPPED, AgentLifecycleState.STOPPING}
    ),
    AgentLifecycleState.STOPPING: frozenset(
        {AgentLifecycleState.STOPPED, AgentLifecycleState.FAILED}
    ),
    AgentLifecycleState.STOPPED: frozenset({AgentLifecycleState.INITIALIZING}),
    AgentLifecycleState.FAILED: frozenset(
        {AgentLifecycleState.INITIALIZING, AgentLifecycleState.STOPPED}
    ),
}


class AgentLifecycleError(Exception):
    """Base exception for agent lifecycle errors."""


class InvalidLifecycleTransitionError(AgentLifecycleError):
    """Raised when a lifecycle state transition is not permitted."""

    def __init__(
        self,
        agent_id: str,
        current: AgentLifecycleState,
        target: AgentLifecycleState,
        operation: str,
    ) -> None:
        super().__init__(
            f"Invalid lifecycle transition for agent {agent_id!r}: "
            f"{current.value} -> {target.value} during {operation}"
        )
        self.agent_id = agent_id
        self.current = current
        self.target = target
        self.operation = operation


class LifecycleOperationConflictError(AgentLifecycleError):
    """Raised when a lifecycle operation conflicts with one in progress."""


class LifecycleOperationError(AgentLifecycleError):
    """Raised when a lifecycle hook fails."""


@dataclass(slots=True)
class LifecycleRecord:
    """Mutable lifecycle tracking record for one agent."""

    state: AgentLifecycleState = AgentLifecycleState.DISCOVERED
    health: AgentLifecycleHealth = AgentLifecycleHealth.STOPPED
    last_error: str | None = None
    last_transition_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operation_in_progress: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleTransitionEvent:
    """Observed lifecycle transition."""

    agent_id: str
    previous_state: AgentLifecycleState
    new_state: AgentLifecycleState
    operation: str
    duration_seconds: float
    error: str | None = None


class AgentLifecycleManager:
    """Manages initialize/start/stop/shutdown/health for registry agents.

    Does NOT discover, select, or execute tasks.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._records: dict[str, LifecycleRecord] = {}
        self._global_lock = asyncio.Lock()
        self._log = logger.bind(component="agent_lifecycle")

    async def mark_discovered(self, agent_id: str) -> AgentLifecycleState:
        """Record a newly registered agent in DISCOVERED state."""
        async with self._global_lock:
            record = self._records.setdefault(agent_id, LifecycleRecord())
            if record.state is AgentLifecycleState.DISCOVERED:
                return record.state
            if record.state is AgentLifecycleState.STOPPED:
                return record.state
            return record.state

    async def sync_from_registry(self) -> tuple[str, ...]:
        """Ensure every registry agent has a lifecycle record."""
        discovered: list[str] = []
        metadata = await self._registry.list_agents()
        async with self._global_lock:
            for item in metadata:
                if item.agent_id not in self._records:
                    self._records[item.agent_id] = LifecycleRecord()
                    discovered.append(item.agent_id)
        return tuple(sorted(discovered))

    def get_state(self, agent_id: str) -> AgentLifecycleState | None:
        record = self._records.get(agent_id)
        return record.state if record is not None else None

    def get_health(self, agent_id: str) -> AgentLifecycleHealth | None:
        record = self._records.get(agent_id)
        return record.health if record is not None else None

    def is_selectable(self, agent_id: str) -> bool:
        record = self._records.get(agent_id)
        if record is None:
            return True
        return record.state in _SELECTABLE_STATES

    async def initialize(self, agent_id: str) -> AgentLifecycleState:
        """Initialize an agent and transition it to READY on success."""
        record = await self._ensure_record(agent_id)
        async with record.lock:
            if record.state in {
                AgentLifecycleState.READY,
                AgentLifecycleState.RUNNING,
                AgentLifecycleState.DEGRADED,
            }:
                return record.state
            if record.operation_in_progress:
                raise LifecycleOperationConflictError(
                    f"Lifecycle operation already in progress for agent {agent_id!r}."
                )

            record.operation_in_progress = True
            start = self._clock()
            try:
                self._transition(record, AgentLifecycleState.INITIALIZING, agent_id, "initialize")
                agent = await self._resolve_base_agent(agent_id)
                if agent is not None:
                    await agent.initialize()
                self._transition(record, AgentLifecycleState.READY, agent_id, "initialize")
                record.health = AgentLifecycleHealth.HEALTHY
                record.last_error = None
                self._log.info(
                    "agent_lifecycle_initialized",
                    agent_id=agent_id,
                    duration_seconds=self._clock() - start,
                )
                return record.state
            except InvalidLifecycleTransitionError:
                raise
            except Exception as exc:
                record.last_error = str(exc)
                if record.state is AgentLifecycleState.INITIALIZING:
                    self._transition(
                        record,
                        AgentLifecycleState.INITIALIZATION_FAILED,
                        agent_id,
                        "initialize",
                    )
                record.health = AgentLifecycleHealth.UNHEALTHY
                self._log.exception(
                    "agent_lifecycle_initialize_failed",
                    agent_id=agent_id,
                    error=str(exc),
                )
                raise LifecycleOperationError(
                    f"Initialization failed for agent {agent_id!r}: {exc}"
                ) from exc
            finally:
                record.operation_in_progress = False

    async def start(self, agent_id: str) -> AgentLifecycleState:
        """Start an agent, transitioning READY/DEGRADED to RUNNING."""
        record = await self._ensure_record(agent_id)
        async with record.lock:
            if record.state is AgentLifecycleState.RUNNING:
                return record.state
            if record.operation_in_progress:
                raise LifecycleOperationConflictError(
                    f"Lifecycle operation already in progress for agent {agent_id!r}."
                )

            record.operation_in_progress = True
            start = self._clock()
            try:
                self._transition(record, AgentLifecycleState.RUNNING, agent_id, "start")
                agent = await self._resolve_base_agent(agent_id)
                if agent is not None:
                    await agent.start()
                record.health = AgentLifecycleHealth.HEALTHY
                record.last_error = None
                self._log.info(
                    "agent_lifecycle_started",
                    agent_id=agent_id,
                    duration_seconds=self._clock() - start,
                )
                return record.state
            except (InvalidLifecycleTransitionError, LifecycleOperationConflictError):
                raise
            except Exception as exc:
                record.last_error = str(exc)
                if record.state is AgentLifecycleState.RUNNING:
                    self._transition(record, AgentLifecycleState.FAILED, agent_id, "start")
                record.health = AgentLifecycleHealth.UNHEALTHY
                raise LifecycleOperationError(f"Start failed for agent {agent_id!r}: {exc}") from exc
            finally:
                record.operation_in_progress = False

    async def stop(self, agent_id: str) -> AgentLifecycleState:
        """Stop an agent, returning it to READY."""
        record = await self._ensure_record(agent_id)
        async with record.lock:
            if record.state is AgentLifecycleState.READY:
                return record.state
            if record.state not in {AgentLifecycleState.RUNNING, AgentLifecycleState.DEGRADED}:
                raise InvalidLifecycleTransitionError(
                    agent_id,
                    record.state,
                    AgentLifecycleState.READY,
                    "stop",
                )
            if record.operation_in_progress:
                raise LifecycleOperationConflictError(
                    f"Lifecycle operation already in progress for agent {agent_id!r}."
                )

            record.operation_in_progress = True
            start = self._clock()
            try:
                agent = await self._resolve_base_agent(agent_id)
                if agent is not None:
                    await agent.stop()
                self._transition(record, AgentLifecycleState.READY, agent_id, "stop")
                record.health = AgentLifecycleHealth.HEALTHY
                self._log.info(
                    "agent_lifecycle_stopped",
                    agent_id=agent_id,
                    duration_seconds=self._clock() - start,
                )
                return record.state
            except InvalidLifecycleTransitionError:
                raise
            except Exception as exc:
                record.last_error = str(exc)
                self._transition(record, AgentLifecycleState.FAILED, agent_id, "stop")
                record.health = AgentLifecycleHealth.UNHEALTHY
                raise LifecycleOperationError(f"Stop failed for agent {agent_id!r}: {exc}") from exc
            finally:
                record.operation_in_progress = False

    async def shutdown(self, agent_id: str) -> AgentLifecycleState:
        """Shut down an agent and release resources."""
        record = await self._ensure_record(agent_id)
        async with record.lock:
            if record.state is AgentLifecycleState.STOPPED:
                return record.state
            if record.operation_in_progress:
                raise LifecycleOperationConflictError(
                    f"Lifecycle operation already in progress for agent {agent_id!r}."
                )

            record.operation_in_progress = True
            start = self._clock()
            try:
                if record.state is AgentLifecycleState.RUNNING:
                    agent = await self._resolve_base_agent(agent_id)
                    if agent is not None:
                        await agent.stop()

                self._transition(record, AgentLifecycleState.STOPPING, agent_id, "shutdown")
                agent = await self._resolve_base_agent(agent_id)
                if agent is not None:
                    await agent.shutdown()
                self._transition(record, AgentLifecycleState.STOPPED, agent_id, "shutdown")
                record.health = AgentLifecycleHealth.STOPPED
                record.last_error = None
                self._log.info(
                    "agent_lifecycle_shutdown",
                    agent_id=agent_id,
                    duration_seconds=self._clock() - start,
                )
                return record.state
            except (InvalidLifecycleTransitionError, LifecycleOperationConflictError):
                raise
            except Exception as exc:
                record.last_error = str(exc)
                if record.state is AgentLifecycleState.STOPPING:
                    self._transition(record, AgentLifecycleState.FAILED, agent_id, "shutdown")
                record.health = AgentLifecycleHealth.UNHEALTHY
                raise LifecycleOperationError(
                    f"Shutdown failed for agent {agent_id!r}: {exc}"
                ) from exc
            finally:
                record.operation_in_progress = False

    async def restart(self, agent_id: str) -> AgentLifecycleState:
        """Restart an agent via stop/shutdown, initialize, and start."""
        state = self.get_state(agent_id)
        if state is AgentLifecycleState.RUNNING:
            await self.stop(agent_id)
        elif state not in {
            AgentLifecycleState.STOPPED,
            AgentLifecycleState.READY,
            AgentLifecycleState.DISCOVERED,
            AgentLifecycleState.INITIALIZATION_FAILED,
            AgentLifecycleState.FAILED,
            None,
        }:
            await self.shutdown(agent_id)
        elif state in {AgentLifecycleState.FAILED, AgentLifecycleState.STOPPED}:
            pass
        await self.initialize(agent_id)
        return await self.start(agent_id)

    async def health_check(self, agent_id: str) -> AgentLifecycleHealth:
        """Evaluate agent health from lifecycle state and optional agent hook."""
        record = await self._ensure_record(agent_id)
        async with record.lock:
            if record.state is AgentLifecycleState.STOPPED:
                record.health = AgentLifecycleHealth.STOPPED
                return record.health
            if record.state in {
                AgentLifecycleState.INITIALIZATION_FAILED,
                AgentLifecycleState.FAILED,
            }:
                record.health = AgentLifecycleHealth.UNHEALTHY
                return record.health
            if record.state is AgentLifecycleState.DEGRADED:
                record.health = AgentLifecycleHealth.DEGRADED
                return record.health

            agent = await self._resolve_base_agent(agent_id)
            if agent is not None:
                try:
                    healthy = bool(await agent.health_check())
                except Exception as exc:
                    record.last_error = str(exc)
                    record.health = AgentLifecycleHealth.UNHEALTHY
                    if record.state is AgentLifecycleState.RUNNING:
                        self._transition(record, AgentLifecycleState.DEGRADED, agent_id, "health_check")
                    return record.health
                if not healthy:
                    record.health = AgentLifecycleHealth.DEGRADED
                    if record.state is AgentLifecycleState.RUNNING:
                        self._transition(record, AgentLifecycleState.DEGRADED, agent_id, "health_check")
                    return record.health

            record.health = AgentLifecycleHealth.HEALTHY
            return record.health

    async def _ensure_record(self, agent_id: str) -> LifecycleRecord:
        if not await self._registry.exists(agent_id):
            raise AgentNotFoundError(agent_id)
        async with self._global_lock:
            return self._records.setdefault(agent_id, LifecycleRecord())

    async def _resolve_base_agent(self, agent_id: str) -> BaseAgent | None:
        instance = await self._registry.get(agent_id)
        if isinstance(instance, DiscoveredAgentHandle):
            return instance.agent
        if isinstance(instance, BaseAgent):
            return instance
        return None

    def _transition(
        self,
        record: LifecycleRecord,
        target: AgentLifecycleState,
        agent_id: str,
        operation: str,
    ) -> LifecycleTransitionEvent:
        current = record.state
        allowed = _VALID_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise InvalidLifecycleTransitionError(agent_id, current, target, operation)
        start = record.last_transition_at
        record.state = target
        now = self._clock()
        record.last_transition_at = now
        event = LifecycleTransitionEvent(
            agent_id=agent_id,
            previous_state=current,
            new_state=target,
            operation=operation,
            duration_seconds=now - start,
        )
        self._log.info(
            "agent_lifecycle_transition",
            agent_id=agent_id,
            previous_state=current.value,
            new_state=target.value,
            operation=operation,
        )
        return event


__all__ = [
    "AgentLifecycleError",
    "AgentLifecycleHealth",
    "AgentLifecycleManager",
    "AgentLifecycleState",
    "InvalidLifecycleTransitionError",
    "LifecycleOperationConflictError",
    "LifecycleOperationError",
    "LifecycleRecord",
    "LifecycleTransitionEvent",
]
