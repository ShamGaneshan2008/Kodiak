"""Canonical agent registry for Kodiak.

Architecture position::

    Supervisor
        -> ExecutionEngine
            -> AgentManager
                -> AgentRegistry   (this module)
                    -> Specialized Agents


Those responsibilities belong to AgentManager and ExecutionEngine.

The registry is the single source of truth for which agents exist and
how to obtain a live instance of each. It is intentionally narrow:

* It registers, unregisters, and retrieves agents.
* It tracks metadata (capabilities, tags, version, dependencies) and
  lightweight statistics.
* It supports lazy, factory-based registration so expensive agents are
  only constructed when first needed, and eager registration for
  agents that are already built (e.g. by a DI container).

It explicitly does **not**:

* Execute agents or invoke their business logic. That is
  :class:`~kodiak.agents.manager.AgentManager`'s job.
* Select or route between agents. That is also :class:`AgentManager`'s
  job.
* Retry failed operations.
* Import or depend on any infrastructure module (databases, queues,
  HTTP clients, LLM clients, GitHub clients, etc). It depends only on
  :class:`AgentProtocol`, a structural typing contract, so it can be
  imported anywhere without pulling in the rest of the system.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)

_AGENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal structural contract required of a registrable agent.

    The registry depends only on this protocol so it never has to
    import concrete agent classes or infrastructure. Any object that
    exposes an ``agent_id`` attribute satisfies it.
    """

    agent_id: str


AgentInstance = AgentProtocol
AgentFactory = Callable[..., AgentInstance | Awaitable[AgentInstance]]


class AgentRegistryError(Exception):
    """Base exception for all agent registry errors."""


class AgentAlreadyRegisteredError(AgentRegistryError):
    """Raised when attempting to register an agent id that already exists."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent '{agent_id}' is already registered")
        self.agent_id = agent_id


class AgentNotFoundError(AgentRegistryError):
    """Raised when an operation references an unknown agent id."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent '{agent_id}' is not registered")
        self.agent_id = agent_id


class InvalidAgentMetadataError(AgentRegistryError):
    """Raised when agent metadata or registration arguments fail validation."""


@dataclass(frozen=True, slots=True)
class AgentMetadata:
    """Descriptive, immutable metadata for a registered agent.

    Attributes:
        agent_id: Unique, stable identifier for the agent.
        name: Human-readable display name.
        version: Semantic version string of the agent implementation.
        capabilities: Named capabilities the agent claims to provide.
        tags: Free-form tags used for filtering and grouping.
        description: Short description of the agent's purpose.
        dependencies: Named dependencies injected into the agent
            factory at instantiation time.
        lazy: Whether the agent is constructed on first retrieval
            rather than at registration time.
        registered_at: UTC timestamp of registration.
    """

    agent_id: str
    name: str
    version: str
    capabilities: tuple[str, ...]
    tags: tuple[str, ...]
    description: str
    dependencies: Mapping[str, Any]
    lazy: bool
    registered_at: datetime


@dataclass(frozen=True, slots=True)
class AgentRegistryStats:
    """Point-in-time statistics describing registry contents.

    Attributes:
        total_registered: Number of agent ids currently registered.
        instantiated: Number of agents with a live instance in memory.
        pending_lazy: Number of lazily-registered agents not yet
            instantiated.
        capabilities: Distinct capabilities across all registered
            agents, sorted.
        tags: Distinct tags across all registered agents, sorted.
    """

    total_registered: int
    instantiated: int
    pending_lazy: int
    capabilities: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(slots=True)
class _RegistryEntry:
    """Internal bookkeeping record for a single registered agent."""

    metadata: AgentMetadata
    factory: AgentFactory | None
    instance: AgentInstance | None
    instantiation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class AgentRegistry:
    """Canonical, in-memory store of agents known to Kodiak.

    The registry is purely responsible for agent lifecycle and
    storage: registering, unregistering, looking up, and describing
    agents. It never executes agent logic and never chooses which
    agent should handle a given task.

    The registry is async-safe: all mutating operations acquire an
    internal :class:`asyncio.Lock`, and lazy instantiation is guarded
    by a per-entry lock so constructing one agent never blocks access
    to the rest of the registry.

    Example:
    async def example() -> None:
        registry = AgentRegistry()

        await registry.register(
            "coder",
            factory=lambda **deps: CoderAgent(**deps),
            capabilities=["write_code", "refactor"],
            dependencies={"llm_client": llm_client},
        )

        agent = await registry.get("coder")
    """

    def __init__(self) -> None:
        """Initializes an empty agent registry."""
        self._entries: dict[str, _RegistryEntry] = {}
        self._lock = asyncio.Lock()
        self._log = logger.bind(component="agent_registry")

    async def register(
        self,
        agent_id: str,
        *,
        instance: AgentInstance | None = None,
        factory: AgentFactory | None = None,
        name: str | None = None,
        version: str = "0.1.0",
        capabilities: Sequence[str] = (),
        tags: Sequence[str] = (),
        description: str = "",
        dependencies: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> AgentMetadata:
        """Registers an agent, eagerly or lazily.

        Exactly one of ``instance`` or ``factory`` must be provided.
        Passing ``instance`` registers an already-constructed agent
        (eager registration), which is how a dependency-injection
        container typically hands the registry a fully-wired agent.
        Passing ``factory`` registers a callable that is invoked on
        first :meth:`get` (lazy registration), keeping startup cheap
        for agents with expensive initialization.

        Args:
            agent_id: Unique identifier for the agent. Must match
                ``^[a-z0-9][a-z0-9_-]*$``.
            instance: A pre-constructed agent satisfying
                :class:`AgentProtocol`. Mutually exclusive with
                ``factory``.
            factory: A callable (sync or async) that constructs and
                returns an agent when called with ``dependencies`` as
                keyword arguments. Mutually exclusive with
                ``instance``.
            name: Human-readable name. Defaults to ``agent_id``.
            version: Semantic version of the agent implementation.
            capabilities: Capabilities the agent claims to provide.
            tags: Free-form tags for filtering and grouping.
            description: Short human-readable description.
            dependencies: Keyword arguments injected into ``factory``
                when the agent is lazily instantiated. Ignored when
                ``instance`` is supplied.
            replace: If ``True``, replaces an existing registration
                for ``agent_id`` instead of raising.

        Returns:
            The :class:`AgentMetadata` recorded for the agent.

        Raises:
            InvalidAgentMetadataError: If ``agent_id`` or ``version``
                is invalid, or if ``instance``/``factory`` are both
                provided or both omitted, or if ``instance`` does not
                satisfy :class:`AgentProtocol`.
            AgentAlreadyRegisteredError: If ``agent_id`` is already
                registered and ``replace`` is ``False``.
        """
        self._validate_agent_id(agent_id)

        if not isinstance(version, str) or not version:
            raise InvalidAgentMetadataError(
                f"version for agent '{agent_id}' must be a non-empty string"
            )

        if (instance is None) == (factory is None):
            raise InvalidAgentMetadataError(
                "exactly one of 'instance' or 'factory' must be provided "
                f"for agent '{agent_id}'"
            )

        if instance is not None and not isinstance(instance, AgentProtocol):
            raise InvalidAgentMetadataError(
                f"instance for agent '{agent_id}' does not satisfy AgentProtocol"
            )

        metadata = AgentMetadata(
            agent_id=agent_id,
            name=name or agent_id,
            version=version,
            capabilities=tuple(capabilities),
            tags=tuple(tags),
            description=description,
            dependencies=dict(dependencies or {}),
            lazy=instance is None,
            registered_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            if agent_id in self._entries and not replace:
                raise AgentAlreadyRegisteredError(agent_id)

            self._entries[agent_id] = _RegistryEntry(
                metadata=metadata,
                factory=factory,
                instance=instance,
            )

        self._log.info(
            "agent_registered",
            agent_id=agent_id,
            lazy=metadata.lazy,
            capabilities=metadata.capabilities,
            replaced=replace,
        )
        return metadata

    async def unregister(self, agent_id: str, *, strict: bool = True) -> bool:
        """Removes an agent from the registry.

        Args:
            agent_id: Identifier of the agent to remove.
            strict: If ``True``, raises when ``agent_id`` is not
                registered. If ``False``, returns ``False`` instead of
                raising.

        Returns:
            ``True`` if an agent was removed, ``False`` if it did not
            exist and ``strict`` is ``False``.

        Raises:
            AgentNotFoundError: If ``agent_id`` is not registered and
                ``strict`` is ``True``.
        """
        async with self._lock:
            entry = self._entries.pop(agent_id, None)

        if entry is None:
            if strict:
                raise AgentNotFoundError(agent_id)
            return False

        self._log.info("agent_unregistered", agent_id=agent_id)
        return True

    async def get(self, agent_id: str) -> AgentInstance:
        """Retrieves an agent instance, instantiating lazy agents on demand.

        Args:
            agent_id: Identifier of the agent to retrieve.

        Returns:
            The live agent instance satisfying :class:`AgentProtocol`.

        Raises:
            AgentNotFoundError: If ``agent_id`` is not registered.
            AgentRegistryError: If a lazy agent's factory fails to
                produce an object satisfying :class:`AgentProtocol`.
        """
        async with self._lock:
            entry = self._entries.get(agent_id)

        if entry is None:
            raise AgentNotFoundError(agent_id)

        if entry.instance is not None:
            return entry.instance

        async with entry.instantiation_lock:
            if entry.instance is not None:
                return entry.instance

            if entry.factory is None:
                raise AgentRegistryError(
                    f"agent '{agent_id}' has neither an instance nor a "
                    "factory; registry state is corrupt"
                )

            self._log.debug("agent_instantiating", agent_id=agent_id)
            result = entry.factory(**entry.metadata.dependencies)
            if inspect.isawaitable(result):
                result = await result

            if not isinstance(result, AgentProtocol):
                raise AgentRegistryError(
                    f"factory for agent '{agent_id}' did not produce an "
                    "object satisfying AgentProtocol"
                )

            entry.instance = result
            self._log.info("agent_instantiated", agent_id=agent_id)
            return entry.instance

    async def exists(self, agent_id: str) -> bool:
        """Checks whether an agent id is registered.

        Args:
            agent_id: Identifier to check.

        Returns:
            ``True`` if the agent is registered, ``False`` otherwise.
        """
        async with self._lock:
            return agent_id in self._entries

    async def list_agents(
        self,
        *,
        tag: str | None = None,
        capability: str | None = None,
    ) -> list[AgentMetadata]:
        """Lists metadata for registered agents, optionally filtered.

        Args:
            tag: If provided, only agents with this tag are included.
            capability: If provided, only agents with this capability
                are included.

        Returns:
            A list of :class:`AgentMetadata`, ordered by ``agent_id``.
        """
        async with self._lock:
            entries = list(self._entries.values())

        results = [
            entry.metadata
            for entry in entries
            if (tag is None or tag in entry.metadata.tags)
            and (capability is None or capability in entry.metadata.capabilities)
        ]
        return sorted(results, key=lambda metadata: metadata.agent_id)

    async def clear(self) -> int:
        """Removes all registered agents.

        Returns:
            The number of agents that were removed.
        """
        async with self._lock:
            removed = len(self._entries)
            self._entries.clear()

        if removed:
            self._log.info("agent_registry_cleared", removed=removed)
        return removed

    async def count(self) -> int:
        """Returns the number of currently registered agents.

        Returns:
            The count of registered agents.
        """
        async with self._lock:
            return len(self._entries)

    async def snapshot(self) -> dict[str, AgentMetadata]:
        """Returns a point-in-time copy of registry metadata.

        Returns:
            A mapping of ``agent_id`` to :class:`AgentMetadata`. The
            returned dict is a copy; mutating it does not affect the
            registry.
        """
        async with self._lock:
            return {agent_id: entry.metadata for agent_id, entry in self._entries.items()}

    async def stats(self) -> AgentRegistryStats:
        """Computes summary statistics for the registry's current contents.

        Returns:
            An :class:`AgentRegistryStats` snapshot.
        """
        async with self._lock:
            entries = list(self._entries.values())

        instantiated = sum(1 for entry in entries if entry.instance is not None)
        pending_lazy = sum(
            1 for entry in entries if entry.metadata.lazy and entry.instance is None
        )
        capabilities = sorted({cap for entry in entries for cap in entry.metadata.capabilities})
        tags = sorted({tag for entry in entries for tag in entry.metadata.tags})

        return AgentRegistryStats(
            total_registered=len(entries),
            instantiated=instantiated,
            pending_lazy=pending_lazy,
            capabilities=tuple(capabilities),
            tags=tuple(tags),
        )

    @staticmethod
    def _validate_agent_id(agent_id: str) -> None:
        """Validates that an agent id is a well-formed identifier.

        Args:
            agent_id: Candidate identifier.

        Raises:
            InvalidAgentMetadataError: If ``agent_id`` is not a
                non-empty string matching the required pattern.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise InvalidAgentMetadataError("agent_id must be a non-empty string")
        if not _AGENT_ID_PATTERN.match(agent_id):
            raise InvalidAgentMetadataError(
                f"agent_id '{agent_id}' must match pattern "
                f"'{_AGENT_ID_PATTERN.pattern}'"
            )