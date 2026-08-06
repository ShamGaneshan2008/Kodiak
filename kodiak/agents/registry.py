"""Agent registry for Kodiak.

This module owns agent lifecycle management:
- registering agents
- removing agents
- looking up agents
- exposing registry snapshots

It intentionally does not:
- execute agents
- select agents
- handle retries
- manage infrastructure

Those responsibilities belong to AgentManager and ExecutionEnfrom .base import Agentgine.
"""

from __future__ import annotations

import asyncio

import structlog



logger = structlog.get_logger(__name__)

__all__ = [
    "AgentRegistry",
    "AgentRegistryError",
    "AgentAlreadyRegisteredError",
    "AgentNotFoundError",
]


class AgentRegistryError(Exception):
    """Base exception for registry failures."""


class AgentAlreadyRegisteredError(AgentRegistryError):
    """Raised when registering an existing agent name."""


class AgentNotFoundError(AgentRegistryError):
    """Raised when an agent cannot be found."""


class AgentRegistry:
    """Thread-safe asynchronous registry of Kodiak agents.

    The registry is the single source of truth for available agents.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = asyncio.Lock()
        self._logger = logger.bind(component="agent_registry")

    async def register(self, agent: Agent) -> None:
        """Register a new agent.

        Args:
            agent: Agent implementation satisfying the Agent protocol.

        Raises:
            AgentAlreadyRegisteredError: If an agent with the same name exists.
        """
        async with self._lock:
            if agent.name in self._agents:
                self._logger.warning(
                    "agent_registration_failed",
                    agent_name=agent.name,
                    reason="already_exists",
                )
                raise AgentAlreadyRegisteredError(
                    f"Agent {agent.name!r} already registered."
                )

            self._agents[agent.name] = agent
            self._logger.info(
                "agent_registered",
                agent_name=agent.name,
                capabilities=list(agent.capabilities),
            )

    async def unregister(self, agent_name: str) -> None:
        """Remove an agent from the registry.

        Args:
            agent_name: Registered agent name.

        Raises:
            AgentNotFoundError: If the agent does not exist.
        """
        async with self._lock:
            if agent_name not in self._agents:
                raise AgentNotFoundError(f"Agent {agent_name!r} not found.")

            del self._agents[agent_name]
            self._logger.info("agent_unregistered", agent_name=agent_name)

    async def get(self, agent_name: str) -> Agent:
        """Retrieve an agent by name.

        Args:
            agent_name: Agent identifier.

        Returns:
            Registered agent instance.

        Raises:
            AgentNotFoundError: If the agent does not exist.
        """
        async with self._lock:
            agent = self._agents.get(agent_name)
            if agent is None:
                raise AgentNotFoundError(f"Agent {agent_name!r} not found.")
            return agent

    async def exists(self, agent_name: str) -> bool:
        """Check whether an agent exists.

        Args:
            agent_name: Agent identifier.

        Returns:
            True if an agent with that name is registered.
        """
        async with self._lock:
            return agent_name in self._agents

    async def list_agents(self) -> list[Agent]:
        """Return a snapshot of all registered agents.

        Returns:
            A list of currently registered agent instances.
        """
        async with self._lock:
            return list(self._agents.values())

    async def names(self) -> list[str]:
        """Return registered agent names.

        Returns:
            A list of currently registered agent names.
        """
        async with self._lock:
            return list(self._agents.keys())

    async def count(self) -> int:
        """Return the number of registered agents.

        Returns:
            The current registry size.
        """
        async with self._lock:
            return len(self._agents)

    async def snapshot(self) -> dict[str, dict[str, object]]:
        """Return a serializable registry snapshot.

        Useful for debugging, API responses, and monitoring dashboards.

        Returns:
            A mapping of agent name to a dict of its name and capabilities.
        """
        async with self._lock:
            return {
                name: {
                    "name": agent.name,
                    "capabilities": list(agent.capabilities),
                }
                for name, agent in self._agents.items()
            }

    async def clear(self) -> None:
        """Remove all registered agents.

        Mainly useful for testing and shutdown procedures.
        """
        async with self._lock:
            self._agents.clear()
            self._logger.info("agent_registry_cleared")