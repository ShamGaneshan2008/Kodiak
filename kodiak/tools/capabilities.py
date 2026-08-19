"""Capability discovery and matching system."""

from __future__ import annotations

from collections.abc import Iterable

import structlog

from kodiak.tools.models import Capability, PermissionLevel, ToolDefinition, ToolExecutionContext
from kodiak.tools.permissions import PermissionEngine
from kodiak.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class CapabilityRegistry:
    """Manages capability definitions, discovery, and tool matching."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._logger = logger.bind(component="capability_registry")

    def register_capability(self, capability: Capability) -> None:
        """Register a new capability.

        Args:
            capability: Capability model to register.
        """
        self._capabilities[capability.identifier] = capability
        self._logger.info(
            "capability_registered",
            identifier=capability.identifier,
            supported_tools=list(capability.supported_tools),
        )

    def get_capability(self, identifier: str) -> Capability | None:
        """Retrieve capability metadata by identifier.

        Args:
            identifier: Capability identifier.

        Returns:
            Capability object if registered, None otherwise.
        """
        return self._capabilities.get(identifier)

    def list_capabilities(self) -> list[Capability]:
        """List all registered capabilities.

        Returns:
            List of Capability models.
        """
        return list(self._capabilities.values())

    def match_tools(
        self,
        required_capabilities: Iterable[str],
        tool_registry: ToolRegistry,
    ) -> list[ToolDefinition]:
        """Find registered tools matching any of the required capabilities.

        Args:
            required_capabilities: Iterable of required capability identifiers.
            tool_registry: Tool registry to search.

        Returns:
            List of ToolDefinition objects matching requirements.
        """
        required_set = set(required_capabilities)
        if not required_set:
            return tool_registry.list_tools()

        matched: list[ToolDefinition] = []
        for tool_def in tool_registry.list_tools():
            # Match tool declared capabilities
            if tool_def.capabilities & required_set:
                matched.append(tool_def)
                continue

            # Match capability registry supported tools mapping
            for cap_id in required_set:
                cap = self._capabilities.get(cap_id)
                if cap and tool_def.name in cap.supported_tools:
                    matched.append(tool_def)
                    break

        return matched

    def discover_tools_for_agent(
        self,
        agent_capabilities: Iterable[str],
        tool_registry: ToolRegistry,
        context: ToolExecutionContext | None = None,
        permission_engine: PermissionEngine | None = None,
    ) -> list[ToolDefinition]:
        """Discover tools that an agent is allowed to use based on its capabilities and permissions.

        Args:
            agent_capabilities: Iterable of capabilities possessed by the agent.
            tool_registry: Tool registry to discover from.
            context: Context containing caller identity and permissions.
            permission_engine: Engine to filter out denied tools.

        Returns:
            List of ToolDefinition objects allowed for the agent.
        """
        granted = set(agent_capabilities)
        if context is not None:
            granted.update(context.granted_capabilities)

        eval_context = context or ToolExecutionContext(granted_capabilities=frozenset(granted))

        pe = permission_engine or PermissionEngine()
        available: list[ToolDefinition] = []

        for tool_def in tool_registry.list_tools():
            perm_level = pe.check_permission(tool_def, eval_context)
            if perm_level != PermissionLevel.DENIED:
                available.append(tool_def)

        return available


__all__ = ["CapabilityRegistry"]
