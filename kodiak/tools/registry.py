"""Central Tool Registry for Kodiak."""

from __future__ import annotations

import asyncio

import structlog

from kodiak.tools.base import ToolAdapter
from kodiak.tools.exceptions import ToolNotFoundError, ToolRegistrationError
from kodiak.tools.models import ToolDefinition

logger = structlog.get_logger(__name__)


class ToolRegistry:
    """Central manager and registry for tools.

    Provides registration, removal, lookup, metadata query, and listing
    of tools. Thread-safe and async-safe.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolAdapter] = {}
        self._lock = asyncio.Lock()
        self._logger = logger.bind(component="tool_registry")

    def register_tool(self, tool: ToolAdapter) -> None:
        """Register a tool instance synchronously or raise if name conflict.

        Args:
            tool: ToolAdapter implementation to register.

        Raises:
            ToolRegistrationError: If a tool with the same name is already registered.
        """
        name = tool.definition.name
        if not name:
            raise ToolRegistrationError("Tool definition must specify a non-empty name.")

        if name in self._tools:
            self._logger.warning("tool_registration_conflict", tool_name=name)
            raise ToolRegistrationError(f"Tool with name {name!r} is already registered.")

        self._tools[name] = tool
        self._logger.info(
            "tool_registered",
            tool_name=name,
            version=tool.definition.version,
            capabilities=list(tool.definition.capabilities),
        )

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Tool identifier to unregister.

        Raises:
            ToolNotFoundError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool {name!r} is not registered.")

        del self._tools[name]
        self._logger.info("tool_unregistered", tool_name=name)

    def get_tool(self, name: str) -> ToolAdapter:
        """Look up a tool by name.

        Args:
            name: Tool identifier to look up.

        Returns:
            The registered ToolAdapter instance.

        Raises:
            ToolNotFoundError: If the tool does not exist.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool {name!r} was not found in registry.")
        return tool

    def has_tool(self, name: str) -> bool:
        """Check whether a tool exists in the registry.

        Args:
            name: Tool identifier.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return name in self._tools

    def list_tools(self) -> list[ToolDefinition]:
        """List metadata definitions of all registered tools.

        Returns:
            List of ToolDefinition objects.
        """
        return [tool.definition for tool in self._tools.values()]

    def get_metadata(self, name: str) -> ToolDefinition:
        """Retrieve metadata for a specific tool.

        Args:
            name: Tool identifier.

        Returns:
            The ToolDefinition associated with the tool.

        Raises:
            ToolNotFoundError: If the tool does not exist.
        """
        return self.get_tool(name).definition

    def clear(self) -> None:
        """Clear all registered tools (primarily used in test teardowns)."""
        self._tools.clear()


__all__ = ["ToolRegistry"]
