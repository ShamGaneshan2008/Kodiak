"""ToolRouter — controlled boundary between agents and tool execution."""

from __future__ import annotations

import time
from typing import Any

import structlog

from kodiak.tools.base import ToolAdapter
from kodiak.tools.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)
from kodiak.tools.invoker import ToolInvoker
from kodiak.tools.models import ToolDefinition, ToolExecutionContext, ToolResult
from kodiak.tools.permissions import PermissionEngine
from kodiak.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class ToolRouter:
    """Routes agent tool requests through registry lookup, permissions, and invocation.

    ToolRouter decides whether and how an agent may call a tool. Individual tools
    perform the actual operation and return :class:`ToolResult`.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        permission_engine: PermissionEngine | None = None,
    ) -> None:
        self._registry = registry or ToolRegistry()
        self._permission_engine = permission_engine or PermissionEngine()
        self._invoker = ToolInvoker(self._registry, self._permission_engine)
        self._logger = logger.bind(component="tool_router")

    @property
    def registry(self) -> ToolRegistry:
        """Underlying tool registry."""
        return self._registry

    @property
    def permission_engine(self) -> PermissionEngine:
        """Permission engine used for authorization checks."""
        return self._permission_engine

    def register_tool(self, tool: ToolAdapter) -> None:
        """Register a tool adapter."""
        self._registry.register_tool(tool)

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name."""
        self._registry.unregister_tool(name)

    def has_tool(self, name: str) -> bool:
        """Return True if the tool is registered."""
        return self._registry.has_tool(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Return deterministic metadata for all registered tools."""
        return sorted(self._registry.list_tools(), key=lambda d: d.name)

    def get_metadata(self, name: str) -> ToolDefinition:
        """Return metadata for a registered tool."""
        return self._registry.get_metadata(name)

    def route(
        self,
        action: str | None = None,
        *,
        required_capability: str | None = None,
    ) -> ToolDefinition | None:
        """Resolve a tool by explicit name or required capability.

        Args:
            action: Exact tool name to match.
            required_capability: Capability that the tool must expose.

        Returns:
            Matching :class:`ToolDefinition`, or ``None`` if no match.
        """
        tools = self.list_tools()
        if action is not None:
            for tool_def in tools:
                if tool_def.name == action:
                    if required_capability and required_capability not in tool_def.capabilities:
                        self._logger.warning(
                            "tool_route_capability_mismatch",
                            tool=tool_def.name,
                            capability=required_capability,
                        )
                        return None
                    return tool_def
            self._logger.warning("tool_route_not_found", action=action)
            return None

        if required_capability:
            for tool_def in tools:
                if required_capability in tool_def.capabilities:
                    return tool_def
            self._logger.warning(
                "tool_route_no_capability_match",
                capability=required_capability,
            )
            return None

        return None

    async def validate(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Validate that a tool exists and accepts the supplied parameters."""
        if not self._registry.has_tool(tool_name):
            return False
        tool = self._registry.get_tool(tool_name)
        return await tool.validate_input(params)

    async def invoke(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Invoke a tool and propagate structured errors as exceptions."""
        return await self._invoker.invoke(tool_name, params, context)

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Execute a tool and always return a normalized :class:`ToolResult`.

        Tool-level failures are captured in the result rather than raised,
        making this safe for agent consumption.
        """
        ctx = context or ToolExecutionContext()
        log = self._logger.bind(
            tool_name=tool_name,
            agent_name=ctx.agent_name,
            task_id=ctx.task_id,
            correlation_id=ctx.correlation_id,
        )
        start = time.monotonic()
        try:
            result = await self._invoker.invoke(tool_name, params, ctx)
            log.info(
                "tool_router_execute_completed",
                success=result.success,
                duration_seconds=result.duration_seconds,
            )
            return result
        except ToolNotFoundError as exc:
            duration = time.monotonic() - start
            log.warning("tool_router_execute_not_found", error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
                duration_seconds=duration,
            )
        except ToolValidationError as exc:
            duration = time.monotonic() - start
            log.warning("tool_router_execute_validation_failed", error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
                duration_seconds=duration,
            )
        except ToolPermissionError as exc:
            duration = time.monotonic() - start
            log.warning("tool_router_execute_permission_denied", error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
                duration_seconds=duration,
            )
        except ToolTimeoutError as exc:
            duration = time.monotonic() - start
            log.error("tool_router_execute_timeout", error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
                duration_seconds=duration,
            )
        except ToolExecutionError as exc:
            duration = time.monotonic() - start
            log.error("tool_router_execute_failed", error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
                duration_seconds=duration,
            )
        except ToolError as exc:
            duration = time.monotonic() - start
            log.error("tool_router_execute_tool_error", error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
                duration_seconds=duration,
            )


__all__ = ["ToolRouter"]
