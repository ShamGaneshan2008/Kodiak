"""Permission engine and security boundary checker for tools."""

from __future__ import annotations

import structlog

from kodiak.tools.exceptions import ToolPermissionError
from kodiak.tools.models import PermissionLevel, ToolDefinition, ToolExecutionContext

logger = structlog.get_logger(__name__)


class PermissionEngine:
    """Evaluates and enforces permission boundaries for tool execution.

    Supports permission levels: ALLOWED, DENIED, RESTRICTED, CONFIRMATION_REQUIRED.
    Does not automatically bypass security restrictions.
    """

    def __init__(self, default_level: PermissionLevel = PermissionLevel.ALLOWED) -> None:
        self._default_level = default_level
        self._logger = logger.bind(component="permission_engine")

    def check_permission(
        self,
        tool_def: ToolDefinition,
        context: ToolExecutionContext | None = None,
    ) -> PermissionLevel:
        """Evaluate permission level for executing a tool under context.

        Args:
            tool_def: Metadata of the target tool.
            context: Context containing caller identity, permissions, and granted capabilities.

        Returns:
            Resolved PermissionLevel (ALLOWED, DENIED, RESTRICTED, CONFIRMATION_REQUIRED).
        """
        # Explicit tool definition level DENIED
        if tool_def.permissions == PermissionLevel.DENIED:
            return PermissionLevel.DENIED

        if context is not None:
            # If context explicitly contains DENIED
            if PermissionLevel.DENIED in context.permissions:
                return PermissionLevel.DENIED

            # If tool is restricted or context requires restriction checks
            if tool_def.is_restricted or tool_def.permissions == PermissionLevel.RESTRICTED:
                # Caller must have at least one matching capability
                if tool_def.capabilities:
                    has_matching_capability = bool(
                        tool_def.capabilities & context.granted_capabilities
                    )
                    if not has_matching_capability:
                        self._logger.warning(
                            "permission_denied_missing_capability",
                            tool_name=tool_def.name,
                            agent_name=context.agent_name,
                            required=list(tool_def.capabilities),
                            granted=list(context.granted_capabilities),
                        )
                        return PermissionLevel.DENIED

        if tool_def.permissions == PermissionLevel.CONFIRMATION_REQUIRED:
            return PermissionLevel.CONFIRMATION_REQUIRED

        return tool_def.permissions

    def validate_or_raise(
        self,
        tool_def: ToolDefinition,
        context: ToolExecutionContext | None = None,
    ) -> None:
        """Validate permission and raise ToolPermissionError if denied.

        Args:
            tool_def: Metadata of the target tool.
            context: Optional execution context.

        Raises:
            ToolPermissionError: If execution is prohibited by permission settings.
        """
        level = self.check_permission(tool_def, context)
        if level == PermissionLevel.DENIED:
            agent = context.agent_name if context else "unknown"
            raise ToolPermissionError(
                f"Permission DENIED for agent {agent!r} executing tool {tool_def.name!r}."
            )


__all__ = ["PermissionEngine"]
