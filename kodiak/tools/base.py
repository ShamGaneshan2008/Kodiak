"""Abstract base class and protocol for Kodiak tool adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from kodiak.tools.models import ToolDefinition, ToolExecutionContext, ToolResult

logger = structlog.get_logger(__name__)


class ToolAdapter(ABC):
    """Abstract interface and contract for implementing tools in Kodiak.

    Tools implement this adapter to expose metadata, input validation,
    async execution, and structured output formatting.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the strongly typed metadata definition for this tool."""

    async def validate_input(self, inputs: dict[str, Any]) -> bool:
        """Validate input parameters before tool execution.

        Default implementation verifies required parameters declared in
        definition.input_schema if present.

        Args:
            inputs: Parameter dictionary passed to the tool.

        Returns:
            True if inputs are valid, False otherwise.
        """
        schema = self.definition.input_schema
        if not schema:
            return True

        required_keys = schema.get("required", [])
        if isinstance(required_keys, (list, tuple, set)):
            for key in required_keys:
                if key not in inputs:
                    logger.warning(
                        "tool_input_missing_required_field",
                        tool_name=self.definition.name,
                        field=key,
                    )
                    return False

        return True

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Execute the tool logic asynchronously.

        Args:
            inputs: Validated parameter map for the tool.
            context: Execution context containing caller details and permissions.

        Returns:
            Structured ToolResult instance.
        """


__all__ = ["ToolAdapter"]
