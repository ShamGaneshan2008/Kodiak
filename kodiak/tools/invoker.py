"""Async-first tool invoker with validation, timeout, permissions, and observability."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from kodiak.tools.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)
from kodiak.tools.models import ToolDefinition, ToolExecutionContext, ToolResult
from kodiak.tools.permissions import PermissionEngine
from kodiak.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class ToolInvoker:
    """Invokes tools with parameter validation, permission enforcement, timeouts, and logging.

    Args:
        registry: ToolRegistry instance containing registered tools.
        permission_engine: Optional PermissionEngine for security enforcement.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_engine: PermissionEngine | None = None,
    ) -> None:
        self._registry = registry
        self._permission_engine = permission_engine or PermissionEngine()
        self._logger = logger.bind(component="tool_invoker")

    async def invoke(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Invoke a registered tool asynchronously.

        Args:
            tool_name: Stable identifier of the requested tool.
            inputs: Parameter dictionary for the tool invocation.
            context: Execution context containing agent identity, task ID, correlation ID, etc.

        Returns:
            Structured ToolResult instance.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            ToolValidationError: If input validation fails.
            ToolPermissionError: If permission checks fail.
            ToolTimeoutError: If execution times out.
            ToolExecutionError: If tool execution encounters an unhandled runtime error.
        """
        ctx = context or ToolExecutionContext()
        log = self._logger.bind(
            tool_name=tool_name,
            agent_name=ctx.agent_name,
            task_id=ctx.task_id,
            correlation_id=ctx.correlation_id,
        )

        start_time = time.monotonic()

        # 1. Resolve Tool
        if not self._registry.has_tool(tool_name):
            log.warning("tool_invocation_failed_not_found")
            raise ToolNotFoundError(f"Tool {tool_name!r} is not registered.")

        tool = self._registry.get_tool(tool_name)
        tool_def = tool.definition

        # 2. Permission Check
        self._permission_engine.validate_or_raise(tool_def, ctx)

        # 3. Input Validation
        is_valid = await tool.validate_input(inputs)
        if not is_valid:
            log.warning("tool_invocation_failed_validation", inputs=inputs)
            raise ToolValidationError(
                f"Input validation failed for tool {tool_name!r} with parameters: {inputs}"
            )

        # 4. Timeout Resolution
        timeout = ctx.timeout_seconds or tool_def.timeout_seconds

        log.info("tool_execution_started", timeout_seconds=timeout)

        # 5. Execution with Timeout & Observability
        try:
            if timeout and timeout > 0:
                result = await asyncio.wait_for(tool.execute(inputs, ctx), timeout=timeout)
            else:
                result = await tool.execute(inputs, ctx)

            duration = time.monotonic() - start_time

            # Construct finalized ToolResult preserving metadata
            final_result = ToolResult(
                success=result.success,
                output=result.output,
                error=result.error,
                execution_metadata=result.execution_metadata,
                duration_seconds=duration,
                tool_name=tool_name,
                correlation_id=ctx.correlation_id,
            )

            log.info(
                "tool_execution_completed",
                success=final_result.success,
                duration_seconds=duration,
            )

            return final_result

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            log.error("tool_execution_timeout", duration_seconds=duration, timeout_seconds=timeout)
            raise ToolTimeoutError(
                f"Tool {tool_name!r} execution timed out after {timeout} seconds."
            ) from None

        except ToolError:
            raise

        except Exception as exc:
            duration = time.monotonic() - start_time
            log.exception("tool_execution_unhandled_error", error=str(exc))
            raise ToolExecutionError(
                f"Unhandled error executing tool {tool_name!r}: {exc}"
            ) from exc


__all__ = ["ToolInvoker"]
