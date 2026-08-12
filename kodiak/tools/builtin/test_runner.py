"""Safe test runner tool abstraction."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from kodiak.tools.base import ToolAdapter
from kodiak.tools.models import PermissionLevel, ToolDefinition, ToolExecutionContext, ToolResult


class TestRunnerTool(ToolAdapter):
    """Tool abstraction for executing test suites safely."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._definition = ToolDefinition(
            name="test_runner",
            description="Executes test targets safely and returns test results.",
            version="1.0.0",
            input_schema={
                "type": "object",
                "properties": {
                    "test_target": {"type": "string", "description": "Test path or expression"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
            },
            capabilities=frozenset({"test_execution", "run_tests"}),
            permissions=PermissionLevel.ALLOWED,
            timeout_seconds=60.0,
            is_async=True,
            is_safe=True,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        test_target = inputs.get("test_target", "tests")
        options = inputs.get("options", [])

        cmd = [sys.executable, "-m", "pytest", test_target] + options

        try:
            timeout = (context and context.timeout_seconds) or self.definition.timeout_seconds or 60.0
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self._workspace_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            success = proc.returncode == 0

            return ToolResult(
                success=success,
                output={
                    "returncode": proc.returncode,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "test_target": test_target,
                },
                error=None if success else f"Tests failed with returncode {proc.returncode}",
                tool_name=self.definition.name,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Test execution timed out after {timeout} seconds.",
                tool_name=self.definition.name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to execute test runner: {exc}",
                tool_name=self.definition.name,
            )


__all__ = ["TestRunnerTool"]
