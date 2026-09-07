"""Safe command execution tool abstraction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kodiak.security.output_filter import OutputFilter
from kodiak.tools.base import ToolAdapter
from kodiak.tools.models import PermissionLevel, ToolDefinition, ToolExecutionContext, ToolResult

# Safe list of binaries allowed for command execution by default
DEFAULT_ALLOWED_COMMANDS: set[str] = {
    "git",
    "python",
    "python3",
    "pytest",
    "npm",
    "ruff",
    "mypy",
    "ls",
    "dir",
    "echo",
}

_MAX_STREAM_CHARS = 20_000


def _decode_and_bound(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= _MAX_STREAM_CHARS:
        return text
    return text[:_MAX_STREAM_CHARS] + "\n[TRUNCATED]"


class CommandExecutionTool(ToolAdapter):
    """Safe command execution abstraction.

    Restricted tool that executes allowed subprocess commands without shell injection risks.
    """

    def __init__(
        self,
        allowed_commands: set[str] | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self._allowed_commands = allowed_commands or DEFAULT_ALLOWED_COMMANDS
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._definition = ToolDefinition(
            name="command_runner",
            description="Safely executes whitelisted CLI commands within workspace boundaries.",
            version="1.0.0",
            input_schema={
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Binary to execute"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                },
            },
            capabilities=frozenset({"command_execution", "terminal"}),
            permissions=PermissionLevel.RESTRICTED,
            timeout_seconds=30.0,
            is_async=True,
            is_safe=False,
            is_restricted=True,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        cmd_name = inputs.get("command", "").strip()
        args = inputs.get("args", [])
        raw_cwd = inputs.get("cwd")

        if not cmd_name:
            return ToolResult(
                success=False,
                error="Command name must not be empty.",
                tool_name=self.definition.name,
            )

        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            return ToolResult(
                success=False,
                error="Command args must be a list of strings.",
                tool_name=self.definition.name,
            )

        # Check allowed commands whitelist
        base_cmd = Path(cmd_name).name.lower().replace(".exe", "")
        if base_cmd not in self._allowed_commands:
            return ToolResult(
                success=False,
                error=f"Command {cmd_name!r} is not in the allowed command whitelist.",
                tool_name=self.definition.name,
            )

        cwd = self._workspace_root
        if raw_cwd:
            target_cwd = Path(raw_cwd)
            if not target_cwd.is_absolute():
                target_cwd = self._workspace_root / target_cwd
            try:
                resolved_cwd = target_cwd.resolve()
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Invalid cwd {raw_cwd!r}: {exc}",
                    tool_name=self.definition.name,
                )
            if (
                self._workspace_root not in resolved_cwd.parents
                and resolved_cwd != self._workspace_root
            ):
                return ToolResult(
                    success=False,
                    error=f"Access denied: cwd {raw_cwd!r} is outside workspace boundary.",
                    tool_name=self.definition.name,
                )
            cwd = resolved_cwd

        proc: asyncio.subprocess.Process | None = None
        try:
            # Run command directly without shell=True to eliminate injection vulnerability
            proc = await asyncio.create_subprocess_exec(
                cmd_name,
                *args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timeout = (
                (context and context.timeout_seconds) or self.definition.timeout_seconds or 30.0
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_text = await OutputFilter().redact_secrets(_decode_and_bound(stdout))
            stderr_text = await OutputFilter().redact_secrets(_decode_and_bound(stderr))

            success = proc.returncode == 0
            return ToolResult(
                success=success,
                output={
                    "returncode": proc.returncode,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "command": cmd_name,
                    "args": args,
                },
                error=None if success else f"Command exited with code {proc.returncode}",
                tool_name=self.definition.name,
            )
        except TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return ToolResult(
                success=False,
                error=f"Command execution timed out after {timeout} seconds.",
                tool_name=self.definition.name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Execution error for command {cmd_name!r}: {exc}",
                tool_name=self.definition.name,
            )


__all__ = ["CommandExecutionTool", "DEFAULT_ALLOWED_COMMANDS"]
