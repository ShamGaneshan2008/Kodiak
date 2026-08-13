"""Built-in safe foundational tools for Kodiak."""

from __future__ import annotations

from pathlib import Path

from kodiak.tools.builtin.command import CommandExecutionTool
from kodiak.tools.builtin.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from kodiak.tools.builtin.test_runner import TestRunnerTool
from kodiak.tools.registry import ToolRegistry

__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "CommandExecutionTool",
    "TestRunnerTool",
    "register_builtin_tools",
]


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    workspace_root: Path | None = None,
) -> None:
    """Register the standard built-in tool set into a registry.

    Args:
        registry: Target tool registry.
        workspace_root: Workspace boundary for filesystem and command tools.
    """
    root = workspace_root or Path.cwd()
    registry.register_tool(ReadFileTool(workspace_root=root))
    registry.register_tool(WriteFileTool(workspace_root=root))
    registry.register_tool(ListDirTool(workspace_root=root))
    registry.register_tool(CommandExecutionTool(workspace_root=root))
    registry.register_tool(TestRunnerTool(workspace_root=root))
