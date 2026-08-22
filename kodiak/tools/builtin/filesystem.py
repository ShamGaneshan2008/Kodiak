"""Safe foundational filesystem tools for Kodiak."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kodiak.tools.base import ToolAdapter
from kodiak.tools.models import PermissionLevel, ToolDefinition, ToolExecutionContext, ToolResult


def _is_path_safe(target_path: Path, base_dir: Path | None = None) -> bool:
    """Ensure path does not attempt directory traversal outside workspace."""
    base = (base_dir or Path.cwd()).resolve()
    try:
        resolved = target_path.resolve()
        return resolved == base or base in resolved.parents
    except Exception:
        return False


class ReadFileTool(ToolAdapter):
    """Tool for reading text files safely."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._definition = ToolDefinition(
            name="read_file",
            description="Reads the text contents of a specified file within workspace.",
            version="1.0.0",
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute file path"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
            },
            output_schema={"type": "object", "properties": {"content": {"type": "string"}}},
            capabilities=frozenset({"read_file", "filesystem"}),
            permissions=PermissionLevel.ALLOWED,
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
        raw_path = inputs.get("path", "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._workspace_root / path

        if not _is_path_safe(path, self._workspace_root):
            return ToolResult(
                success=False,
                error=f"Access denied: path {raw_path!r} is outside workspace boundary.",
                tool_name=self.definition.name,
            )

        if not path.exists() or not path.is_file():
            return ToolResult(
                success=False,
                error=f"File not found: {raw_path!r}",
                tool_name=self.definition.name,
            )

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            start_line = inputs.get("start_line")
            end_line = inputs.get("end_line")

            if start_line is not None or end_line is not None:
                start = max((start_line or 1) - 1, 0)
                end = end_line if end_line is not None else len(lines)
                content = "".join(lines[start:end])

            return ToolResult(
                success=True,
                output={"content": content, "path": str(path), "total_lines": len(lines)},
                tool_name=self.definition.name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to read file {raw_path!r}: {exc}",
                tool_name=self.definition.name,
            )


class WriteFileTool(ToolAdapter):
    """Tool for writing text files safely within workspace."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._definition = ToolDefinition(
            name="write_file",
            description="Writes text contents to a specified file within workspace.",
            version="1.0.0",
            input_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
            capabilities=frozenset({"write_file", "filesystem"}),
            permissions=PermissionLevel.RESTRICTED,
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
        raw_path = inputs.get("path", "")
        content = inputs.get("content", "")

        path = Path(raw_path)
        if not path.is_absolute():
            path = self._workspace_root / path

        if not _is_path_safe(path, self._workspace_root):
            return ToolResult(
                success=False,
                error=f"Access denied: path {raw_path!r} is outside workspace boundary.",
                tool_name=self.definition.name,
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output={"path": str(path), "bytes_written": len(content.encode("utf-8"))},
                tool_name=self.definition.name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to write file {raw_path!r}: {exc}",
                tool_name=self.definition.name,
            )


class ListDirTool(ToolAdapter):
    """Tool for listing directory contents safely."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._definition = ToolDefinition(
            name="list_dir",
            description="Lists entries in a directory within workspace.",
            version="1.0.0",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
            capabilities=frozenset({"list_dir", "filesystem"}),
            permissions=PermissionLevel.ALLOWED,
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
        raw_path = inputs.get("path", ".")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._workspace_root / path

        if not _is_path_safe(path, self._workspace_root):
            return ToolResult(
                success=False,
                error=f"Access denied: path {raw_path!r} is outside workspace boundary.",
                tool_name=self.definition.name,
            )

        if not path.exists() or not path.is_dir():
            return ToolResult(
                success=False,
                error=f"Directory not found: {raw_path!r}",
                tool_name=self.definition.name,
            )

        try:
            entries = []
            for entry in path.iterdir():
                entries.append(
                    {
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size": entry.stat().st_size if entry.is_file() else 0,
                    }
                )
            return ToolResult(
                success=True,
                output={"path": str(path), "entries": entries},
                tool_name=self.definition.name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to list directory {raw_path!r}: {exc}",
                tool_name=self.definition.name,
            )


__all__ = ["ReadFileTool", "WriteFileTool", "ListDirTool"]
