"""Focused tests for ToolRouter, registry, permissions, and timeouts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kodiak.tools.base import ToolAdapter
from kodiak.tools.builtin import ListDirTool, ReadFileTool, WriteFileTool, register_builtin_tools
from kodiak.tools.exceptions import (
    ToolNotFoundError,
    ToolPermissionError,
    ToolRegistrationError,
    ToolTimeoutError,
    ToolValidationError,
)
from kodiak.tools.models import ToolDefinition, ToolExecutionContext, ToolResult
from kodiak.tools.registry import ToolRegistry
from kodiak.tools.router import ToolRouter


class _EchoTool(ToolAdapter):
    def __init__(self, name: str = "echo", *, delay: float = 0.0) -> None:
        self._delay = delay
        self._definition = ToolDefinition(
            name=name,
            description="Echo inputs",
            input_schema={"type": "object", "required": ["message"]},
            capabilities=frozenset({"echo"}),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        return ToolResult(
            success=True,
            output={"message": inputs.get("message", "")},
            tool_name=self.definition.name,
        )


class _FailingTool(ToolAdapter):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="failing", description="Always fails")

    async def execute(
        self,
        inputs: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        raise RuntimeError("boom")


def test_tool_registration_and_lookup() -> None:
    registry = ToolRegistry()
    registry.register_tool(_EchoTool())
    assert registry.has_tool("echo")
    assert registry.get_metadata("echo").name == "echo"


def test_duplicate_registration_rejected() -> None:
    registry = ToolRegistry()
    registry.register_tool(_EchoTool())
    with pytest.raises(ToolRegistrationError):
        registry.register_tool(_EchoTool())


def test_deterministic_tool_listing() -> None:
    router = ToolRouter()
    router.register_tool(_EchoTool(name="beta"))
    router.register_tool(_EchoTool(name="alpha"))
    names = [tool.name for tool in router.list_tools()]
    assert names == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_valid_tool_execution(tmp_path: Path) -> None:
    sample = tmp_path / "hello.txt"
    sample.write_text("hello", encoding="utf-8")

    router = ToolRouter()
    router.register_tool(ReadFileTool(workspace_root=tmp_path))

    result = await router.execute("read_file", {"path": "hello.txt"})
    assert result.success is True
    assert result.output["content"] == "hello"


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_failure() -> None:
    router = ToolRouter()
    result = await router.execute("missing", {})
    assert result.success is False
    assert "not registered" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_unknown_tool_invoke_raises() -> None:
    router = ToolRouter()
    with pytest.raises(ToolNotFoundError):
        await router.invoke("missing", {})


@pytest.mark.asyncio
async def test_invalid_arguments_rejected() -> None:
    router = ToolRouter()
    router.register_tool(ReadFileTool())

    with pytest.raises(ToolValidationError):
        await router.invoke("read_file", {})

    result = await router.execute("read_file", {})
    assert result.success is False


@pytest.mark.asyncio
async def test_permission_rejection(tmp_path: Path) -> None:
    router = ToolRouter()
    router.register_tool(WriteFileTool(workspace_root=tmp_path))

    context = ToolExecutionContext(
        agent_name="tester",
        granted_capabilities=frozenset({"read_file"}),
    )
    with pytest.raises(ToolPermissionError):
        await router.invoke(
            "write_file",
            {"path": "out.txt", "content": "x"},
            context,
        )

    result = await router.execute(
        "write_file",
        {"path": "out.txt", "content": "x"},
        context,
    )
    assert result.success is False
    assert "denied" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_tool_timeout() -> None:
    router = ToolRouter()
    slow = _EchoTool(name="slow", delay=0.2)
    slow._definition = ToolDefinition(
        name="slow",
        description="slow echo",
        input_schema={"type": "object", "required": ["message"]},
        timeout_seconds=0.05,
    )
    router.register_tool(slow)

    with pytest.raises(ToolTimeoutError):
        await router.invoke("slow", {"message": "hi"})

    result = await router.execute("slow", {"message": "hi"})
    assert result.success is False
    assert "timed out" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_tool_execution_failure_propagation() -> None:
    router = ToolRouter()
    router.register_tool(_FailingTool())

    with pytest.raises(RuntimeError, match="boom"):
        await router.invoke("failing", {})

    result = await router.execute("failing", {})
    assert result.success is False
    assert result.error


@pytest.mark.asyncio
async def test_route_by_capability() -> None:
    router = ToolRouter()
    router.register_tool(ListDirTool())
    matched = router.route(required_capability="list_dir")
    assert matched is not None
    assert matched.name == "list_dir"


@pytest.mark.asyncio
async def test_register_builtin_tools(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_builtin_tools(registry, workspace_root=tmp_path)
    names = sorted(tool.name for tool in registry.list_tools())
    assert "read_file" in names
    assert "list_dir" in names
    assert "command_runner" in names


@pytest.mark.asyncio
async def test_validate_existing_tool() -> None:
    router = ToolRouter()
    router.register_tool(_EchoTool())
    assert await router.validate("echo", {"message": "ok"}) is True
    assert await router.validate("echo", {}) is False
    assert await router.validate("missing", {"message": "ok"}) is False
