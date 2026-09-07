"""Safety and resource-bound tests for sandbox command execution."""

from __future__ import annotations

import asyncio

import pytest

from kodiak.sandbox.docker_backend import SandboxContainer
from kodiak.sandbox.executor import ExecutionRequest, SandboxExecutor


class _Backend:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[str] = []

    async def execute_command(
        self, container: SandboxContainer, command: str
    ) -> tuple[int, str, str]:
        self.calls.append(command)
        await asyncio.sleep(self.delay)
        return 0, "ok", ""


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "sudo rm -fr /workspace", "mkfs.ext4 /dev/x", "dd if=/dev/zero of=x"],
)
async def test_dangerous_commands_are_rejected_before_backend(command: str) -> None:
    backend = _Backend()
    executor = SandboxExecutor(backend)  # type: ignore[arg-type]

    result = await executor.execute(
        SandboxContainer(container_id="test"), ExecutionRequest(command=command)
    )

    assert not result.success
    assert result.stderr == "Command validation failed"
    assert backend.calls == []


async def test_command_timeout_is_bounded_and_reported() -> None:
    backend = _Backend(delay=0.1)
    executor = SandboxExecutor(backend)  # type: ignore[arg-type]

    result = await executor.execute(
        SandboxContainer(container_id="test"),
        ExecutionRequest(command="python -V", timeout_seconds=0.01),
    )

    assert not result.success
    assert result.exit_code == -1
    assert result.stderr == "Command timed out after 0.01 seconds"
