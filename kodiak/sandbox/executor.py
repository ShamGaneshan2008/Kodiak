import re
import shlex
import time

import structlog
from pydantic import BaseModel, Field

from kodiak.sandbox.docker_backend import DockerBackend, SandboxContainer

logger = structlog.get_logger(__name__)

DANGEROUS_COMMANDS = re.compile(
    r"(\b rm -rf \b|\b mkfs\. \b|\b dd if= \b|\b :(){ :|:& };: \b)",
    re.IGNORECASE,
)


class ExecutionRequest(BaseModel):
    command: str
    timeout_seconds: float = 30.0
    environment: dict[str, str] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0


class SandboxExecutor:
    def __init__(self, backend: DockerBackend) -> None:
        self._backend = backend

    def validate_command(self, command: str) -> bool:
        if DANGEROUS_COMMANDS.search(command):
            logger.warning("dangerous_command_blocked", command=command)
            return False
        if not command.strip():
            return False
        return True

    async def execute(
        self, container: SandboxContainer, request: ExecutionRequest
    ) -> ExecutionResult:
        if not self.validate_command(request.command):
            return ExecutionResult(success=False, exit_code=-1, stderr="Command validation failed")

        start = time.perf_counter()
        try:
            exit_code, stdout, stderr = await self._backend.execute_command(
                container, request.command
            )
            duration = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.exception("execution_failed", error=str(e))
            return ExecutionResult(success=False, exit_code=-1, stderr=str(e), duration_ms=duration)

    async def execute_python(
        self,
        container: SandboxContainer,
        script: str,
        timeout: float = 30.0,  # noqa: ASYNC109
    ) -> ExecutionResult:
        safe_script = shlex.quote(script)
        command = f"python3 -c {safe_script}"
        request = ExecutionRequest(command=command, timeout_seconds=timeout)
        return await self.execute(container, request)

    async def execute_shell(
        self,
        container: SandboxContainer,
        shell_command: str,
        timeout: float = 30.0,  # noqa: ASYNC109
    ) -> ExecutionResult:
        request = ExecutionRequest(command=shell_command, timeout_seconds=timeout)
        return await self.execute(container, request)
