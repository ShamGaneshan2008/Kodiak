"""Command verification via ToolRouter."""

from __future__ import annotations

import time

from kodiak.orchestration.verification.base import Verifier
from kodiak.orchestration.verification.models import (
    VerificationContext,
    VerificationEvidence,
    VerificationStatus,
)
from kodiak.tools.models import ToolExecutionContext
from kodiak.tools.router import ToolRouter


def _summarize(text: str | None, limit: int = 500) -> str | None:
    if not text:
        return None
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 3] + "..."


class CommandVerifier(Verifier):
    """Run allowed validation commands through ToolRouter."""

    name = "command"

    def __init__(self, tool_router: ToolRouter | None = None) -> None:
        self._tool_router = tool_router

    def applies(self, context: VerificationContext) -> bool:
        return bool(context.success_criteria.get("commands"))

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        start = time.monotonic()
        commands = context.success_criteria.get("commands", [])
        if not isinstance(commands, list) or not commands:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.INCONCLUSIVE,
                duration_seconds=time.monotonic() - start,
                message="No commands configured for command verification.",
            )

        if self._tool_router is None:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.INCONCLUSIVE,
                duration_seconds=time.monotonic() - start,
                message="Command verification requested but ToolRouter is not configured.",
            )

        last_failure: VerificationEvidence | None = None
        for entry in commands:
            if isinstance(entry, str):
                command = entry
                args: list[str] = []
            elif isinstance(entry, dict):
                command = str(entry.get("command", ""))
                args = [str(arg) for arg in entry.get("args", [])]
            else:
                continue

            if not command:
                continue

            tool_context = ToolExecutionContext(
                agent_name="verification",
                task_id=str(context.task.id),
                granted_capabilities=frozenset({"command_execution", "terminal"}),
                timeout_seconds=(entry.get("timeout_seconds") if isinstance(entry, dict) else None),
            )
            result = await self._tool_router.execute(
                "command_runner",
                {"command": command, "args": args},
                tool_context,
            )
            command_label = " ".join([command, *args]).strip()
            stdout = _summarize(result.output.get("stdout") if result.output else None)
            stderr = _summarize(result.output.get("stderr") if result.output else None)
            exit_code = result.output.get("returncode") if result.output else None

            if not result.success:
                last_failure = VerificationEvidence(
                    verifier=self.name,
                    status=VerificationStatus.FAILED,
                    duration_seconds=time.monotonic() - start,
                    message=result.error or f"Command failed: {command_label}",
                    command=command_label,
                    exit_code=exit_code,
                    stdout_summary=stdout,
                    stderr_summary=stderr,
                )
                break

        duration = time.monotonic() - start
        if last_failure is not None:
            return VerificationEvidence(
                verifier=last_failure.verifier,
                status=last_failure.status,
                duration_seconds=duration,
                message=last_failure.message,
                command=last_failure.command,
                exit_code=last_failure.exit_code,
                stdout_summary=last_failure.stdout_summary,
                stderr_summary=last_failure.stderr_summary,
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.VERIFIED,
            duration_seconds=duration,
            message="All configured validation commands passed.",
            metadata={"commands_run": len(commands)},
        )


__all__ = ["CommandVerifier"]
