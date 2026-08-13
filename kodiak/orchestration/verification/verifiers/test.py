"""Test execution verification via ToolRouter."""

from __future__ import annotations

import time
from typing import Any

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


class TestVerifier(Verifier):
    """Run configured tests through the existing ToolRouter boundary."""

    name = "test"

    def __init__(self, tool_router: ToolRouter | None = None) -> None:
        self._tool_router = tool_router

    def applies(self, context: VerificationContext) -> bool:
        return bool(context.success_criteria.get("run_tests"))

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        start = time.monotonic()
        config = context.success_criteria.get("run_tests", {})
        if not isinstance(config, dict):
            config = {"test_target": str(config)}

        if self._tool_router is None:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.INCONCLUSIVE,
                duration_seconds=time.monotonic() - start,
                message="Test verification requested but ToolRouter is not configured.",
            )

        test_target = config.get("test_target", "tests")
        options = config.get("options", [])
        inputs: dict[str, Any] = {"test_target": test_target, "options": options}

        tool_context = ToolExecutionContext(
            agent_name="verification",
            task_id=str(context.task.id),
            granted_capabilities=frozenset({"run_tests", "test_execution"}),
            timeout_seconds=config.get("timeout_seconds"),
        )

        result = await self._tool_router.execute("test_runner", inputs, tool_context)
        duration = time.monotonic() - start
        stdout = _summarize(result.output.get("stdout") if result.output else None)
        stderr = _summarize(result.output.get("stderr") if result.output else None)
        exit_code = result.output.get("returncode") if result.output else None

        if result.success:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.VERIFIED,
                duration_seconds=duration,
                message=f"Tests passed for target {test_target!r}.",
                command=f"pytest {test_target}",
                exit_code=exit_code,
                stdout_summary=stdout,
                stderr_summary=stderr,
                metadata={"test_target": test_target},
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.FAILED,
            duration_seconds=duration,
            message=result.error or f"Tests failed for target {test_target!r}.",
            command=f"pytest {test_target}",
            exit_code=exit_code,
            stdout_summary=stdout,
            stderr_summary=stderr,
            metadata={"test_target": test_target},
        )


__all__ = ["TestVerifier"]
