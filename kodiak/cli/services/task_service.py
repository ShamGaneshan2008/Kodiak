"""Service-layer facade for the deterministic Kodiak v1 task workflow."""

from __future__ import annotations

from pathlib import Path

from kodiak.orchestration.task_orchestrator import (
    InvalidRepositoryError,
    TaskOrchestrator,
    TaskRunResult,
)


class InvalidTaskInputError(ValueError):
    """Raised when a task instruction or repository path is invalid."""


class TaskWorkflowFailedError(RuntimeError):
    """Raised when the orchestrator returns an internal workflow error."""


class TaskService:
    """Expose the local v1 workflow without depending on Typer or Rich."""

    def __init__(self, orchestrator: TaskOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or TaskOrchestrator()

    def run_task(
        self,
        repository_path: str | Path,
        instruction: str,
        *,
        dry_run: bool = False,
        no_commit: bool = False,
    ) -> TaskRunResult:
        """Run one task and return its typed, JSON-serializable result."""
        try:
            result = self.orchestrator.run(
                instruction,
                repository_path,
                dry_run=dry_run,
                no_commit=no_commit,
            )
        except (InvalidRepositoryError, ValueError) as exc:
            raise InvalidTaskInputError(str(exc)) from exc
        if result.final_status == "error":
            raise TaskWorkflowFailedError(result.error or "Task workflow failed.")
        return result
