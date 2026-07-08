
"""Reusable Rich progress components for the Kodiak CLI.

This module is a pure presentation layer: it provides factory functions
and context managers that build `rich.progress.Progress` instances and
tasks for common Kodiak workflow stages (repository scanning, AI
planning, code generation, test execution, pull request creation) plus
a generic spinner helper. Nothing here contains business logic — callers
drive the actual work and simply update the returned progress handles.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

__all__ = [
    "ProgressHandle",
    "spinner_progress",
    "repository_scan_progress",
    "ai_planning_progress",
    "code_generation_progress",
    "test_execution_progress",
    "pull_request_progress",
    "simple_spinner",
]

console: Final[Console] = Console()

_SPINNER_STYLE: Final[str] = "bright_yellow"


@dataclass(slots=True)
class ProgressHandle:
    """A handle bundling a Rich Progress instance with its primary task.

    Attributes:
        progress: The underlying Rich Progress instance.
        task_id: The ID of the primary task created for this stage.
    """

    progress: Progress
    task_id: TaskID

    def update(
        self,
        *,
        advance: float | None = None,
        completed: float | None = None,
        description: str | None = None,
        total: float | None = None,
    ) -> None:
        """Update the primary task's progress state.

        Args:
            advance: Amount to increment the completed count by.
            completed: Absolute completed count to set.
            description: New description text for the task, if any.
            total: New total count for the task, if any.
        """
        kwargs: dict[str, object] = {}
        if advance is not None:
            kwargs["advance"] = advance
        if completed is not None:
            kwargs["completed"] = completed
        if description is not None:
            kwargs["description"] = description
        if total is not None:
            kwargs["total"] = total
        self.progress.update(self.task_id, **kwargs)

    def add_subtask(self, description: str, *, total: float | None = 100.0) -> TaskID:
        """Add an additional task to the same progress instance.

        Args:
            description: Description text for the new subtask.
            total: Total units of work for the subtask. Use ``None`` for
                an indeterminate task.

        Returns:
            The TaskID of the newly created subtask.
        """
        return self.progress.add_task(description, total=total)


def _build_progress(
    *,
    spinner_style: str = _SPINNER_STYLE,
    show_bar: bool = True,
    show_counts: bool = False,
) -> Progress:
    """Construct a Progress instance with a consistent column layout.

    Args:
        spinner_style: Rich color/style string for the spinner column.
        show_bar: Whether to include a progress bar column.
        show_counts: Whether to include an "M of N" completed column.

    Returns:
        A configured Rich Progress instance (not yet started).
    """
    columns: list[object] = [
        SpinnerColumn(style=spinner_style),
        TextColumn("[progress.description]{task.description}"),
    ]
    if show_bar:
        columns.append(BarColumn(bar_width=None))
    if show_counts:
        columns.append(MofNCompleteColumn())
    columns.append(TimeElapsedColumn())

    return Progress(*columns, console=console, transient=False)


@contextmanager
def spinner_progress(
    description: str,
    *,
    spinner_style: str = _SPINNER_STYLE,
) -> Iterator[ProgressHandle]:
    """Provide an indeterminate spinner for a single ongoing operation.

    Args:
        description: The label describing what is in progress.
        spinner_style: Rich color/style string for the spinner column.

    Yields:
        A ProgressHandle wrapping the running Progress and its task.
    """
    progress = _build_progress(spinner_style=spinner_style, show_bar=False)
    with progress:
        task_id = progress.add_task(description, total=None)
        yield ProgressHandle(progress=progress, task_id=task_id)


@contextmanager
def repository_scan_progress(
    description: str = "Scanning repository...",
    *,
    total: float | None = 100.0,
) -> Iterator[ProgressHandle]:
    """Provide a progress display for the repository scan stage.

    Args:
        description: Initial description text for the scan task.
        total: Total units of work, or ``None`` for indeterminate.

    Yields:
        A ProgressHandle for tracking repository scan progress.
    """
    progress = _build_progress(spinner_style="bright_cyan", show_counts=total is not None)
    with progress:
        task_id = progress.add_task(description, total=total)
        yield ProgressHandle(progress=progress, task_id=task_id)


@contextmanager
def ai_planning_progress(
    description: str = "Planning with AI...",
    *,
    total: float | None = None,
) -> Iterator[ProgressHandle]:
    """Provide a progress display for the AI planning stage.

    Args:
        description: Initial description text for the planning task.
        total: Total units of work, or ``None`` for indeterminate
            (planning is typically indeterminate).

    Yields:
        A ProgressHandle for tracking AI planning progress.
    """
    progress = _build_progress(
        spinner_style="bright_magenta",
        show_bar=total is not None,
        show_counts=False,
    )
    with progress:
        task_id = progress.add_task(description, total=total)
        yield ProgressHandle(progress=progress, task_id=task_id)


@contextmanager
def code_generation_progress(
    description: str = "Generating code...",
    *,
    total: float | None = 100.0,
) -> Iterator[ProgressHandle]:
    """Provide a progress display for the code generation stage.

    Args:
        description: Initial description text for the generation task.
        total: Total units of work (e.g. number of files), or ``None``
            for indeterminate.

    Yields:
        A ProgressHandle for tracking code generation progress.
    """
    progress = _build_progress(spinner_style="bright_green", show_counts=total is not None)
    with progress:
        task_id = progress.add_task(description, total=total)
        yield ProgressHandle(progress=progress, task_id=task_id)


@contextmanager
def test_execution_progress(
    description: str = "Running tests...",
    *,
    total: float | None = 100.0,
) -> Iterator[ProgressHandle]:
    """Provide a progress display for the test execution stage.

    Args:
        description: Initial description text for the test run task.
        total: Total units of work (e.g. number of tests), or ``None``
            for indeterminate.

    Yields:
        A ProgressHandle for tracking test execution progress.
    """
    progress = _build_progress(spinner_style="bright_blue", show_counts=total is not None)
    with progress:
        task_id = progress.add_task(description, total=total)
        yield ProgressHandle(progress=progress, task_id=task_id)


@contextmanager
def pull_request_progress(
    description: str = "Creating pull request...",
    *,
    total: float | None = None,
) -> Iterator[ProgressHandle]:
    """Provide a progress display for the pull request creation stage.

    Args:
        description: Initial description text for the PR task.
        total: Total units of work, or ``None`` for indeterminate
            (PR creation is typically indeterminate).

    Yields:
        A ProgressHandle for tracking pull request creation progress.
    """
    progress = _build_progress(
        spinner_style="bright_yellow",
        show_bar=total is not None,
        show_counts=False,
    )
    with progress:
        task_id = progress.add_task(description, total=total)
        yield ProgressHandle(progress=progress, task_id=task_id)


@contextmanager
def simple_spinner(description: str) -> Iterator[None]:
    """Provide a minimal, no-frills spinner for quick operations.

    Args:
        description: The label describing what is in progress.

    Yields:
        None. This helper is intended for simple ``with`` blocks where
        no manual progress updates are needed.
    """
    progress = Progress(
        SpinnerColumn(style=_SPINNER_STYLE),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )
    with progress:
        progress.add_task(description, total=None)
        yield
