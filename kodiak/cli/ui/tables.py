"""Reusable Rich table components for the Kodiak CLI.

This module is a pure presentation layer: it builds `rich.table.Table`
objects from supplied row models. It contains no business logic, no
I/O, and no printing — callers own the data and are responsible for
rendering the returned tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rich.table import Table

__all__ = [
    "RepositoryAnalysisRow",
    "ReviewFindingRow",
    "ImplementationPlanRow",
    "TaskSummaryRow",
    "VersionRow",
    "RepositoryStatisticRow",
    "build_repository_analysis_table",
    "build_review_findings_table",
    "build_implementation_plan_table",
    "build_task_summary_table",
    "build_version_table",
    "build_repository_statistics_table",
]

_HEADER_STYLE: Final[str] = "bold bright_white"
_BORDER_STYLE: Final[str] = "bright_black"


def _base_table(title: str, *, caption: str | None = None) -> Table:
    """Construct a Table with consistent chrome.

    Args:
        title: The table's title text.
        caption: Optional caption text shown beneath the table.

    Returns:
        A configured, empty Rich Table (no columns or rows added).
    """
    return Table(
        title=title,
        caption=caption,
        header_style=_HEADER_STYLE,
        border_style=_BORDER_STYLE,
        title_style="bold bright_cyan",
        caption_style="dim",
        expand=True,
        show_lines=False,
    )


# --------------------------------------------------------------------------
# Repository Analysis Table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepositoryAnalysisRow:
    """A single row of repository analysis output.

    Attributes:
        path: File or module path being described.
        language: Detected programming language.
        lines_of_code: Total lines of code in the path.
        complexity: A human-readable complexity indicator (e.g. "Low",
            "Medium", "High", or a numeric score as a string).
        notes: Optional free-text notes about the path.
    """

    path: str
    language: str
    lines_of_code: int
    complexity: str
    notes: str = ""


def build_repository_analysis_table(
    rows: list[RepositoryAnalysisRow],
    *,
    title: str = "Repository Analysis",
) -> Table:
    """Build a table summarizing per-file repository analysis results.

    Args:
        rows: The repository analysis rows to render.
        title: The table title.

    Returns:
        A Rich Table populated with the supplied rows.
    """
    table = _base_table(title)
    table.add_column("Path", style="bright_white", no_wrap=False)
    table.add_column("Language", style="bright_cyan")
    table.add_column("LOC", style="bright_yellow", justify="right")
    table.add_column("Complexity", style="bright_magenta")
    table.add_column("Notes", style="dim")

    for row in rows:
        table.add_row(row.path, row.language, str(row.lines_of_code), row.complexity, row.notes)

    return table


# --------------------------------------------------------------------------
# Review Findings Table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReviewFindingRow:
    """A single code review finding.

    Attributes:
        severity: Severity label (e.g. "Critical", "Major", "Minor",
            "Info").
        file: The file path the finding applies to.
        line: The line number the finding applies to, if applicable.
        message: A description of the finding.
    """

    severity: str
    file: str
    message: str
    line: int | None = None


_SEVERITY_STYLES: Final[dict[str, str]] = {
    "critical": "bold bright_red",
    "major": "bright_red",
    "minor": "bright_yellow",
    "info": "bright_cyan",
}


def build_review_findings_table(
    rows: list[ReviewFindingRow],
    *,
    title: str = "Review Findings",
) -> Table:
    """Build a table summarizing code review findings.

    Args:
        rows: The review finding rows to render.
        title: The table title.

    Returns:
        A Rich Table populated with the supplied rows.
    """
    table = _base_table(title)
    table.add_column("Severity")
    table.add_column("File", style="bright_white")
    table.add_column("Line", justify="right", style="dim")
    table.add_column("Finding", style="bright_white")

    for row in rows:
        style = _SEVERITY_STYLES.get(row.severity.lower(), "bright_white")
        line_text = str(row.line) if row.line is not None else "-"
        table.add_row(f"[{style}]{row.severity}[/{style}]", row.file, line_text, row.message)

    return table


# --------------------------------------------------------------------------
# Implementation Plan Table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImplementationPlanRow:
    """A single step in an implementation plan.

    Attributes:
        step: The step number or identifier.
        action: A short description of the action to take.
        target: The file, module, or component the action targets.
        status: The step's current status (e.g. "Pending", "In
            Progress", "Done").
    """

    step: int
    action: str
    target: str
    status: str = "Pending"


_STATUS_STYLES: Final[dict[str, str]] = {
    "pending": "dim",
    "in progress": "bright_yellow",
    "done": "bright_green",
    "failed": "bright_red",
}


def build_implementation_plan_table(
    rows: list[ImplementationPlanRow],
    *,
    title: str = "Implementation Plan",
) -> Table:
    """Build a table summarizing an implementation plan.

    Args:
        rows: The implementation plan rows to render.
        title: The table title.

    Returns:
        A Rich Table populated with the supplied rows.
    """
    table = _base_table(title)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Action", style="bright_white")
    table.add_column("Target", style="bright_cyan")
    table.add_column("Status")

    for row in rows:
        style = _STATUS_STYLES.get(row.status.lower(), "bright_white")
        table.add_row(str(row.step), row.action, row.target, f"[{style}]{row.status}[/{style}]")

    return table


# --------------------------------------------------------------------------
# Task Summary Table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskSummaryRow:
    """A single row summarizing a completed or ongoing task.

    Attributes:
        name: The task's name.
        duration: A human-readable duration string (e.g. "3.2s").
        status: The task's outcome (e.g. "Success", "Failed", "Skipped").
        details: Optional additional detail about the task's outcome.
    """

    name: str
    duration: str
    status: str
    details: str = ""


_TASK_STATUS_STYLES: Final[dict[str, str]] = {
    "success": "bright_green",
    "failed": "bright_red",
    "skipped": "dim",
}


def build_task_summary_table(
    rows: list[TaskSummaryRow],
    *,
    title: str = "Task Summary",
) -> Table:
    """Build a table summarizing the outcomes of executed tasks.

    Args:
        rows: The task summary rows to render.
        title: The table title.

    Returns:
        A Rich Table populated with the supplied rows.
    """
    table = _base_table(title)
    table.add_column("Task", style="bright_white")
    table.add_column("Duration", justify="right", style="bright_cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    for row in rows:
        style = _TASK_STATUS_STYLES.get(row.status.lower(), "bright_white")
        table.add_row(row.name, row.duration, f"[{style}]{row.status}[/{style}]", row.details)

    return table


# --------------------------------------------------------------------------
# Version Table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VersionRow:
    """A single row describing a component's version.

    Attributes:
        component: The component name (e.g. "Kodiak", "Python",
            "LangGraph").
        version: The version string.
        status: Optional status indicator (e.g. "Up to date",
            "Outdated").
    """

    component: str
    version: str
    status: str = ""


def build_version_table(
    rows: list[VersionRow],
    *,
    title: str = "Version Information",
) -> Table:
    """Build a table summarizing component version information.

    Args:
        rows: The version rows to render.
        title: The table title.

    Returns:
        A Rich Table populated with the supplied rows.
    """
    table = _base_table(title)
    table.add_column("Component", style="bright_white")
    table.add_column("Version", style="bright_cyan")
    table.add_column("Status", style="dim")

    for row in rows:
        table.add_row(row.component, row.version, row.status)

    return table


# --------------------------------------------------------------------------
# Repository Statistics Table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepositoryStatisticRow:
    """A single repository-level statistic.

    Attributes:
        metric: The name of the statistic (e.g. "Total Files", "Total
            Commits", "Contributors").
        value: The statistic's value, formatted as a string.
    """

    metric: str
    value: str


def build_repository_statistics_table(
    rows: list[RepositoryStatisticRow],
    *,
    title: str = "Repository Statistics",
) -> Table:
    """Build a table summarizing repository-level statistics.

    Args:
        rows: The repository statistic rows to render.
        title: The table title.

    Returns:
        A Rich Table populated with the supplied rows.
    """
    table = _base_table(title)
    table.add_column("Metric", style="dim", justify="right")
    table.add_column("Value", style="bold bright_white")

    for row in rows:
        table.add_row(row.metric, row.value)

    return table
