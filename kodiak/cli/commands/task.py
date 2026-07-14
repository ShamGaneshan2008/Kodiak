"""CLI commands for Kodiak autonomous task workflows.

This module defines the ``kodiak task`` command group. It is strictly a
presentation-layer component and performs no analysis, planning, review,
or execution itself.

Autonomous task execution is NOT currently available from the CLI. The
backend's :class:`kodiak.services.task_service.TaskService` requires four
pre-built agent instances (``RepositoryAnalyzerAgent``, ``PlanningAgent``,
``ReviewAgent``, ``ExecutionAgent``) whose constructors are not part of
the CLI's known, verified surface, and it has no task persistence layer
(no list/show/cancel/delete). Rather than invent backend models or
constructors to paper over that gap, every command in this group reports
the limitation clearly.

Typical usage example:

    $ kodiak task run
    (reports that task execution is not available from the CLI)
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(name="task", help="Kodiak autonomous task workflows (currently unavailable via CLI).")

console = Console()

_UNAVAILABLE_MESSAGE = (
    "Autonomous task execution is not available from the CLI. The backend's "
    "TaskService requires pre-built agent instances (RepositoryAnalyzerAgent, "
    "PlanningAgent, ReviewAgent, ExecutionAgent) whose constructors are not "
    "exposed to or verified in the CLI layer, and there is no backend task "
    "persistence layer to list, show, cancel, or delete tasks."
)


def _render_unavailable_panel() -> None:
    """Render a panel explaining that task execution is unavailable."""
    console.print(
        Panel.fit(
            f"[bold yellow]{_UNAVAILABLE_MESSAGE}[/bold yellow]",
            title="[bold yellow]Feature Not Available[/bold yellow]",
            border_style="yellow",
        )
    )


@app.command("run")
def run() -> None:
    """Report that autonomous task execution is not available from the CLI.

    Raises:
        typer.Exit: Always, with code ``2``.
    """
    _render_unavailable_panel()
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()