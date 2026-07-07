"""CLI commands for managing Kodiak tasks.

This module defines the ``kodiak task`` command group (``create``,
``list``, ``show``, ``cancel``, ``delete``). It is strictly a
presentation-layer component: it performs no database access and no AI
calls itself. All business logic is delegated to
:class:`kodiak.services.task_service.TaskService`.

Typical usage example:

    $ kodiak task create --title "Fix bug" --priority high
    $ kodiak task list
    $ kodiak task show <task_id>
    $ kodiak task cancel <task_id>
    $ kodiak task delete <task_id>
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kodiak.services.task_service import (
    TaskNotFoundError,
    TaskService,
    TaskServiceError,
)

app = typer.Typer(name="task", help="Create and manage Kodiak tasks.")

console = Console()

_PRIORITY_STYLES = {
    "low": "dim",
    "medium": "yellow",
    "high": "bold red",
}

_STATUS_STYLES = {
    "pending": "yellow",
    "running": "cyan",
    "completed": "green",
    "cancelled": "dim",
    "failed": "red",
}


def _render_success_panel(title: str, message: str) -> None:
    """Render a success panel.

    Args:
        title: Short title describing the completed action.
        message: Human-readable success message.
    """
    console.print(
        Panel.fit(
            f"[bold green]{message}[/bold green]",
            title=f"[bold green]{title}[/bold green]",
            border_style="green",
        )
    )


def _render_error_panel(title: str, message: str) -> None:
    """Render a friendly error panel.

    Args:
        title: Short title describing the failure category.
        message: Human-readable explanation of what went wrong.
    """
    console.print(
        Panel.fit(
            f"[bold red]{message}[/bold red]",
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
        )
    )


def _render_task_panel(task: "Task") -> None:  # noqa: F821
    """Render a detailed panel for a single task.

    Args:
        task: The task data returned by :class:`TaskService`.
    """
    priority_style = _PRIORITY_STYLES.get(task.priority.lower(), "white")
    status_style = _STATUS_STYLES.get(task.status.lower(), "white")

    table = Table(show_header=False, expand=True, padding=(0, 1))
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("ID", task.id)
    table.add_row("Title", task.title)
    table.add_row("Description", task.description or "[dim]None[/dim]")
    table.add_row("Priority", f"[{priority_style}]{task.priority}[/{priority_style}]")
    table.add_row("Status", f"[{status_style}]{task.status}[/{status_style}]")
    table.add_row("Repository", task.repository or "[dim]None[/dim]")
    table.add_row("Branch", task.branch or "[dim]None[/dim]")
    table.add_row("Created at", str(task.created_at))

    console.print(
        Panel(table, title=f"[bold cyan]Task {task.id}[/bold cyan]", border_style="cyan")
    )


@app.command("create")
def create(
    title: str = typer.Option(..., "--title", "-t", help="Short title for the task."),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Detailed description of the task."
    ),
    priority: str = typer.Option(
        "medium",
        "--priority",
        "-p",
        help="Task priority: low, medium, or high.",
    ),
    repository: Optional[str] = typer.Option(
        None, "--repository", "-r", help="Target repository for the task."
    ),
    branch: Optional[str] = typer.Option(
        None, "--branch", "-b", help="Target branch for the task."
    ),
) -> None:
    """Create a new Kodiak task.

    Collects task attributes from the command line and delegates
    creation to :class:`TaskService`. Displays a success panel with the
    created task's details, or a friendly error panel on failure.

    Args:
        title: Short title for the task.
        description: Detailed description of the task.
        priority: Task priority, one of ``low``, ``medium``, ``high``.
        repository: Target repository for the task.
        branch: Target branch for the task.

    Raises:
        typer.Exit: With code ``0`` on success, ``1`` on validation or
            service failure, or ``2`` on any unexpected error.
    """
    task_service = TaskService()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Creating task...[/bold cyan]"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("create", total=None)
            task = task_service.create_task(
                title=title,
                description=description,
                priority=priority,
                repository=repository,
                branch=branch,
            )

    except TaskServiceError as exc:
        _render_error_panel("Task Creation Failed", str(exc))
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel("Unexpected Error", f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=2) from exc

    _render_success_panel("Task Created", f"Task '{task.title}' created successfully.")
    _render_task_panel(task)
    raise typer.Exit(code=0)


@app.command("list")
def list_tasks() -> None:
    """List all Kodiak tasks.

    Delegates retrieval to :class:`TaskService` and displays the results
    in a Rich table. Shows an informational panel if no tasks exist.

    Raises:
        typer.Exit: With code ``0`` on success or ``1`` on failure.
    """
    task_service = TaskService()

    try:
        with console.status("[bold cyan]Loading tasks...[/bold cyan]", spinner="dots"):
            tasks = task_service.list_tasks()

    except TaskServiceError as exc:
        _render_error_panel("Could Not Load Tasks", str(exc))
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel("Unexpected Error", f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=1) from exc

    if not tasks:
        console.print(
            Panel.fit(
                "[bold cyan]No tasks found.[/bold cyan]",
                title="[bold cyan]Kodiak Tasks[/bold cyan]",
                border_style="cyan",
            )
        )
        raise typer.Exit(code=0)

    table = Table(title="Kodiak Tasks", expand=True)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Priority")
    table.add_column("Status")
    table.add_column("Repository")
    table.add_column("Branch")

    for task in tasks:
        priority_style = _PRIORITY_STYLES.get(task.priority.lower(), "white")
        status_style = _STATUS_STYLES.get(task.status.lower(), "white")
        table.add_row(
            task.id,
            task.title,
            f"[{priority_style}]{task.priority}[/{priority_style}]",
            f"[{status_style}]{task.status}[/{status_style}]",
            task.repository or "[dim]None[/dim]",
            task.branch or "[dim]None[/dim]",
        )

    console.print(table)
    raise typer.Exit(code=0)


@app.command("show")
def show(task_id: str = typer.Argument(..., help="ID of the task to show.")) -> None:
    """Show detailed information about a single task.

    Args:
        task_id: ID of the task to display.

    Raises:
        typer.Exit: With code ``0`` on success, ``1`` if the task is not
            found, or ``2`` on any unexpected error.
    """
    task_service = TaskService()

    try:
        with console.status("[bold cyan]Fetching task...[/bold cyan]", spinner="dots"):
            task = task_service.get_task(task_id)

    except TaskNotFoundError as exc:
        _render_error_panel("Task Not Found", f"No task found with ID '{task_id}'.")
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel("Unexpected Error", f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=2) from exc

    _render_task_panel(task)
    raise typer.Exit(code=0)


@app.command("cancel")
def cancel(task_id: str = typer.Argument(..., help="ID of the task to cancel.")) -> None:
    """Cancel a running or pending task.

    Args:
        task_id: ID of the task to cancel.

    Raises:
        typer.Exit: With code ``0`` on success, ``1`` if the task is not
            found or cannot be cancelled, or ``2`` on any unexpected error.
    """
    task_service = TaskService()

    if not typer.confirm(f"Cancel task '{task_id}'?", default=False):
        console.print("[yellow]Cancellation aborted.[/yellow]")
        raise typer.Exit(code=0)

    try:
        with console.status(
            "[bold cyan]Cancelling task...[/bold cyan]", spinner="dots"
        ):
            task_service.cancel_task(task_id)

    except TaskNotFoundError as exc:
        _render_error_panel("Task Not Found", f"No task found with ID '{task_id}'.")
        raise typer.Exit(code=1) from exc

    except TaskServiceError as exc:
        _render_error_panel("Cancellation Failed", str(exc))
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel("Unexpected Error", f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=2) from exc

    _render_success_panel("Task Cancelled", f"Task '{task_id}' has been cancelled.")
    raise typer.Exit(code=0)


@app.command("delete")
def delete(
    task_id: str = typer.Argument(..., help="ID of the task to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Permanently delete a task.

    Args:
        task_id: ID of the task to delete.
        yes: If ``True``, skips the interactive confirmation prompt.

    Raises:
        typer.Exit: With code ``0`` on success, ``1`` if the task is not
            found or cannot be deleted, or ``2`` on any unexpected error.
    """
    task_service = TaskService()

    if not yes and not typer.confirm(
        f"Permanently delete task '{task_id}'?", default=False
    ):
        console.print("[yellow]Deletion aborted.[/yellow]")
        raise typer.Exit(code=0)

    try:
        with console.status("[bold cyan]Deleting task...[/bold cyan]", spinner="dots"):
            task_service.delete_task(task_id)

    except TaskNotFoundError as exc:
        _render_error_panel("Task Not Found", f"No task found with ID '{task_id}'.")
        raise typer.Exit(code=1) from exc

    except TaskServiceError as exc:
        _render_error_panel("Deletion Failed", str(exc))
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel("Unexpected Error", f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=2) from exc

    _render_success_panel("Task Deleted", f"Task '{task_id}' has been permanently deleted.")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()