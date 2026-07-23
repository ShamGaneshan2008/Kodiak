"""CLI command for displaying Kodiak's current status.

This module defines the ``kodiak status`` command. It is strictly a
presentation-layer component: it performs no environment inspection,
network calls, or git operations itself. All of that work is delegated
to :class:`kodiak.services.status_service.StatusService`.

Typical usage example:

    $ kodiak status
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kodiak.services.status_service import StatusService, StatusUnavailableError

app = typer.Typer(
    name="status",
    help="Display the current Kodiak session and environment status.",
)

console = Console()


def _connection_style(is_connected: bool) -> str:
    """Return a Rich color style for a boolean connection state.

    Args:
        is_connected: Whether the connection is currently active.

    Returns:
        A Rich style string, either ``"green"`` or ``"red"``.
    """
    return "green" if is_connected else "red"


def _render_status_table(status: StatusSnapshot) -> Table:  # noqa: F821
    """Build a Rich table summarizing the current status snapshot.

    Args:
        status: The status data returned by :class:`StatusService`.

    Returns:
        A populated :class:`rich.table.Table` ready for printing.
    """
    table = Table(show_header=False, expand=True, padding=(0, 1))
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Logged-in user", status.username or "[dim]Not logged in[/dim]")
    table.add_row("Current project", status.project_name or "[dim]None[/dim]")
    table.add_row("Current task", status.current_task or "[dim]None[/dim]")
    table.add_row("Active agent", status.active_agent or "[dim]None[/dim]")

    backend_style = _connection_style(status.backend_connected)
    backend_label = "Connected" if status.backend_connected else "Disconnected"
    table.add_row(
        "Backend connection",
        f"[{backend_style}]{backend_label}[/{backend_style}]",
    )

    redis_style = _connection_style(status.redis_connected)
    redis_label = "Connected" if status.redis_connected else "Disconnected"
    table.add_row(
        "Redis status",
        f"[{redis_style}]{redis_label}[/{redis_style}]",
    )

    table.add_row("Git repository", status.git_repository or "[dim]None[/dim]")
    table.add_row("Git branch", status.git_branch or "[dim]None[/dim]")
    table.add_row("Project language", status.project_language or "[dim]Unknown[/dim]")
    table.add_row("Framework", status.framework or "[dim]Unknown[/dim]")
    table.add_row("Package manager", status.package_manager or "[dim]Unknown[/dim]")

    return table


def _render_error_panel(message: str) -> None:
    """Render a friendly error panel.

    Args:
        message: Human-readable explanation of what went wrong.
    """
    console.print(
        Panel.fit(
            f"[bold red]{message}[/bold red]",
            title="[bold red]Status Unavailable[/bold red]",
            border_style="red",
        )
    )


@app.command()
def status() -> None:
    """Display the current Kodiak session and environment status.

    This command delegates all data gathering to :class:`StatusService`
    and presents the resulting snapshot in a Rich table wrapped in a
    colored panel. On failure, a descriptive error panel is displayed
    and the process exits with a non-zero status code.

    Raises:
        typer.Exit: With code ``0`` on success or ``1`` if the status
            snapshot could not be retrieved.
    """
    status_service = StatusService()

    try:
        with console.status("[bold cyan]Gathering Kodiak status...[/bold cyan]", spinner="dots"):
            snapshot = status_service.get_status()

    except StatusUnavailableError as exc:
        _render_error_panel(f"Kodiak could not retrieve the current status.\n{exc}")
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel(f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=1) from exc

    table = _render_status_table(snapshot)
    console.print(
        Panel(
            table,
            title="[bold cyan]Kodiak Status[/bold cyan]",
            border_style="cyan",
        )
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
