"""CLI command for logging a Kodiak user out.

This module defines the ``kodiak logout`` command. It is strictly a
presentation-layer component: it performs no credential removal or
storage logic itself. All of that work is delegated to
:class:`kodiak.services.auth_service.AuthService`.

Typical usage example:

    $ kodiak logout
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from kodiak.cli.services.auth_service import AuthService, CredentialStorageError

app = typer.Typer(
    name="logout",
    help="Log out of Kodiak and remove stored GitHub credentials.",
)

console = Console()


def _render_success_panel() -> None:
    """Render a success panel after a completed logout."""
    console.print(
        Panel.fit(
            "[bold green]Logged out successfully.[/bold green]\n"
            "Your stored credentials have been removed.",
            title="[bold green]Kodiak[/bold green]",
            border_style="green",
        )
    )


def _render_info_panel(message: str) -> None:
    """Render an informational panel.

    Args:
        message: Human-readable informational message.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]{message}[/bold cyan]",
            title="[bold cyan]Kodiak[/bold cyan]",
            border_style="cyan",
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


@app.command()
def logout(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Log out of Kodiak by removing stored GitHub credentials.

    This command delegates credential removal to :class:`AuthService`.
    If no credentials are currently stored, the user is informed and no
    action is taken. On success, a confirmation panel is shown. On
    failure, a descriptive error panel is displayed and the process exits
    with a non-zero status code.

    Args:
        yes: If ``True``, skips the interactive confirmation prompt.

    Raises:
        typer.Exit: With code ``0`` on success or no-op, ``1`` on
            credential removal failure, or ``2`` on any unexpected error.
    """
    auth_service = AuthService()

    if not auth_service.has_stored_credentials():
        _render_info_panel("You are not currently logged in.")
        raise typer.Exit(code=0)

    if not yes and not typer.confirm(
        "Are you sure you want to log out?", default=True
    ):
        console.print("[yellow]Logout cancelled.[/yellow]")
        raise typer.Exit(code=0)

    try:
        with console.status(
            "[bold cyan]Removing stored credentials...[/bold cyan]",
            spinner="dots",
        ):
            auth_service.clear_credentials()

    except CredentialStorageError as exc:
        _render_error_panel(
            "Logout Failed",
            f"Kodiak could not remove your stored credentials.\n{exc}",
        )
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel(
            "Unexpected Error",
            f"An unexpected error occurred during logout.\n{exc}",
        )
        raise typer.Exit(code=2) from exc

    _render_success_panel()
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()