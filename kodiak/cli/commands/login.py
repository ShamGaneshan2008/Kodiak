"""CLI command for authenticating a Kodiak user with GitHub.

This module defines the ``kodiak login`` command. It is strictly a
presentation-layer component: it collects no credentials itself, performs
no HTTP or OAuth calls, and contains no persistence logic. All of that
work is delegated to :class:`kodiak.services.auth_service.AuthService`.

Typical usage example:

    $ kodiak login
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from kodiak.services.auth_service import (
    AuthenticationError,
    AuthService,
    CredentialStorageError,
)

app = typer.Typer(
    name="login",
    help="Authenticate Kodiak with your GitHub account.",
)

console = Console()


def _render_success_panel(username: str) -> None:
    """Render a success panel after a completed login.

    Args:
        username: The GitHub username that was authenticated.
    """
    console.print(
        Panel.fit(
            f"[bold green]Login successful![/bold green]\n"
            f"Authenticated as [bold cyan]{username}[/bold cyan].\n"
            f"Credentials have been saved securely.",
            title="[bold green]Kodiak[/bold green]",
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


@app.command()
def login() -> None:
    """Authenticate the current user with GitHub.

    This command guides the user through the GitHub authentication flow,
    delegating all business logic to :class:`AuthService`. On success, the
    obtained credentials are persisted and a confirmation panel is shown.
    On failure, a descriptive error panel is displayed and the process
    exits with a non-zero status code.

    Raises:
        typer.Exit: With code ``0`` on success, ``1`` on authentication
            failure, ``2`` on credential storage failure, or ``3`` on any
            unexpected error.
    """
    console.print(
        Panel.fit(
            "Kodiak needs to authenticate with your [bold]GitHub[/bold] "
            "account to continue.",
            title="[bold cyan]GitHub Authentication[/bold cyan]",
            border_style="cyan",
        )
    )

    if not typer.confirm("Proceed with GitHub authentication?", default=True):
        console.print("[yellow]Login cancelled.[/yellow]")
        raise typer.Exit(code=0)

    auth_service = AuthService()

    try:
        with console.status(
            "[bold cyan]Waiting for GitHub authentication...[/bold cyan]",
            spinner="dots",
        ):
            credentials = auth_service.authenticate_with_github()
            auth_service.save_credentials(credentials)

    except AuthenticationError as exc:
        _render_error_panel(
            "Authentication Failed",
            f"Kodiak could not authenticate with GitHub.\n{exc}",
        )
        raise typer.Exit(code=1) from exc

    except CredentialStorageError as exc:
        _render_error_panel(
            "Credential Storage Failed",
            f"Authentication succeeded, but credentials could not be saved.\n{exc}",
        )
        raise typer.Exit(code=2) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel(
            "Unexpected Error",
            f"An unexpected error occurred during login.\n{exc}",
        )
        raise typer.Exit(code=3) from exc

    _render_success_panel(credentials.username)
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()