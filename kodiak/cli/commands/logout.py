"""CLI command for logging a Kodiak user out.

This module defines the ``kodiak logout`` command. The current backend
(``kodiak.api.routers.auth``) issues stateless JWT access/refresh tokens
and exposes no logout, session-revocation, or credential-storage
mechanism of any kind -- there is no ``AuthService``, no local credential
store, and no ``/auth/logout`` endpoint to call. As a result, this
command performs no real action; it simply informs the user that logout
is not supported by the current backend.

Typical usage example:

    $ kodiak logout
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="logout",
    help="Log out of Kodiak.",
)

console = Console()


@app.command()
def logout() -> None:
    """Inform the user that logout is not supported by the current backend.

    Kodiak's authentication backend issues stateless JWT access and
    refresh tokens and provides no server-side session or token
    revocation, and no local credential storage exists in the CLI to
    clear. There is therefore no action this command can safely
    perform.

    Raises:
        typer.Exit: Always exits with code ``0``.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Logout is not supported by the current backend.[/bold cyan]\n"
            "Kodiak issues stateless access tokens and does not track "
            "sessions or local credentials to remove.",
            title="[bold cyan]Kodiak[/bold cyan]",
            border_style="cyan",
        )
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()