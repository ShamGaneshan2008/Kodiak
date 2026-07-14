"""CLI command for authenticating a Kodiak user with GitHub.

This module defines the ``kodiak login`` command. The current backend
(``kodiak.api.routers.auth``) exposes GitHub authentication only as a
web OAuth redirect flow (``GET /auth/github/callback``), designed to be
driven by a browser redirecting back to a running API server with a
``code``/``state`` pair. There is no ``AuthService``, no local HTTP
client, and no local credential storage anywhere in this repository that
the CLI could use to drive that flow or persist the resulting tokens.
As a result, this command performs no real authentication; it informs
the user that interactive login is not currently supported by the CLI
and exits cleanly.

Typical usage example:

    $ kodiak login
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="login",
    help="Authenticate Kodiak with your GitHub account.",
)

console = Console()


@app.command()
def login() -> None:
    """Inform the user that CLI login is not currently supported.

    GitHub authentication in the current backend is a web OAuth redirect
    flow served by the API (``/auth/github/callback``) and requires a
    browser and a running API server to complete. No CLI-side HTTP
    client or local credential storage exists to drive this flow or
    persist tokens, so there is no action this command can safely
    perform.

    Raises:
        typer.Exit: Always exits with code ``0``.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Interactive login is not currently supported by the CLI.[/bold cyan]\n"
            "Kodiak authenticates via a web-based GitHub OAuth flow "
            "handled by the API server (/auth/github/callback).\n"
            "Please authenticate through the Kodiak web application.",
            title="[bold cyan]Kodiak[/bold cyan]",
            border_style="cyan",
        )
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()