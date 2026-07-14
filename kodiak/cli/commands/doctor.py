"""CLI command for verifying the health of a Kodiak installation.

This module contains presentation logic only.

Health checks are NOT currently available from the CLI. The previous
implementation depended on ``kodiak.cli.services.doctor_service.DoctorService``,
which does not exist in this repository. There is no other existing
backend module that performs the Python/Git/Docker/API-key/connectivity/
GitHub-auth/configuration/workspace/sandbox checks this command
previously described. Rather than invent a replacement service, this
command reports the limitation clearly.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()
error_console = Console(stderr=True)

app = typer.Typer()

_UNAVAILABLE_MESSAGE = (
    "Health checks are not available from the CLI. The backend service "
    "this command depended on (DoctorService) does not exist in this "
    "repository, and no other existing backend module performs these "
    "checks."
)


@app.command("doctor")
def doctor() -> None:
    """Report that health checks are not available from the CLI.

    Raises:
        typer.Exit: Always, with code ``2``.
    """
    error_console.print(
        Panel(
            _UNAVAILABLE_MESSAGE,
            title="Feature Not Available",
            border_style="yellow",
            title_align="left",
        )
    )
    raise typer.Exit(code=2)