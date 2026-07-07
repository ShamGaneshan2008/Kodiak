"""CLI command for initializing Kodiak inside a repository.

This module contains presentation logic only. Git validation, directory
and config creation, workspace setup, and language/package-manager/
framework detection are all delegated to
:class:`kodiak.services.init_service.InitService`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from kodiak.schemas.init import InitRequest, InitResult, InitStep
from kodiak.services.exceptions import InitError
from kodiak.services.init_service import InitService

console = Console()
error_console = Console(stderr=True)

app = typer.Typer()

_EXPECTED_STEPS = 7


@app.command("init")
def init(
    target: Path = typer.Argument(
        Path("."),
        help="Directory to initialize Kodiak in.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Reinitialize an existing Kodiak workspace, overwriting "
        "existing configuration.",
    ),
) -> None:
    """Initialize Kodiak inside a repository.

    Validates that the target is a Git repository, creates the
    ``.kodiak`` directory and ``config.toml``, initializes the workspace,
    and detects the project's language, package manager, and framework.
    Progress is shown as a Rich progress bar, and a summary panel is
    displayed once initialization completes.

    Args:
        target: Directory to initialize. Created if it does not exist.
        force: If True, reinitialize even if Kodiak is already set up.

    Examples:
        kodiak init
        kodiak init .
        kodiak init my-project
        kodiak init my-project --force
    """
    resolved_path = target.resolve()

    if resolved_path.exists() and not resolved_path.is_dir():
        _render_error(f"Path exists and is not a directory: {resolved_path}")
        raise typer.Exit(code=1)

    request = InitRequest(path=resolved_path, force=force)
    service = InitService()

    try:
        result = asyncio.run(_run_init(service, request))
    except InitError as exc:
        _render_error(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Unexpected error during initialization: {exc}")
        raise typer.Exit(code=2)

    _render_summary(result)
    raise typer.Exit(code=0)


async def _run_init(service: InitService, request: InitRequest) -> InitResult:
    """Run initialization while displaying a Rich progress bar.

    Args:
        service: The service responsible for performing initialization.
        request: The validated initialization request.

    Returns:
        The final initialization result, including detected project
        metadata.

    Raises:
        InitError: If the service reports a recoverable initialization
            failure, such as a missing Git repository or an already
            initialized workspace without ``--force``.
    """
    result: Optional[InitResult] = None

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Initializing Kodiak...", total=_EXPECTED_STEPS)

        async for step in service.initialize(request):
            progress.update(task_id, advance=1, description=step.label)
            if step.result is not None:
                result = step.result

        progress.update(task_id, completed=_EXPECTED_STEPS)

    if result is None:
        raise InitError("Initialization completed without a final result.")
    return result


def _render_summary(result: InitResult) -> None:
    """Render the initialization summary as a table inside a success panel.

    Args:
        result: The final initialization result to display.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row(
        "Git repository",
        "[green]yes[/green]" if result.git_repository else "[yellow]no[/yellow]",
    )
    table.add_row("Language", result.language or "[dim]not detected[/dim]")
    table.add_row(
        "Package manager", result.package_manager or "[dim]not detected[/dim]"
    )
    table.add_row("Framework", result.framework or "[dim]not detected[/dim]")
    table.add_row("Config file", str(result.config_path))
    table.add_row("Workspace", str(result.workspace_path))

    console.print(
        Panel(
            table,
            title="Kodiak Initialized",
            border_style="green",
            title_align="left",
        )
    )


def _render_error(message: str) -> None:
    """Render an error message inside a red Rich panel.

    Args:
        message: Human-readable error description.
    """
    error_console.print(
        Panel(
            message,
            title="Initialization Failed",
            border_style="red",
            title_align="left",
        )
    )