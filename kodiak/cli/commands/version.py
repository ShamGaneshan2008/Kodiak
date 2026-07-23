from __future__ import annotations

import platform
from importlib import metadata

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="version",
    help="Display Kodiak version and environment information.",
    add_completion=False,
    no_args_is_help=False,
)

console = Console()

PACKAGE_NAME = "kodiak"
FALLBACK_VERSION = "0.0.0-dev"


def _get_kodiak_version() -> str:
    """Resolve the installed Kodiak version.

    Returns the version reported by the installed package metadata, or a
    development fallback if the package is not installed (e.g. running
    from a local checkout without an installed distribution).
    """
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return FALLBACK_VERSION


def _get_installation_type() -> str:
    """Determine whether Kodiak is running from an installed distribution or source."""
    try:
        metadata.version(PACKAGE_NAME)
        return "Installed Package"
    except metadata.PackageNotFoundError:
        return "Source Checkout"


def _get_build_mode() -> str:
    """Determine the build mode from package metadata, defaulting to Development."""
    try:
        dist = metadata.distribution(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "Development"

    classifiers = dist.metadata.get_all("Classifier") or []
    for classifier in classifiers:
        if "Development Status" in classifier:
            return classifier.split("::")[-1].strip()

    return "Development"


def _build_version_table() -> Table:
    """Construct the Rich table summarizing version and environment details."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Kodiak Version", _get_kodiak_version())
    table.add_row("Python Version", platform.python_version())
    table.add_row("Platform", platform.system())
    table.add_row("Architecture", platform.machine())
    table.add_row("Installation Type", _get_installation_type())
    table.add_row("Build Mode", _get_build_mode())

    return table


def _print_version() -> None:
    """Render the version panel to the console."""
    table = _build_version_table()
    panel = Panel(
        table,
        title="[bold green]Kodiak[/bold green]",
        subtitle="[dim]kodiak version[/dim]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


@app.callback(invoke_without_command=True)
def version(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show Kodiak version information and exit.",
    ),
) -> None:
    """Show Kodiak version information.

    Displays the installed Kodiak version alongside relevant environment
    details such as Python version, platform, architecture, installation
    type, and build mode.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        _print_version()
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] Failed to resolve version information: {exc}")
        raise typer.Exit(code=1) from exc

    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
