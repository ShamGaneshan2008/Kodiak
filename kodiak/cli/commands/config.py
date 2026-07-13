"""CLI command group for viewing and modifying Kodiak configuration.

This module contains presentation logic only. All configuration reads,
writes, and validation are delegated to
:class:`kodiak.services.config_service.ConfigService`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kodiak.cli.services.config_service import ConfigService
from kodiak.cli.services.exceptions import ConfigKeyError, ConfigValidationError

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(help="View and modify Kodiak configuration.")


@app.command("show")
def show() -> None:
    """Display the full active configuration as a table.

    Nested configuration sections are flattened into dot-notation keys
    (for example ``llm.model``) for readability.

    Example:
        kodiak config show
    """
    service = ConfigService()

    try:
        snapshot = asyncio.run(service.get_all())
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to load configuration: {exc}")
        raise typer.Exit(code=2)

    _render_table(snapshot)
    raise typer.Exit(code=0)


@app.command("get")
def get(
    key: str = typer.Argument(
        ...,
        help="Dot-notation configuration key, e.g. llm.model",
    ),
) -> None:
    """Display the value of a single configuration key.

    Args:
        key: Dot-notation configuration key to look up.

    Example:
        kodiak config get llm.model
    """
    service = ConfigService()

    try:
        value = asyncio.run(service.get(key))
    except ConfigKeyError as exc:
        _render_error(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to read configuration: {exc}")
        raise typer.Exit(code=2)

    console.print(f"[bold]{key}[/bold] = {value}")
    raise typer.Exit(code=0)


@app.command("set")
def set_command(
    key: str = typer.Argument(
        ...,
        help="Dot-notation configuration key, e.g. llm.model",
    ),
    value: str = typer.Argument(
        ...,
        help="New value to assign to the key.",
    ),
) -> None:
    """Set a configuration key to a new value.

    The value is validated by ConfigService before being persisted; no
    validation or type coercion happens in this command.

    Args:
        key: Dot-notation configuration key to update.
        value: New value, as a raw string, to assign.

    Example:
        kodiak config set llm.model claude-sonnet-5
    """
    service = ConfigService()

    try:
        updated = asyncio.run(service.set(key, value))
    except ConfigKeyError as exc:
        _render_error(str(exc))
        raise typer.Exit(code=1)
    except ConfigValidationError as exc:
        _render_error(f"Invalid value for '{key}': {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to update configuration: {exc}")
        raise typer.Exit(code=2)

    console.print(
        Panel(
            f"[bold]{key}[/bold] set to [green]{updated}[/green]",
            title="Configuration Updated",
            border_style="green",
        )
    )
    raise typer.Exit(code=0)


@app.command("reset")
def reset(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Reset configuration to its default values.

    This is a destructive operation. Unless ``--yes`` is passed, the user
    is prompted for confirmation before any changes are made.

    Args:
        yes: If True, skip the interactive confirmation prompt.

    Example:
        kodiak config reset
        kodiak config reset --yes
    """
    if not yes:
        confirmed = typer.confirm(
            "This will overwrite your current configuration with defaults. Continue?"
        )
        if not confirmed:
            console.print("[dim]Reset cancelled.[/dim]")
            raise typer.Exit(code=0)

    service = ConfigService()

    try:
        asyncio.run(service.reset())
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to reset configuration: {exc}")
        raise typer.Exit(code=2)

    console.print(
        Panel(
            "Configuration has been reset to defaults.",
            title="Configuration Reset",
            border_style="green",
        )
    )
    raise typer.Exit(code=0)


@app.command("path")
def path() -> None:
    """Display the filesystem path to the active configuration file.

    Example:
        kodiak config path
    """
    service = ConfigService()

    try:
        config_path: Path = asyncio.run(service.get_config_path())
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to resolve configuration path: {exc}")
        raise typer.Exit(code=2)

    console.print(str(config_path))
    raise typer.Exit(code=0)


def _render_table(snapshot: dict[str, Any]) -> None:
    """Render a flattened configuration snapshot as a Rich table.

    Args:
        snapshot: Nested configuration data as returned by ConfigService.
    """
    table = Table(title="Kodiak Configuration")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    for key, value in _flatten(snapshot).items():
        table.add_row(key, str(value))

    console.print(table)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary into dot-notation keys.

    Args:
        data: Nested dictionary to flatten.
        prefix: Key prefix accumulated from parent levels.

    Returns:
        A single-level dictionary mapping dot-notation keys to values.
    """
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def _render_error(message: str) -> None:
    """Render an error message inside a red Rich panel.

    Args:
        message: Human-readable error description.
    """
    error_console.print(
        Panel(
            message,
            title="Configuration Error",
            border_style="red",
            title_align="left",
        )
    )