"""CLI command group for viewing Kodiak configuration.

This module contains presentation logic only. Configuration is read
directly from :func:`kodiak.config.settings.get_settings`, a cached,
environment-driven ``pydantic_settings.BaseSettings`` instance. The
current backend exposes no configuration service, no key-level setter,
no reset mechanism, and no persisted/mutable config store -- settings
are resolved once from the environment/``.env`` file and cached for the
lifetime of the process. Accordingly, this CLI only supports viewing
configuration, not modifying it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kodiak.config.settings import get_settings

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(help="View Kodiak configuration.")

_SENSITIVE_KEYS = {
    "SECRET_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
    "REDIS_URL",
}


@app.command("show")
def show() -> None:
    """Display the full active configuration as a table.

    Sensitive fields (API keys, secrets, and connection URLs that embed
    credentials) are masked.

    Example:
        kodiak config show
    """
    try:
        snapshot = get_settings().model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to load configuration: {exc}")
        raise typer.Exit(code=2) from exc

    _render_table(snapshot)
    raise typer.Exit(code=0)


@app.command("get")
def get(
    key: str = typer.Argument(
        ...,
        help="Configuration field name, e.g. APP_NAME or JWT_ALGORITHM",
    ),
) -> None:
    """Display the value of a single configuration field.

    Args:
        key: Configuration field name to look up (case-insensitive).

    Example:
        kodiak config get APP_NAME
    """
    settings = get_settings()
    field_name = key.upper()

    if field_name not in type(settings).model_fields:
        _render_error(f"Unknown configuration key: '{key}'")
        raise typer.Exit(code=1)

    value = getattr(settings, field_name)
    if field_name in _SENSITIVE_KEYS:
        value = _mask(value)

    console.print(f"[bold]{field_name}[/bold] = {value}")
    raise typer.Exit(code=0)


@app.command("path")
def path() -> None:
    """Display the configured environment file used to load settings.

    This reflects ``Settings.model_config['env_file']`` and is resolved
    relative to the current working directory. It does not guarantee the
    file exists -- ``pydantic-settings`` falls back to process
    environment variables if it does not.

    Example:
        kodiak config path
    """
    env_file = get_settings().model_config.get("env_file")
    if not env_file:
        console.print("[dim]No env file configured; using process environment only.[/dim]")
        raise typer.Exit(code=0)

    console.print(str(Path(str(env_file)).resolve()))
    raise typer.Exit(code=0)


def _render_table(snapshot: dict[str, Any]) -> None:
    """Render a configuration snapshot as a Rich table, masking secrets.

    Args:
        snapshot: Flat configuration data as returned by ``Settings.model_dump``.
    """
    table = Table(title="Kodiak Configuration")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    for key, value in snapshot.items():
        display_value = _mask(value) if key in _SENSITIVE_KEYS else value
        table.add_row(key, str(display_value))

    console.print(table)


def _mask(value: Any) -> str:
    """Mask a sensitive value for display, showing only a short suffix.

    Args:
        value: The raw sensitive value.

    Returns:
        A masked string, or ``"(unset)"`` if the value is empty.
    """
    text = str(value) if value is not None else ""
    if not text:
        return "(unset)"
    tail = text[-4:] if len(text) > 4 else text
    return f"{'*' * 8}{tail}"


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
