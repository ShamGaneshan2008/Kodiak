"""Reusable CLI output formatting helpers for the Kodiak command-line interface.

This module centralizes creation of the Rich `Console` instance and
exposes a small, consistent set of helper functions for printing status
messages. It contains no business logic — callers decide *what* to
print, this module only decides *how* it looks.
"""

from __future__ import annotations

from typing import Final

from rich.console import Console

__all__ = [
    "get_console",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
]

_CONSOLE: Final[Console] = Console()


def get_console() -> Console:
    """Return the shared Rich console instance used by the CLI.

    Centralizing console creation ensures consistent output behavior
    (e.g. width, color system, and stream) across all CLI commands.

    Returns:
        The shared `Console` instance.
    """
    return _CONSOLE


def _print(symbol: str, style: str, message: str) -> None:
    """Print a symbol-prefixed message in a given style.

    Args:
        symbol: A short glyph indicating the message category.
        style: Rich style string applied to the symbol.
        message: The message to display.
    """
    _CONSOLE.print(f"[{style}]{symbol}[/{style}] {message}")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: The message to display.
    """
    _print("✔", "bold green", message)


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: The message to display.
    """
    _print("✖", "bold red", message)


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: The message to display.
    """
    _print("⚠", "bold yellow", message)


def print_info(message: str) -> None:
    """Print an informational message.

    Args:
        message: The message to display.
    """
    _print("ℹ", "bold cyan", message)
