"""Public API for the Kodiak CLI package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import CLIConfig, ConfigError, ConfigManager

if TYPE_CHECKING:
    from typer import Typer


def create_app() -> Typer:
    """Create the CLI lazily so service imports do not initialize every command."""
    from .app import create_app as build_app

    return build_app()


__all__ = [
    "create_app",
    "CLIConfig",
    "ConfigError",
    "ConfigManager",
]
