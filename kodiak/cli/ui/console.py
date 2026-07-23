"""Shared Rich Console management for the Kodiak CLI.

This module is the single source of truth for console rendering: it
owns the shared stdout `Console`, a companion stderr `Console`, quiet
mode support, and small helper functions for terminal capability
detection (width, interactivity, color system, unicode support).

Every UI component in `kodiak.cli.ui` should obtain its console through
this module (via `get_console()` / `get_error_console()`) rather than
constructing its own `Console` instance.

This module contains no business logic.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Final

from rich.console import Console

__all__ = [
    "ConsoleConfig",
    "DEFAULT_WIDTH",
    "FALLBACK_WIDTH",
    "get_console",
    "get_error_console",
    "is_terminal_interactive",
    "terminal_width",
    "supports_color",
    "supports_unicode",
    "set_quiet_mode",
    "is_quiet_mode",
    "reset_consoles",
]

DEFAULT_WIDTH: Final[int] = 100
"""Default console width used when no terminal size can be detected."""

FALLBACK_WIDTH: Final[int] = 80
"""Minimum safe width used as an absolute fallback."""

_NO_COLOR_ENV_VARS: Final[tuple[str, ...]] = ("NO_COLOR", "KODIAK_NO_COLOR")
_FORCE_COLOR_ENV_VARS: Final[tuple[str, ...]] = ("FORCE_COLOR", "KODIAK_FORCE_COLOR")


@dataclass(slots=True)
class ConsoleConfig:
    """Runtime configuration controlling console behavior.

    Attributes:
        quiet: If True, output consoles are muted (rendering is
            suppressed at print time by callers checking `is_quiet_mode()`,
            or by the console's own `quiet` flag where supported).
        force_terminal: If not None, overrides Rich's auto-detection of
            whether output is an interactive terminal.
        no_color: If True, disables color output regardless of terminal
            capability.
        width: If not None, overrides automatic terminal width detection.
    """

    quiet: bool = False
    force_terminal: bool | None = None
    no_color: bool = False
    width: int | None = None


_config: ConsoleConfig = ConsoleConfig()
_console: Console | None = None
_error_console: Console | None = None


def _env_flag_set(names: tuple[str, ...]) -> bool:
    """Check whether any of the given environment variables is set truthy.

    Args:
        names: Environment variable names to check.

    Returns:
        True if any variable is present and not empty/"0"/"false".
    """
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip().lower() not in ("", "0", "false"):
            return True
    return False


def _resolve_no_color() -> bool:
    """Determine whether color output should be disabled.

    Returns:
        True if color should be disabled, considering explicit config
        and standard environment variables.
    """
    if _config.no_color:
        return True
    if _env_flag_set(_FORCE_COLOR_ENV_VARS):
        return False
    return _env_flag_set(_NO_COLOR_ENV_VARS)


def _build_console(*, file: object | None = None, stderr: bool = False) -> Console:
    """Construct a new Console instance honoring the current configuration.

    Args:
        file: An optional writable file-like object to render into.
            Defaults to sys.stdout (or sys.stderr when `stderr` is True).
        stderr: Whether this console targets stderr.

    Returns:
        A configured Rich Console instance.
    """
    return Console(
        file=file,  # type: ignore[arg-type]
        stderr=stderr,
        force_terminal=_config.force_terminal,
        no_color=_resolve_no_color(),
        width=_config.width,
        highlight=False,
        soft_wrap=False,
    )


def get_console() -> Console:
    """Return the shared stdout Console instance, creating it if needed.

    Returns:
        The process-wide stdout Console used for standard CLI output.
    """
    global _console
    if _console is None:
        _console = _build_console()
    return _console


def get_error_console() -> Console:
    """Return the shared stderr Console instance, creating it if needed.

    Returns:
        The process-wide stderr Console used for error and diagnostic
        output, kept separate from stdout so redirection/piping of
        normal output is unaffected.
    """
    global _error_console
    if _error_console is None:
        _error_console = _build_console(stderr=True)
    return _error_console


def reset_consoles() -> None:
    """Discard cached Console instances so they are rebuilt on next use.

    This is primarily useful after changing configuration (quiet mode,
    color, width, or force_terminal) or in test suites that need a
    fresh Console per test.
    """
    global _console, _error_console
    _console = None
    _error_console = None


def set_quiet_mode(enabled: bool) -> None:
    """Enable or disable quiet mode for shared consoles.

    Args:
        enabled: If True, subsequent output should be suppressed by
            callers checking `is_quiet_mode()`.
    """
    _config.quiet = enabled


def is_quiet_mode() -> bool:
    """Return whether quiet mode is currently enabled.

    Returns:
        True if quiet mode is active.
    """
    return _config.quiet


def is_terminal_interactive() -> bool:
    """Determine whether the current stdout is an interactive terminal.

    Returns:
        True if stdout is attached to an interactive terminal (i.e. not
        redirected to a file or pipe), False otherwise.
    """
    return get_console().is_terminal


def terminal_width(*, fallback: int = DEFAULT_WIDTH) -> int:
    """Return the current terminal width in columns.

    Args:
        fallback: The width to use if the terminal size cannot be
            determined (e.g. output is redirected).

    Returns:
        The detected terminal width, or `fallback` if detection fails.
    """
    if _config.width is not None:
        return _config.width

    try:
        size = os.get_terminal_size(sys.__stdout__.fileno())  # type: ignore[union-attr]
        return size.columns
    except (AttributeError, OSError, ValueError):
        return max(fallback, FALLBACK_WIDTH)


def supports_color() -> bool:
    """Determine whether the shared console supports color output.

    Returns:
        True if the console's detected color system is not None and
        color has not been explicitly disabled.
    """
    if _resolve_no_color():
        return False
    return get_console().color_system is not None


def supports_unicode() -> bool:
    """Determine whether the shared console supports unicode output.

    Returns:
        True if the console's output encoding can represent common
        unicode glyphs (e.g. box-drawing characters and emoji), False
        if it is restricted to a legacy encoding such as ASCII.
    """
    encoding = getattr(get_console().file, "encoding", None) or ""
    normalized = encoding.lower().replace("-", "")
    return normalized not in ("ascii", "usascii", "")
