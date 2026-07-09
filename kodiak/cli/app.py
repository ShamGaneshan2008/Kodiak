"""Root Typer application for the Kodiak CLI.

This module belongs to the CLI presentation layer. Its sole responsibility
is constructing and configuring the root Typer application instance that
other CLI modules extend via ``app.add_typer(...)``. It contains no
business logic, no command registrations, and no I/O.
"""

from __future__ import annotations

from typing import Final, Optional

import typer

APP_NAME: Final[str] = "kodiak"
APP_HELP: Final[str] = (
    "Kodiak: an autonomous AI software engineering platform."
)

_RICH_MARKUP_MODE: Final[str] = "rich"
_PRETTY_EXCEPTIONS_ENABLE: Final[bool] = True
_PRETTY_EXCEPTIONS_SHOW_LOCALS: Final[bool] = False
_ADD_COMPLETION: Final[bool] = True
_NO_ARGS_IS_HELP: Final[bool] = True
_HELP_OPTION_NAMES: Final[list[str]] = ["-h", "--help"]


def _root_callback(ctx: typer.Context) -> None:
    """Root callback and extension point for future global options.

    This callback intentionally performs no business logic. It exists so
    that global options (e.g. ``--verbose``, ``--debug``, ``--config``,
    ``--no-color``) can be added in the future without altering the
    structure of the root application.

    Args:
        ctx: The Typer context for the current invocation.
    """
    if ctx.obj is None:
        ctx.obj = {}


def create_app() -> typer.Typer:
    """Construct and configure the root Typer application.

    Returns:
        A fully configured, empty root Typer application ready to have
        command groups attached via ``app.add_typer(...)``.
    """
    application = typer.Typer(
        name=APP_NAME,
        help=APP_HELP,
        rich_markup_mode=_RICH_MARKUP_MODE,
        pretty_exceptions_enable=_PRETTY_EXCEPTIONS_ENABLE,
        pretty_exceptions_show_locals=_PRETTY_EXCEPTIONS_SHOW_LOCALS,
        add_completion=_ADD_COMPLETION,
        no_args_is_help=_NO_ARGS_IS_HELP,
        context_settings={"help_option_names": _HELP_OPTION_NAMES},
    )
    application.callback()(_root_callback)
    return application


app: Final[typer.Typer] = create_app()


if __name__ == "__main__":
    app()