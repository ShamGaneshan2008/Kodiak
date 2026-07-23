"""Root Typer application for the Kodiak CLI.

This module belongs to the CLI presentation layer. Its sole responsibility
is constructing and configuring the root Typer application instance that
other CLI modules extend via ``app.add_typer(...)``. It contains no
business logic, no command registrations, and no I/O.
"""

from __future__ import annotations

from typing import Final

import typer

from kodiak.cli.commands.analyze import app as analyze_app

# from kodiak.cli.commands.config import app as config_app
# from kodiak.cli.commands.doctor import app as doctor_app
# from kodiak.cli.commands.init import app as init_app
# from kodiak.cli.commands.login import app as login_app
from kodiak.cli.commands.logout import app as logout_app
from kodiak.cli.commands.memory import app as memory_app
from kodiak.cli.commands.plan import app as plan_app

# from kodiak.cli.commands.review import app as review_app
# from kodiak.cli.commands.status import app as status_app
from kodiak.cli.commands.task import app as task_app
from kodiak.cli.commands.version import app as version_app

APP_NAME: Final[str] = "kodiak"
APP_HELP: Final[str] = "Kodiak: an autonomous AI software engineering platform."

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
app.add_typer(analyze_app, name="analyze")
# app.add_typer(config_app, name="config")
# app.add_typer(doctor_app, name="doctor")

# app.add_typer(init_app, name="init")
# app.add_typer(login_app, name="login")
app.add_typer(logout_app, name="logout")
app.add_typer(memory_app, name="memory")
app.add_typer(plan_app, name="plan")
# app.add_typer(review_app, name="review")
# app.add_typer(status_app, name="status")
app.add_typer(task_app, name="task")
app.add_typer(version_app, name="version")

if __name__ == "__main__":
    app()
