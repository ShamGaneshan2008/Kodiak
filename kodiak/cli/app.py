"""Root Typer application for the Kodiak CLI.

This module belongs to the CLI presentation layer. Its sole responsibility
is constructing and configuring the root Typer application instance that
other CLI modules extend via ``app.add_typer(...)``. It contains no
business logic, no command registrations, and no I/O.
"""

from __future__ import annotations

from typing import Final, Literal

import typer
from click import Context
from typer.core import TyperGroup

from kodiak.cli.commands.agents import app as agents_app
from kodiak.cli.commands.analyze import app as analyze_app
from kodiak.cli.commands.approval import app as approval_app
from kodiak.cli.commands.doctor import doctor
from kodiak.cli.commands.git import app as git_app

# from kodiak.cli.commands.config import app as config_app
# from kodiak.cli.commands.doctor import app as doctor_app
# from kodiak.cli.commands.init import app as init_app
# from kodiak.cli.commands.login import app as login_app
from kodiak.cli.commands.logout import app as logout_app
from kodiak.cli.commands.memory import app as memory_app
from kodiak.cli.commands.plan import app as plan_app

# from kodiak.cli.commands.review import app as review_app
# from kodiak.cli.commands.status import app as status_app
from kodiak.cli.commands.task_v1 import app as task_app
from kodiak.cli.commands.version import app as version_app
from kodiak.cli.ui.banner import print_banner, should_show_banner
from kodiak.cli.ui.console import get_console, terminal_width

APP_NAME: Final[str] = "kodiak"
APP_HELP: Final[str] = "Kodiak: an experimental, approval-gated software engineering toolkit."

_RICH_MARKUP_MODE: Final[Literal["rich"]] = "rich"
_PRETTY_EXCEPTIONS_ENABLE: Final[bool] = True
_PRETTY_EXCEPTIONS_SHOW_LOCALS: Final[bool] = False
_ADD_COMPLETION: Final[bool] = True
_NO_ARGS_IS_HELP: Final[bool] = True
_HELP_OPTION_NAMES: Final[list[str]] = ["-h", "--help"]
_BANNER_WIDTH: Final[int] = 60
_INVOCATION_ARGS_KEY: Final[str] = "kodiak_invocation_args"


class _KodiakGroup(TyperGroup):
    """Root group that retains arguments needed for output-mode detection."""

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        ctx.meta[_INVOCATION_ARGS_KEY] = tuple(args)
        return super().parse_args(ctx, args)


def _root_callback(
    ctx: typer.Context,
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Do not show the Kodiak startup banner.",
    ),
) -> None:
    """Configure invocation-wide presentation options and startup branding.

    Args:
        ctx: The Typer context for the current invocation.
        no_banner: Whether startup branding was disabled explicitly.
    """
    if ctx.obj is None:
        ctx.obj = {}
    args = ctx.meta.get(_INVOCATION_ARGS_KEY, ())
    json_mode = "--json" in args
    help_mode = any(option in args for option in _HELP_OPTION_NAMES)
    if not help_mode and should_show_banner(json_mode=json_mode, no_banner=no_banner):
        print_banner(get_console(), compact=terminal_width() < _BANNER_WIDTH)


def create_app() -> typer.Typer:
    """Construct and configure the root Typer application.

    Returns:
        A fully configured, empty root Typer application ready to have
        command groups attached via ``app.add_typer(...)``.
    """
    application = typer.Typer(
        cls=_KodiakGroup,
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
app.add_typer(agents_app, name="agents")
app.add_typer(approval_app, name="approval")
# app.add_typer(config_app, name="config")
app.command("doctor")(doctor)

# app.add_typer(init_app, name="init")
# app.add_typer(login_app, name="login")
app.add_typer(logout_app, name="logout")
app.add_typer(git_app, name="git")
app.add_typer(memory_app, name="memory")
app.add_typer(plan_app, name="plan")
# app.add_typer(review_app, name="review")
# app.add_typer(status_app, name="status")
app.add_typer(task_app, name="task")
app.add_typer(version_app, name="version")

if __name__ == "__main__":
    app()
