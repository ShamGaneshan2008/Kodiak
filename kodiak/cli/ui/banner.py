"""Banner rendering utilities for the Kodiak CLI.

This module is a pure presentation layer. It renders Rich-based banners
(welcome, success, error, warning, version, task, and analysis banners)
used across the Kodiak command-line interface.

Notes:
    This module contains no business logic, no I/O beyond stdout rendering,
    and no dependencies on Typer, databases, GitHub, or AI backends. It only
    knows how to turn strings into styled Rich renderables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED, Box
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

__all__ = [
    "KODIAK_LOGO",
    "BannerStyle",
    "render_welcome_banner",
    "render_success_banner",
    "render_error_banner",
    "render_warning_banner",
    "render_version_banner",
    "render_task_banner",
    "render_analysis_banner",
]

console: Final[Console] = Console()

KODIAK_LOGO: Final[str] = r"""
   ██╗  ██╗ ██████╗ ██████╗ ██╗ █████╗ ██╗  ██╗
   ██║ ██╔╝██╔═══██╗██╔══██╗██║██╔══██╗██║ ██╔╝
   █████╔╝ ██║   ██║██║  ██║██║███████║█████╔╝
   ██╔═██╗ ██║   ██║██║  ██║██║██╔══██║██╔═██╗
   ██║  ██╗╚██████╔╝██████╔╝██║██║  ██║██║  ██╗
   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
""".strip("\n")


@dataclass(frozen=True, slots=True)
class BannerStyle:
    """Visual style configuration for a banner.

    Attributes:
        border_style: Rich color/style string applied to the panel border.
        title_style: Rich style string applied to the panel title text.
        box: Rich box-drawing character set used for the panel border.
        icon: Optional glyph prefixed to the banner's heading line.
    """

    border_style: str
    title_style: str
    box: Box = ROUNDED
    icon: str = ""


_STYLE_WELCOME: Final[BannerStyle] = BannerStyle(
    border_style="bright_yellow",
    title_style="bold bright_yellow",
    box=DOUBLE,
    icon="🐻",
)
_STYLE_SUCCESS: Final[BannerStyle] = BannerStyle(
    border_style="bright_green",
    title_style="bold bright_green",
    box=ROUNDED,
    icon="✔",
)
_STYLE_ERROR: Final[BannerStyle] = BannerStyle(
    border_style="bright_red",
    title_style="bold bright_red",
    box=HEAVY,
    icon="✖",
)
_STYLE_WARNING: Final[BannerStyle] = BannerStyle(
    border_style="bright_yellow",
    title_style="bold bright_yellow",
    box=ROUNDED,
    icon="⚠",
)
_STYLE_VERSION: Final[BannerStyle] = BannerStyle(
    border_style="bright_cyan",
    title_style="bold bright_cyan",
    box=ROUNDED,
    icon="ℹ",
)
_STYLE_TASK: Final[BannerStyle] = BannerStyle(
    border_style="bright_magenta",
    title_style="bold bright_magenta",
    box=ROUNDED,
    icon="▶",
)
_STYLE_ANALYSIS: Final[BannerStyle] = BannerStyle(
    border_style="bright_blue",
    title_style="bold bright_blue",
    box=ROUNDED,
    icon="◆",
)


def _build_heading(text: str, style: BannerStyle) -> Text:
    """Build a styled heading line for a banner.

    Args:
        text: The heading text to display.
        style: The banner style controlling icon and title color.

    Returns:
        A Rich Text object with the icon (if any) and heading applied.
    """
    heading = f"{style.icon}  {text}" if style.icon else text
    return Text(heading, style=style.title_style, justify="center")


def _build_panel(
    body: RenderableType,
    *,
    style: BannerStyle,
    title: str | None = None,
    subtitle: str | None = None,
    padding: tuple[int, int] = (1, 4),
) -> Panel:
    """Wrap a renderable body in a consistently styled Rich panel.

    Args:
        body: The renderable content to place inside the panel.
        style: The banner style controlling border color and box type.
        title: Optional panel title, shown embedded in the top border.
        subtitle: Optional panel subtitle, shown embedded in the bottom border.
        padding: Padding applied around the body as (vertical, horizontal).

    Returns:
        A configured Rich Panel ready to be printed or composed further.
    """
    return Panel(
        Padding(body, padding),
        title=title,
        subtitle=subtitle,
        border_style=style.border_style,
        box=style.box,
        expand=True,
    )


def _render(panel: Panel) -> None:
    """Print a panel to the shared console, centered horizontally.

    Args:
        panel: The Rich Panel to render.
    """
    console.print(Align.center(panel))


def render_welcome_banner(
    version: str,
    tagline: str = "Autonomous AI Software Engineering",
) -> None:
    """Render the Kodiak welcome banner shown at CLI startup.

    Args:
        version: The Kodiak version string, e.g. "1.4.0".
        tagline: A short subtitle describing Kodiak, shown beneath the logo.
    """
    logo = Text(KODIAK_LOGO, style="bold bright_yellow", justify="center")
    subtitle = Text(tagline, style="italic bright_white", justify="center")
    version_line = Text(f"v{version}", style="dim bright_yellow", justify="center")

    body = Group(logo, Text(""), subtitle, version_line)
    panel = _build_panel(body, style=_STYLE_WELCOME)
    _render(panel)


def render_success_banner(message: str, *, title: str = "Success") -> None:
    """Render a success banner.

    Args:
        message: The success message to display.
        title: The panel title. Defaults to "Success".
    """
    heading = _build_heading(message, _STYLE_SUCCESS)
    panel = _build_panel(heading, style=_STYLE_SUCCESS, title=title)
    _render(panel)


def render_error_banner(
    message: str,
    *,
    title: str = "Error",
    detail: str | None = None,
) -> None:
    """Render an error banner.

    Args:
        message: The primary error message to display.
        title: The panel title. Defaults to "Error".
        detail: Optional secondary line with additional error context.
    """
    heading = _build_heading(message, _STYLE_ERROR)
    renderables: list[RenderableType] = [heading]
    if detail:
        renderables.append(Text(""))
        renderables.append(Text(detail, style="dim bright_red", justify="center"))

    panel = _build_panel(Group(*renderables), style=_STYLE_ERROR, title=title)
    _render(panel)


def render_warning_banner(message: str, *, title: str = "Warning") -> None:
    """Render a warning banner.

    Args:
        message: The warning message to display.
        title: The panel title. Defaults to "Warning".
    """
    heading = _build_heading(message, _STYLE_WARNING)
    panel = _build_panel(heading, style=_STYLE_WARNING, title=title)
    _render(panel)


def render_version_banner(
    version: str,
    *,
    python_version: str | None = None,
    platform_name: str | None = None,
) -> None:
    """Render a version information banner.

    Args:
        version: The Kodiak version string, e.g. "1.4.0".
        python_version: Optional Python runtime version to display.
        platform_name: Optional platform/OS identifier to display.
    """
    lines: list[RenderableType] = [
        Text(f"Kodiak {version}", style=_STYLE_VERSION.title_style, justify="center")
    ]
    if python_version:
        lines.append(Text(f"Python {python_version}", style="dim bright_cyan", justify="center"))
    if platform_name:
        lines.append(Text(platform_name, style="dim bright_cyan", justify="center"))

    panel = _build_panel(Group(*lines), style=_STYLE_VERSION, title="Version")
    _render(panel)


def render_task_banner(
    task_name: str,
    *,
    description: str | None = None,
    step: tuple[int, int] | None = None,
) -> None:
    """Render a banner announcing the start of a task.

    Args:
        task_name: The name of the task being executed.
        description: Optional short description of what the task does.
        step: Optional (current, total) step counter, e.g. (2, 5).
    """
    title = "Task"
    if step is not None:
        current, total = step
        title = f"Task [{current}/{total}]"

    heading = _build_heading(task_name, _STYLE_TASK)
    renderables: list[RenderableType] = [heading]
    if description:
        renderables.append(Text(""))
        renderables.append(Text(description, style="bright_white", justify="center"))

    panel = _build_panel(Group(*renderables), style=_STYLE_TASK, title=title)
    _render(panel)


def render_analysis_banner(
    subject: str,
    *,
    metrics: dict[str, str] | None = None,
) -> None:
    """Render a banner summarizing an analysis result.

    Args:
        subject: The name of the analyzed subject, e.g. a repository or file.
        metrics: Optional mapping of metric labels to their string values,
            rendered as aligned key/value lines beneath the heading.
    """
    heading = _build_heading(subject, _STYLE_ANALYSIS)
    renderables: list[RenderableType] = [heading]

    if metrics:
        renderables.append(Text(""))
        label_width = max(len(label) for label in metrics)
        for label, value in metrics.items():
            line = Text(justify="center")
            line.append(f"{label.rjust(label_width)}  ", style="dim bright_blue")
            line.append(value, style="bold bright_white")
            renderables.append(line)

    panel = _build_panel(Group(*renderables), style=_STYLE_ANALYSIS, title="Analysis")
    _render(panel)
