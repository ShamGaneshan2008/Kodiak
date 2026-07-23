"""Reusable Rich panel components for the Kodiak CLI.

This module is a pure presentation layer: it builds and returns
`rich.panel.Panel` objects with consistent spacing, borders, colors,
and titles. Nothing in this module prints to the console or performs
any business logic — callers are responsible for rendering the
returned panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rich.box import HEAVY, ROUNDED, Box
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

__all__ = [
    "PanelStyle",
    "SuccessPanel",
    "ErrorPanel",
    "WarningPanel",
    "InfoPanel",
    "SummaryPanel",
    "ReviewPanel",
    "AnalysisPanel",
]

_DEFAULT_PADDING: Final[tuple[int, int]] = (1, 3)


@dataclass(frozen=True, slots=True)
class PanelStyle:
    """Visual style configuration for a panel.

    Attributes:
        border_style: Rich color/style string applied to the panel border.
        title_style: Rich style string applied to the panel title text.
        box: Rich box-drawing character set used for the panel border.
        icon: Optional glyph prefixed to the panel title.
    """

    border_style: str
    title_style: str
    box: Box = ROUNDED
    icon: str = ""

    def format_title(self, title: str) -> str:
        """Combine the style's icon with a title string.

        Args:
            title: The base title text.

        Returns:
            The title prefixed with the icon, or unchanged if no icon.
        """
        return f"{self.icon} {title}" if self.icon else title


_STYLE_SUCCESS: Final[PanelStyle] = PanelStyle(
    border_style="bright_green",
    title_style="bold bright_green",
    box=ROUNDED,
    icon="✔",
)
_STYLE_ERROR: Final[PanelStyle] = PanelStyle(
    border_style="bright_red",
    title_style="bold bright_red",
    box=HEAVY,
    icon="✖",
)
_STYLE_WARNING: Final[PanelStyle] = PanelStyle(
    border_style="bright_yellow",
    title_style="bold bright_yellow",
    box=ROUNDED,
    icon="⚠",
)
_STYLE_INFO: Final[PanelStyle] = PanelStyle(
    border_style="bright_cyan",
    title_style="bold bright_cyan",
    box=ROUNDED,
    icon="ℹ",
)
_STYLE_SUMMARY: Final[PanelStyle] = PanelStyle(
    border_style="bright_magenta",
    title_style="bold bright_magenta",
    box=ROUNDED,
    icon="≡",
)
_STYLE_REVIEW: Final[PanelStyle] = PanelStyle(
    border_style="bright_blue",
    title_style="bold bright_blue",
    box=ROUNDED,
    icon="🔍",
)
_STYLE_ANALYSIS: Final[PanelStyle] = PanelStyle(
    border_style="bright_blue",
    title_style="bold bright_blue",
    box=ROUNDED,
    icon="◆",
)


def _base_panel(
    body: RenderableType,
    *,
    style: PanelStyle,
    title: str,
    subtitle: str | None = None,
    padding: tuple[int, int] = _DEFAULT_PADDING,
) -> Panel:
    """Construct a Panel with consistent styling.

    Args:
        body: The renderable content to place inside the panel.
        style: The style controlling border color, box, and title icon.
        title: The panel title text (icon is added automatically).
        subtitle: Optional subtitle shown embedded in the bottom border.
        padding: Padding applied around the body as (vertical, horizontal).

    Returns:
        A configured, unprinted Rich Panel object.
    """
    return Panel(
        Padding(body, padding),
        title=style.format_title(title),
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=style.border_style,
        box=style.box,
        expand=True,
    )


def _key_value_table(data: dict[str, str]) -> Table:
    """Build a borderless two-column table for label/value pairs.

    Args:
        data: A mapping of labels to their string values.

    Returns:
        A Rich Table with no header or grid lines, suited for embedding
        inside a Panel.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim")
    table.add_column(justify="left", style="bold bright_white")
    for label, value in data.items():
        table.add_row(f"{label}:", value)
    return table


def SuccessPanel(
    message: str,
    *,
    title: str = "Success",
    subtitle: str | None = None,
) -> Panel:
    """Build a panel for a successful operation.

    Args:
        message: The success message to display.
        title: The panel title. Defaults to "Success".
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    body = Text(message, style="bright_white")
    return _base_panel(body, style=_STYLE_SUCCESS, title=title, subtitle=subtitle)


def ErrorPanel(
    message: str,
    *,
    title: str = "Error",
    detail: str | None = None,
    subtitle: str | None = None,
) -> Panel:
    """Build a panel for an error condition.

    Args:
        message: The primary error message to display.
        title: The panel title. Defaults to "Error".
        detail: Optional secondary line with additional error context
            (e.g. an exception message or traceback summary).
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    lines: list[RenderableType] = [Text(message, style="bold bright_white")]
    if detail:
        lines.append(Text(""))
        lines.append(Text(detail, style="dim bright_red"))

    return _base_panel(Group(*lines), style=_STYLE_ERROR, title=title, subtitle=subtitle)


def WarningPanel(
    message: str,
    *,
    title: str = "Warning",
    subtitle: str | None = None,
) -> Panel:
    """Build a panel for a warning condition.

    Args:
        message: The warning message to display.
        title: The panel title. Defaults to "Warning".
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    body = Text(message, style="bright_white")
    return _base_panel(body, style=_STYLE_WARNING, title=title, subtitle=subtitle)


def InfoPanel(
    message: str,
    *,
    title: str = "Info",
    subtitle: str | None = None,
) -> Panel:
    """Build a panel for general informational messages.

    Args:
        message: The informational message to display.
        title: The panel title. Defaults to "Info".
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    body = Text(message, style="bright_white")
    return _base_panel(body, style=_STYLE_INFO, title=title, subtitle=subtitle)


def SummaryPanel(
    summary: str | None = None,
    *,
    metrics: dict[str, str] | None = None,
    title: str = "Summary",
    subtitle: str | None = None,
) -> Panel:
    """Build a panel summarizing the outcome of a multi-step operation.

    Args:
        summary: Optional free-text summary line shown above the metrics.
        metrics: Optional mapping of metric labels to their string values,
            rendered as an aligned key/value table.
        title: The panel title. Defaults to "Summary".
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    sections: list[RenderableType] = []
    if summary:
        sections.append(Text(summary, style="bright_white"))
    if metrics:
        if sections:
            sections.append(Text(""))
        sections.append(_key_value_table(metrics))

    if not sections:
        sections.append(Text("Nothing to summarize.", style="dim"))

    return _base_panel(Group(*sections), style=_STYLE_SUMMARY, title=title, subtitle=subtitle)


def ReviewPanel(
    content: str,
    *,
    reviewer: str | None = None,
    verdict: str | None = None,
    title: str = "Review",
    subtitle: str | None = None,
) -> Panel:
    """Build a panel presenting a code or task review.

    Args:
        content: The main review body text (e.g. findings or comments).
        reviewer: Optional name or identifier of the reviewer/agent.
        verdict: Optional short verdict label (e.g. "Approved",
            "Changes Requested"), rendered as a header line.
        title: The panel title. Defaults to "Review".
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    sections: list[RenderableType] = []

    if reviewer or verdict:
        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        left = Text(f"Reviewer: {reviewer}", style="dim") if reviewer else Text("")
        right = Text(verdict, style="bold bright_blue") if verdict else Text("")
        header.add_row(left, right)
        sections.append(header)
        sections.append(Text(""))

    sections.append(Text(content, style="bright_white"))

    return _base_panel(Group(*sections), style=_STYLE_REVIEW, title=title, subtitle=subtitle)


def AnalysisPanel(
    subject: str,
    *,
    metrics: dict[str, str] | None = None,
    findings: list[str] | None = None,
    title: str = "Analysis",
    subtitle: str | None = None,
) -> Panel:
    """Build a panel presenting analysis results for a subject.

    Args:
        subject: The name of the analyzed subject (e.g. a file, module,
            or repository), shown as the panel's heading line.
        metrics: Optional mapping of metric labels to their string values,
            rendered as an aligned key/value table.
        findings: Optional list of finding strings, rendered as a bullet
            list beneath the metrics.
        title: The panel title. Defaults to "Analysis".
        subtitle: Optional subtitle shown in the bottom border.

    Returns:
        A Rich Panel object.
    """
    sections: list[RenderableType] = [Text(subject, style="bold bright_white")]

    if metrics:
        sections.append(Text(""))
        sections.append(_key_value_table(metrics))

    if findings:
        sections.append(Text(""))
        bullet_lines = Text()
        for index, finding in enumerate(findings):
            if index:
                bullet_lines.append("\n")
            bullet_lines.append("• ", style="bright_blue")
            bullet_lines.append(finding, style="bright_white")
        sections.append(bullet_lines)

    return _base_panel(Group(*sections), style=_STYLE_ANALYSIS, title=title, subtitle=subtitle)
