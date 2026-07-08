"""Central Rich theme for the Kodiak CLI.

This module is the single source of truth for Kodiak's visual identity:
the shared `Console` instance, the Rich `Theme`, the color palette,
icons/emoji constants, shared semantic styles, and border style
presets. Every UI component in `kodiak.cli.ui` should import its colors,
styles, and console from this module rather than defining its own.

This module contains no business logic and performs no I/O beyond
constructing the shared Console.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rich.box import Box, DOUBLE, HEAVY, ROUNDED
from rich.console import Console
from rich.theme import Theme

__all__ = [
    "Palette",
    "Icons",
    "Emoji",
    "Borders",
    "KODIAK_THEME",
    "console",
]


# --------------------------------------------------------------------------
# Color Palette
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Palette:
    """Kodiak's core color palette.

    Attributes:
        primary: The primary brand color, used for the logo and welcome
            banner (amber/yellow).
        success: Color used to indicate success states.
        error: Color used to indicate error states.
        warning: Color used to indicate warning states.
        info: Color used to indicate informational states.
        accent: Secondary accent color, used for tasks and highlights.
        analysis: Color used for analysis/review-oriented content.
        muted: Muted/dim color for secondary text.
        text: Default bright text color.
        border: Default neutral border color.
    """

    primary: str = "bright_yellow"
    success: str = "bright_green"
    error: str = "bright_red"
    warning: str = "bright_yellow"
    info: str = "bright_cyan"
    accent: str = "bright_magenta"
    analysis: str = "bright_blue"
    muted: str = "bright_black"
    text: str = "bright_white"
    border: str = "bright_black"


PALETTE: Final[Palette] = Palette()


# --------------------------------------------------------------------------
# Icons
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Icons:
    """Plain-text/unicode glyph icons used across CLI components.

    Attributes:
        success: Glyph used for success banners and panels.
        error: Glyph used for error banners and panels.
        warning: Glyph used for warning banners and panels.
        info: Glyph used for informational banners and panels.
        task: Glyph used for task banners.
        analysis: Glyph used for analysis banners and panels.
        summary: Glyph used for summary panels.
        review: Glyph used for review panels.
        bullet: Glyph used for bullet list items.
        arrow: Glyph used to indicate progression or direction.
    """

    success: str = "\u2714"  # ✔
    error: str = "\u2716"  # ✖
    warning: str = "\u26a0"  # ⚠
    info: str = "\u2139"  # ℹ
    task: str = "\u25b6"  # ▶
    analysis: str = "\u25c6"  # ◆
    summary: str = "\u2261"  # ≡
    review: str = "\U0001f50d"  # 🔍
    bullet: str = "\u2022"  # •
    arrow: str = "\u2192"  # →


ICONS: Final[Icons] = Icons()


@dataclass(frozen=True, slots=True)
class Emoji:
    """Emoji constants used sparingly for brand and celebratory moments.

    Attributes:
        bear: The Kodiak brand emoji, used in the welcome banner.
        rocket: Used for launch/deploy-related messaging.
        sparkles: Used for highlighting new or generated content.
        package: Used for build/packaging-related messaging.
        check_mark: Used for lightweight inline success indicators.
        cross_mark: Used for lightweight inline failure indicators.
    """

    bear: str = "\U0001f43b"  # 🐻
    rocket: str = "\U0001f680"  # 🚀
    sparkles: str = "\u2728"  # ✨
    package: str = "\U0001f4e6"  # 📦
    check_mark: str = "\u2705"  # ✅
    cross_mark: str = "\u274c"  # ❌


EMOJI: Final[Emoji] = Emoji()


# --------------------------------------------------------------------------
# Border Styles
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Borders:
    """Rich box presets used for panels across the CLI.

    Attributes:
        welcome: Box style used for the welcome banner (double line).
        emphasis: Box style used for error banners and other
            high-emphasis content (heavy line).
        standard: Default box style used for most panels and banners.
    """

    welcome: Box = DOUBLE
    emphasis: Box = HEAVY
    standard: Box = ROUNDED


BORDERS: Final[Borders] = Borders()


# --------------------------------------------------------------------------
# Shared Styles
# --------------------------------------------------------------------------

# Semantic style name -> Rich style string. These names are registered
# into the Rich Theme below and can be referenced directly in markup,
# e.g. console.print("[success]Done[/success]").
_STYLES: Final[dict[str, str]] = {
    # Semantic states
    "success": f"bold {PALETTE.success}",
    "error": f"bold {PALETTE.error}",
    "warning": f"bold {PALETTE.warning}",
    "info": f"bold {PALETTE.info}",
    "accent": f"bold {PALETTE.accent}",
    "analysis": f"bold {PALETTE.analysis}",
    # Text
    "text": PALETTE.text,
    "muted": f"dim {PALETTE.muted}",
    "heading": f"bold {PALETTE.text}",
    "subheading": f"italic {PALETTE.text}",
    # Brand
    "brand": f"bold {PALETTE.primary}",
    "brand.dim": f"dim {PALETTE.primary}",
    # Panel/table chrome
    "panel.border.success": PALETTE.success,
    "panel.border.error": PALETTE.error,
    "panel.border.warning": PALETTE.warning,
    "panel.border.info": PALETTE.info,
    "panel.border.accent": PALETTE.accent,
    "panel.border.analysis": PALETTE.analysis,
    "panel.border.default": PALETTE.border,
    "table.header": f"bold {PALETTE.text}",
    # Status labels (task/plan/review tables)
    "status.pending": "dim",
    "status.in_progress": PALETTE.warning,
    "status.done": PALETTE.success,
    "status.failed": PALETTE.error,
    "severity.critical": f"bold {PALETTE.error}",
    "severity.major": PALETTE.error,
    "severity.minor": PALETTE.warning,
    "severity.info": PALETTE.info,
}


# --------------------------------------------------------------------------
# Theme & Console
# --------------------------------------------------------------------------

KODIAK_THEME: Final[Theme] = Theme(_STYLES)
"""The shared Rich Theme registering all Kodiak semantic styles."""

console: Final[Console] = Console(theme=KODIAK_THEME)
"""The shared Console instance. All UI components should render through
this console (or a Console constructed with `KODIAK_THEME`) to ensure
consistent styling across the CLI."""