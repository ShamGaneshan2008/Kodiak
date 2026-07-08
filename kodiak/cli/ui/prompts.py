"""Reusable Rich prompt components for the Kodiak CLI.

This module is a pure presentation layer: it wraps `rich.prompt`
classes (Prompt, Confirm, IntPrompt) into small, reusable helper
functions for collecting user input (issues, confirmations, repository
selection, API keys, passwords, and multi-line text). Nothing here
contains business logic, validation against external services, or
Typer dependencies — callers are responsible for acting on the
returned values.
"""

from __future__ import annotations

from typing import Final

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

__all__ = [
    "prompt_github_issue_number",
    "confirm_action",
    "prompt_yes_no",
    "prompt_repository_selection",
    "prompt_api_key",
    "prompt_password",
    "prompt_multiline_issue",
]

console: Final[Console] = Console()

_PROMPT_STYLE: Final[str] = "bold bright_cyan"
_HINT_STYLE: Final[str] = "dim"


def prompt_github_issue_number(
    message: str = "Enter the GitHub issue number or URL",
) -> str:
    """Prompt the user for a GitHub issue reference.

    Args:
        message: The prompt text shown to the user.

    Returns:
        The raw string entered by the user (issue number or URL), with
        surrounding whitespace stripped.
    """
    value = Prompt.ask(f"[{_PROMPT_STYLE}]{message}[/{_PROMPT_STYLE}]", console=console)
    return value.strip()


def confirm_action(
    message: str,
    *,
    default: bool = True,
) -> bool:
    """Ask the user to confirm an action before proceeding.

    Args:
        message: The description of the action to confirm.
        default: The default answer used if the user presses Enter.

    Returns:
        True if the user confirmed, False otherwise.
    """
    return Confirm.ask(
        f"[{_PROMPT_STYLE}]{message}[/{_PROMPT_STYLE}]",
        default=default,
        console=console,
    )


def prompt_yes_no(
    question: str,
    *,
    default: bool | None = None,
) -> bool:
    """Ask the user a simple yes/no question.

    Args:
        question: The question text to display.
        default: The default answer used if the user presses Enter.
            Pass ``None`` to require an explicit answer.

    Returns:
        True for "yes", False for "no".
    """
    return Confirm.ask(
        f"[{_PROMPT_STYLE}]{question}[/{_PROMPT_STYLE}]",
        default=default,
        console=console,
    )


def prompt_repository_selection(
    repositories: list[str],
    *,
    message: str = "Select a repository",
) -> str:
    """Prompt the user to select a repository from a list of choices.

    Args:
        repositories: The available repository names or slugs to choose
            from (e.g. ``["owner/repo-a", "owner/repo-b"]``).
        message: The prompt text shown above the choices.

    Returns:
        The repository string selected by the user.

    Raises:
        ValueError: If ``repositories`` is empty.
    """
    if not repositories:
        raise ValueError("repositories must contain at least one entry.")

    console.print(Text(message, style=_PROMPT_STYLE))
    for index, repo in enumerate(repositories, start=1):
        console.print(f"  [{_HINT_STYLE}]{index}.[/{_HINT_STYLE}] {repo}")

    choice = Prompt.ask(
        f"[{_PROMPT_STYLE}]Enter a number[/{_PROMPT_STYLE}]",
        choices=[str(i) for i in range(1, len(repositories) + 1)],
        show_choices=False,
        console=console,
    )
    return repositories[int(choice) - 1]


def prompt_api_key(
    provider: str = "API",
    *,
    message: str | None = None,
) -> str:
    """Prompt the user for an API key, masking the input.

    Args:
        provider: The name of the provider the key belongs to, used to
            build a default message (e.g. "OpenAI", "GitHub").
        message: Optional custom prompt text overriding the default
            provider-based message.

    Returns:
        The API key string entered by the user, with surrounding
        whitespace stripped.
    """
    text = message or f"Enter your {provider} API key"
    value = Prompt.ask(
        f"[{_PROMPT_STYLE}]{text}[/{_PROMPT_STYLE}]",
        password=True,
        console=console,
    )
    return value.strip()


def prompt_password(
    message: str = "Enter password",
    *,
    confirm: bool = False,
) -> str:
    """Prompt the user for a password, masking the input.

    Args:
        message: The prompt text shown to the user.
        confirm: If True, prompt a second time and re-prompt until the
            two entries match.

    Returns:
        The password string entered by the user.
    """
    while True:
        password = Prompt.ask(
            f"[{_PROMPT_STYLE}]{message}[/{_PROMPT_STYLE}]",
            password=True,
            console=console,
        )
        if not confirm:
            return password

        confirmation = Prompt.ask(
            f"[{_PROMPT_STYLE}]Confirm password[/{_PROMPT_STYLE}]",
            password=True,
            console=console,
        )
        if password == confirmation:
            return password

        console.print("[bold bright_red]Passwords do not match. Please try again.[/bold bright_red]")


def prompt_multiline_issue(
    message: str = "Describe the issue (submit an empty line to finish)",
) -> str:
    """Prompt the user for multi-line free-text issue input.

    Args:
        message: The instruction text shown before input begins.

    Returns:
        The collected multi-line text, joined with newlines and with
        trailing whitespace stripped.
    """
    console.print(
        Panel(
            Text(message, style=_PROMPT_STYLE),
            border_style="bright_cyan",
            title="Issue Description",
            title_align="left",
        )
    )

    lines: list[str] = []
    while True:
        line = Prompt.ask("", console=console)
        if line == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()