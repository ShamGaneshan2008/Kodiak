"""Reusable spinner components for the Kodiak CLI.

This module is a pure presentation layer: it wraps `rich.status.Status`
into small, reusable helpers and context managers for common Kodiak
workflow stages (repository analysis, planning, task execution, AI
reasoning, repository indexing, running tests, formatting, creating
commits, and creating pull requests).

Nothing in this module contains business logic. It renders exclusively
through Rich's `Status`/`Live` APIs and the shared console defined in
`kodiak.cli.ui.console` — it never calls `print` directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

from rich.console import Console
from rich.status import Status

from kodiak.cli.ui.console import get_console

__all__ = [
    "SpinnerPreset",
    "SPINNER_PRESETS",
    "Spinner",
    "SpinnerManager",
    "create_spinner",
    "start_spinner",
    "stop_spinner",
    "update_message",
    "repository_analysis_spinner",
    "planning_spinner",
    "task_execution_spinner",
    "ai_reasoning_spinner",
    "repository_indexing_spinner",
    "test_run_spinner",
    "formatting_spinner",
    "commit_spinner",
    "pull_request_spinner",
]

_DEFAULT_SPINNER_NAME: Final[str] = "dots"


@dataclass(frozen=True, slots=True)
class SpinnerPreset:
    """A named spinner configuration for a specific workflow stage.

    Attributes:
        message: The default status message shown while active.
        spinner: The Rich spinner animation name (e.g. "dots", "line").
        style: The Rich color/style string applied to the spinner glyph.
    """

    message: str
    spinner: str = _DEFAULT_SPINNER_NAME
    style: str = "bright_yellow"


SPINNER_PRESETS: Final[dict[str, SpinnerPreset]] = {
    "repository_analysis": SpinnerPreset(
        message="Analyzing repository...", spinner="dots", style="bright_cyan"
    ),
    "planning": SpinnerPreset(message="Planning...", spinner="dots", style="bright_magenta"),
    "task_execution": SpinnerPreset(
        message="Executing task...", spinner="dots", style="bright_yellow"
    ),
    "ai_reasoning": SpinnerPreset(message="Reasoning...", spinner="dots", style="bright_magenta"),
    "repository_indexing": SpinnerPreset(
        message="Indexing repository...", spinner="dots", style="bright_cyan"
    ),
    "test_run": SpinnerPreset(message="Running tests...", spinner="dots", style="bright_blue"),
    "formatting": SpinnerPreset(message="Formatting code...", spinner="dots", style="bright_green"),
    "commit": SpinnerPreset(message="Creating commit...", spinner="dots", style="bright_green"),
    "pull_request": SpinnerPreset(
        message="Creating pull request...", spinner="dots", style="bright_yellow"
    ),
}


class Spinner:
    """A thin, reusable wrapper around `rich.status.Status`.

    This class manages the lifecycle of a single Rich `Status` instance:
    starting it, updating its message, and stopping it. It never prints
    directly; all rendering is delegated to the underlying `Status`.
    """

    def __init__(
        self,
        message: str,
        *,
        console: Console | None = None,
        spinner: str = _DEFAULT_SPINNER_NAME,
        style: str = "bright_yellow",
    ) -> None:
        """Initialize the spinner.

        Args:
            message: The initial status message to display.
            console: The console to render into. Defaults to the shared
                Kodiak console.
            spinner: The Rich spinner animation name.
            style: The Rich color/style string applied to the spinner.
        """
        self._console = console or get_console()
        self._message = message
        self._spinner_name = spinner
        self._style = style
        self._status: Status | None = None

    @property
    def message(self) -> str:
        """The current status message.

        Returns:
            The most recently set status message.
        """
        return self._message

    @property
    def is_active(self) -> bool:
        """Whether the spinner is currently running.

        Returns:
            True if the spinner has been started and not yet stopped.
        """
        return self._status is not None

    def start(self) -> Spinner:
        """Start rendering the spinner.

        Returns:
            This Spinner instance, to allow chaining.
        """
        if self._status is None:
            self._status = self._console.status(
                self._message,
                spinner=self._spinner_name,
                spinner_style=self._style,
            )
            self._status.start()
        return self

    def stop(self) -> None:
        """Stop rendering the spinner, if active."""
        if self._status is not None:
            self._status.stop()
            self._status = None

    def update(
        self,
        message: str,
        *,
        spinner: str | None = None,
        style: str | None = None,
    ) -> None:
        """Update the spinner's message and optionally its appearance.

        Args:
            message: The new status message to display.
            spinner: Optional new spinner animation name.
            style: Optional new spinner color/style string.

        Raises:
            RuntimeError: If the spinner has not been started.
        """
        if self._status is None:
            raise RuntimeError("Cannot update a spinner that has not been started.")

        self._message = message
        self._status.update(
            status=message,
            spinner=spinner or self._spinner_name,
            spinner_style=style or self._style,
        )
        if spinner is not None:
            self._spinner_name = spinner
        if style is not None:
            self._style = style

    def __enter__(self) -> Spinner:
        """Start the spinner as a context manager.

        Returns:
            This Spinner instance.
        """
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        """Stop the spinner when exiting the context manager."""
        self.stop()


@dataclass(slots=True)
class SpinnerManager:
    """A registry for managing multiple named spinners.

    Only one spinner is intended to be visually active at a time (Rich
    does not support overlapping `Status` renders on the same console),
    but this manager allows callers to create, look up, and stop
    spinners by name across a workflow with several sequential stages.

    Attributes:
        console: The console shared by spinners created through this
            manager.
    """

    console: Console = field(default_factory=get_console)
    _spinners: dict[str, Spinner] = field(default_factory=dict, init=False, repr=False)

    def create(
        self,
        name: str,
        message: str,
        *,
        spinner: str = _DEFAULT_SPINNER_NAME,
        style: str = "bright_yellow",
    ) -> Spinner:
        """Create and register a new spinner under the given name.

        Args:
            name: A unique key identifying this spinner within the
                manager.
            message: The initial status message to display.
            spinner: The Rich spinner animation name.
            style: The Rich color/style string applied to the spinner.

        Returns:
            The newly created Spinner instance (not yet started).
        """
        instance = Spinner(message, console=self.console, spinner=spinner, style=style)
        self._spinners[name] = instance
        return instance

    def get(self, name: str) -> Spinner | None:
        """Retrieve a previously created spinner by name.

        Args:
            name: The key the spinner was registered under.

        Returns:
            The Spinner instance, or None if no spinner is registered
            under that name.
        """
        return self._spinners.get(name)

    def start(self, name: str) -> Spinner:
        """Start a registered spinner by name.

        Args:
            name: The key the spinner was registered under.

        Returns:
            The started Spinner instance.

        Raises:
            KeyError: If no spinner is registered under `name`.
        """
        return self._spinners[name].start()

    def stop(self, name: str) -> None:
        """Stop a registered spinner by name, if it exists.

        Args:
            name: The key the spinner was registered under.
        """
        instance = self._spinners.get(name)
        if instance is not None:
            instance.stop()

    def stop_all(self) -> None:
        """Stop every spinner currently registered with this manager."""
        for instance in self._spinners.values():
            instance.stop()

    def update(self, name: str, message: str) -> None:
        """Update the message of a registered, active spinner.

        Args:
            name: The key the spinner was registered under.
            message: The new status message to display.

        Raises:
            KeyError: If no spinner is registered under `name`.
        """
        self._spinners[name].update(message)


def create_spinner(
    message: str,
    *,
    console: Console | None = None,
    spinner: str = _DEFAULT_SPINNER_NAME,
    style: str = "bright_yellow",
) -> Spinner:
    """Create a standalone Spinner instance without starting it.

    Args:
        message: The initial status message to display.
        console: The console to render into. Defaults to the shared
            Kodiak console.
        spinner: The Rich spinner animation name.
        style: The Rich color/style string applied to the spinner.

    Returns:
        A new, unstarted Spinner instance.
    """
    return Spinner(message, console=console, spinner=spinner, style=style)


def start_spinner(spinner_instance: Spinner) -> Spinner:
    """Start a previously created Spinner instance.

    Args:
        spinner_instance: The Spinner to start.

    Returns:
        The same Spinner instance, now active.
    """
    return spinner_instance.start()


def stop_spinner(spinner_instance: Spinner) -> None:
    """Stop a previously started Spinner instance.

    Args:
        spinner_instance: The Spinner to stop.
    """
    spinner_instance.stop()


def update_message(
    spinner_instance: Spinner,
    message: str,
    *,
    spinner: str | None = None,
    style: str | None = None,
) -> None:
    """Update the message of an active Spinner instance.

    Args:
        spinner_instance: The Spinner to update.
        message: The new status message to display.
        spinner: Optional new spinner animation name.
        style: Optional new spinner color/style string.
    """
    spinner_instance.update(message, spinner=spinner, style=style)


def _preset_spinner(preset_key: str, *, message: str | None = None) -> Spinner:
    """Build a Spinner from a named preset configuration.

    Args:
        preset_key: The key into `SPINNER_PRESETS`.
        message: An optional message overriding the preset's default.

    Returns:
        A new, unstarted Spinner instance configured from the preset.
    """
    preset = SPINNER_PRESETS[preset_key]
    return Spinner(
        message or preset.message,
        spinner=preset.spinner,
        style=preset.style,
    )


@contextmanager
def repository_analysis_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for the repository analysis stage.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("repository_analysis", message=message) as instance:
        yield instance


@contextmanager
def planning_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for the planning stage.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("planning", message=message) as instance:
        yield instance


@contextmanager
def task_execution_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for general task execution.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("task_execution", message=message) as instance:
        yield instance


@contextmanager
def ai_reasoning_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for the AI reasoning stage.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("ai_reasoning", message=message) as instance:
        yield instance


@contextmanager
def repository_indexing_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for the repository indexing stage.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("repository_indexing", message=message) as instance:
        yield instance


@contextmanager
def test_run_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for running tests.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("test_run", message=message) as instance:
        yield instance


@contextmanager
def formatting_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for code formatting.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("formatting", message=message) as instance:
        yield instance


@contextmanager
def commit_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for creating a commit.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("commit", message=message) as instance:
        yield instance


@contextmanager
def pull_request_spinner(message: str | None = None) -> Iterator[Spinner]:
    """Provide a spinner for creating a pull request.

    Args:
        message: Optional message overriding the preset default.

    Yields:
        The active Spinner instance.
    """
    with _preset_spinner("pull_request", message=message) as instance:
        yield instance
