"""Kodiak Logs Command.

Provides access to Kodiak platform logs with filtering, tailing,
and real-time following capabilities.

This command follows the layered architecture:
    CLI → Validation → CLI Service → Backend → Output
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator, Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="logs",
    help="View and follow Kodiak platform logs.",
    no_args_is_help=False,
)


class LogLevel(str, Enum):
    """Supported log levels for filtering."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class LogEntry:
    """Represents a single log entry.

    Attributes:
        timestamp: When the log was created.
        level: The severity level of the log.
        message: The log message content.
        source: The component that generated the log.
        metadata: Additional structured data attached to the log.
    """

    timestamp: datetime
    level: str
    message: str
    source: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert log entry to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "metadata": self.metadata,
        }


class LogsService(ABC):
    """Abstract interface for logs service.

    This defines the contract that the backend service must fulfill.
    The actual implementation will be provided by the backend layer.
    """

    @abstractmethod
    def get_logs(
        self,
        tail: int | None = None,
        level: LogLevel | None = None,
    ) -> Sequence[LogEntry]:
        """Retrieve log entries.

        Args:
            tail: If provided, return only the last N entries.
            level: If provided, filter to only this log level.

        Returns:
            A sequence of log entries matching the criteria.
        """

    @abstractmethod
    def follow_logs(
        self,
        level: LogLevel | None = None,
    ) -> Iterator[LogEntry]:
        """Stream log entries in real-time.

        Args:
            level: If provided, filter to only this log level.

        Yields:
            Log entries as they are produced.
        """


class StubLogsService(LogsService):
    """Stub implementation for when backend is unavailable.

    This allows the CLI command to function and demonstrate
    its interface while the backend is being developed.

    Replace this with the real implementation when available.
    """

    def __init__(self) -> None:
        """Initialize with sample log data."""
        self._sample_logs: list[LogEntry] = [
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:00"),
                level="info",
                message="Kodiak CLI initialized successfully",
                source="cli.core",
                metadata={"version": "0.1.0"},
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:01"),
                level="debug",
                message="Loading configuration from ~/.kodiak/config.toml",
                source="cli.config",
                metadata={"path": "~/.kodiak/config.toml"},
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:02"),
                level="info",
                message="Authentication token validated",
                source="cli.auth",
                metadata={"provider": "github"},
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:05"),
                level="warning",
                message="Rate limit approaching: 4500/5000 requests remaining",
                source="api.github",
                metadata={
                    "remaining": 4500,
                    "limit": 5000,
                    "reset": "2024-01-15T11:30:00",
                },
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:10"),
                level="info",
                message="Repository analysis started",
                source="agents.analyzer",
                metadata={"repository": "owner/repo", "task_id": "task_abc123"},
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:15"),
                level="error",
                message="Failed to fetch branch metadata: 403 Forbidden",
                source="api.github",
                metadata={
                    "error_code": 403,
                    "branch": "main",
                    "repository": "owner/repo",
                },
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:20"),
                level="info",
                message="Falling back to cached repository data",
                source="agents.analyzer",
                metadata={"cache_age": "2h 15m"},
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:25"),
                level="debug",
                message="Parsing AST for 142 files",
                source="agents.analyzer",
                metadata={
                    "file_count": 142,
                    "language_distribution": {"python": 98, "typescript": 44},
                },
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:30"),
                level="info",
                message="Repository analysis completed",
                source="agents.analyzer",
                metadata={"duration_seconds": 25.3, "files_analyzed": 142},
            ),
            LogEntry(
                timestamp=datetime.fromisoformat("2024-01-15T10:30:35"),
                level="warning",
                message="Memory usage above threshold: 83% of 4GB limit",
                source="system.monitor",
                metadata={"used_mb": 3400, "limit_mb": 4096, "percentage": 83.0},
            ),
        ]

    def get_logs(
        self,
        tail: int | None = None,
        level: LogLevel | None = None,
    ) -> Sequence[LogEntry]:
        """Return sample logs, optionally filtered."""
        logs = self._sample_logs

        if level is not None:
            logs = [log for log in logs if log.level == level.value]

        if tail is not None:
            logs = logs[-tail:]

        return logs

    def follow_logs(
        self,
        level: LogLevel | None = None,
    ) -> Iterator[LogEntry]:
        """Yield sample logs one by one to simulate streaming."""
        logs = self._sample_logs

        if level is not None:
            logs = [log for log in logs if log.level == level.value]

        for log in logs:
            yield log


def get_logs_service() -> LogsService:
    """Factory function to retrieve the logs service.

    In production, this would inject the real backend service.
    Currently returns a stub for development.

    Returns:
        An instance of LogsService.
    """
    return StubLogsService()


def validate_tail(value: int) -> int:
    """Validate the tail argument value.

    Args:
        value: The number of lines to show.

    Returns:
        The validated value.

    Raises:
        typer.BadParameter: If value is not positive or exceeds maximum.
    """
    if value < 1:
        raise typer.BadParameter("Must be a positive integer greater than 0.")
    if value > 10000:
        raise typer.BadParameter("Maximum allowed value is 10000.")
    return value


def create_log_table(entries: Sequence[LogEntry]) -> Table:
    """Create a Rich table for displaying log entries.

    Args:
        entries: The log entries to display.

    Returns:
        A configured Rich Table instance.
    """
    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        padding=(0, 1),
        expand=True,
    )

    table.add_column("TIMESTAMP", style="dim", width=19, no_wrap=True)
    table.add_column("LEVEL", width=7, no_wrap=True)
    table.add_column("SOURCE", style="cyan", width=20, no_wrap=True)
    table.add_column("MESSAGE", width=None)

    level_styles: dict[str, str] = {
        "debug": "dim",
        "info": "green",
        "warning": "yellow",
        "error": "red bold",
    }

    for entry in entries:
        style = level_styles.get(entry.level, "")
        table.add_row(
            entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            Text(entry.level.upper(), style=style),
            entry.source,
            entry.message,
        )

    return table


def format_entry_json(entry: LogEntry) -> str:
    """Format a log entry as JSON.

    Args:
        entry: The log entry to format.

    Returns:
        JSON string representation.
    """
    return json.dumps(entry.to_dict(), ensure_ascii=False)


def create_summary_panel(
    entry_count: int,
    level_filter: LogLevel | None,
) -> Panel:
    """Create a summary panel for the log output.

    Args:
        entry_count: Number of entries being displayed.
        level_filter: The level filter applied, if any.

    Returns:
        A Rich Panel with summary information.
    """
    filter_text = (
        f" [dim](filtered: {level_filter.value})[/dim]" if level_filter else ""
    )
    summary_text = Text()
    summary_text.append("Showing ", style="dim")
    summary_text.append(str(entry_count), style="bold cyan")
    summary_text.append(f" log entries{filter_text}", style="dim")

    return Panel(
        summary_text,
        border_style="dim",
        padding=(0, 1),
    )


@app.callback(invoke_without_command=True)
def logs(
    ctx: typer.Context,
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Follow log output in real-time.",
    ),
    tail: int | None = typer.Option(
        None,
        "--tail",
        "-n",
        help="Show only the last N log entries.",
        callback=validate_tail,
    ),
    level: LogLevel | None = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter logs by level.",
        case_sensitive=False,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output logs in JSON format.",
    ),
) -> int:
    """View and follow Kodiak platform logs.

    Display logs from the Kodiak platform with optional filtering by level,
    tailing to show recent entries, or following for real-time updates.

    Examples:
        # Show all logs
        kodiak logs

        # Show last 20 logs
        kodiak logs --tail 20

        # Show only error logs
        kodiak logs --level error

        # Follow logs in real-time
        kodiak logs --follow

        # Follow only errors in JSON format
        kodiak logs --follow --level error --json

    Exit Codes:
        0: Success
        1: Error occurred
    """
    console = Console()
    exit_code = 0

    try:
        service = get_logs_service()

        if follow:
            _handle_follow_mode(console, service, level, json_output)
        else:
            _handle_static_mode(console, service, tail, level, json_output)

    except KeyboardInterrupt:
        console.print()
        console.print("[dim]Interrupted.[/dim]")
        exit_code = 0

    except typer.BadParameter:
        raise

    except Exception:
        console.print()
        console.print(
            Panel(
                "[red]Failed to retrieve logs.[/red]\n\n"
                "[dim]Run [bold]kodiak doctor[/bold] to check system status.[/dim]",
                title="[red]Error[/red]",
                border_style="red",
                padding=(1, 2),
            )
        )
        exit_code = 1

    return exit_code


def _handle_static_mode(
    console: Console,
    service: LogsService,
    tail: int | None,
    level: LogLevel | None,
    json_output: bool,
) -> None:
    """Handle static (non-following) log display.

    Args:
        console: The Rich console instance.
        service: The logs service.
        tail: Optional tail count.
        level: Optional level filter.
        json_output: Whether to output as JSON.
    """
    with console.status("[dim]Loading logs...[/dim]", spinner="dots"):
        entries = service.get_logs(tail=tail, level=level)

    if not entries:
        console.print()
        console.print(
            Panel(
                "[dim]No log entries found matching the specified criteria.[/dim]\n\n"
                "[dim]Try removing filters or check [bold]kodiak doctor[/bold] "
                "for issues.[/dim]",
                title="No Logs",
                border_style="dim",
                padding=(1, 2),
            )
        )
        return

    console.print()

    if json_output:
        for entry in entries:
            console.print(format_entry_json(entry))
    else:
        console.print(create_summary_panel(len(entries), level))
        console.print()
        console.print(create_log_table(entries))

    console.print()


def _handle_follow_mode(
    console: Console,
    service: LogsService,
    level: LogLevel | None,
    json_output: bool,
) -> None:
    """Handle real-time log following.

    Args:
        console: The Rich console instance.
        service: The logs service.
        level: Optional level filter.
        json_output: Whether to output as JSON.
    """
    level_styles: dict[str, str] = {
        "debug": "dim",
        "info": "green",
        "warning": "yellow",
        "error": "red bold",
    }

    console.print()
    console.print(
        Panel(
            "[dim]Following logs... Press [bold]Ctrl+C[/bold] to stop.[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    console.print()

    for entry in service.follow_logs(level=level):
        if json_output:
            console.print(format_entry_json(entry))
        else:
            timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            level_style = level_styles.get(entry.level, "")
            console.print(
                f"[dim]{timestamp_str}[/dim] "
                f"[{level_style}]{entry.level.upper():<7}[/{level_style}] "
                f"[cyan]{entry.source:<20}[/cyan] "
                f"{entry.message}"
            )


if __name__ == "__main__":
    app()