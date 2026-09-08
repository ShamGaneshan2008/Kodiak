"""Read-only Git CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kodiak.cli.services.git_service import GitService

app = typer.Typer(help="Inspect local Git state without mutating it.", no_args_is_help=True)
console = Console()
error_console = Console(stderr=True)


@app.command("diff-summary")
def diff_summary(
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repository path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show branch, dirty state, changed-file groups, and diff statistics."""
    summary = GitService().diff_summary(path)
    if json_output:
        print(json.dumps(summary.to_dict(), indent=2, default=str))
        return
    if not summary.is_git_repo:
        error_console.print(f"[bold yellow]Not a Git repository:[/bold yellow] {summary.error}")
        return
    table = Table(title="Git diff summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Repository", summary.repository_path)
    table.add_row("Current branch", summary.current_branch or "unknown")
    table.add_row("Dirty", "yes" if summary.dirty else "no")
    table.add_row("Staged files", _paths(summary.staged_files))
    table.add_row("Unstaged files", _paths(summary.unstaged_files))
    table.add_row("Untracked files", _paths(summary.untracked_files))
    table.add_row("Changed files", _paths(summary.changed_files))
    table.add_row("Diff stat", summary.diff_stat or "no tracked diff")
    console.print(table)


def _paths(paths: tuple[str, ...]) -> str:
    return "\n".join(paths) if paths else "-"
