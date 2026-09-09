"""CLI entrypoint for the deterministic local Kodiak v1 task workflow."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kodiak.orchestration.task_orchestrator import (
    InvalidRepositoryError,
    TaskOrchestrator,
    TaskRunResult,
)

app = typer.Typer(
    help="Plan and run bounded, approval-gated local coding tasks.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


@app.command("run")
def run_task(
    instruction: str = typer.Argument(..., help="Coding task to plan and run."),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repository path."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan and preview changes without source-file writes."
    ),
    no_commit: bool = typer.Option(
        False, "--no-commit", help="Apply supported edits but never create a Git commit."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Analyze, plan, safely edit, check, review, and record one local task."""
    try:
        result: TaskRunResult = TaskOrchestrator().run(
            instruction,
            path,
            dry_run=dry_run,
            no_commit=no_commit,
        )
    except (InvalidRepositoryError, ValueError) as exc:
        error_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _render(result)
    if result.final_status in {"error", "failed_checks", "manual_required"}:
        raise typer.Exit(code=1)


def _render(result: TaskRunResult) -> None:
    plan = result.plan
    if plan is not None:
        plan_table = Table(title="Task plan")
        plan_table.add_column("Step")
        plan_table.add_column("Action")
        for number, step in enumerate(plan.steps, start=1):
            plan_table.add_row(str(number), step)
        console.print(plan_table)

    if result.proposed_changes:
        for change in result.proposed_changes:
            console.print(
                Panel(
                    change.to_dict()["diff"] or "No textual difference.",
                    title=f"Proposed change: {change.path}",
                    border_style="cyan",
                )
            )
    else:
        console.print("[dim]No file changes were proposed.[/dim]")

    if result.checks:
        checks = Table(title="Checks")
        checks.add_column("Check")
        checks.add_column("Status")
        checks.add_column("Summary")
        for check in result.checks:
            checks.add_row(check.name, check.status, check.summary)
        console.print(checks)

    mode = "dry-run" if result.dry_run else "no-commit" if result.no_commit else "standard"
    details = [
        f"[bold]Task ID:[/bold] {result.task_id}",
        f"[bold]Status:[/bold] {result.final_status}",
        f"[bold]Mode:[/bold] {mode}",
        f"[bold]Changed files:[/bold] {', '.join(result.changed_files) or 'none'}",
        f"[bold]Review:[/bold] {result.review_summary}",
    ]
    if result.approval_id:
        details.append(f"[bold]Approval:[/bold] {result.approval_id}")
    if result.error:
        details.append(f"[bold red]Error:[/bold red] {result.error}")
    if result.dry_run:
        details.append("[bold]Dry run:[/bold] no source files were changed.")
    elif result.no_commit:
        details.append("[bold]Commit:[/bold] disabled; no commit was created.")
        details.append(
            f'[bold]Inspect:[/bold] kodiak git diff-summary --path "{result.repository_path}"'
        )
        details.append(f'[bold]Inspect:[/bold] git -C "{result.repository_path}" diff --')
    console.print(Panel("\n".join(details), title="Kodiak task report", border_style="green"))
