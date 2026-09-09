"""CLI presentation for repository-local approval records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from kodiak.orchestration.approval_manager import APPROVAL_STATUSES, ApprovalManager

app = typer.Typer(help="List and resolve repository-local approval requests.", no_args_is_help=True)
console = Console()
error_console = Console(stderr=True)


@app.command("list")
def list_approvals(
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repository path."),
    status: str | None = typer.Option(
        None, "--status", help="Filter by pending, approved, or rejected."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List local approvals, newest first."""
    if status is not None and status not in APPROVAL_STATUSES:
        _fail(f"Unknown approval status: {status}")
    try:
        approvals = ApprovalManager(path).list(status=status)
    except ValueError as exc:
        _fail(str(exc))
    if json_output:
        print(json.dumps(approvals, indent=2, default=str))
        return
    if not approvals:
        console.print("[dim]No approval requests found.[/dim]")
        return
    console.print("[bold]Kodiak approvals[/bold]")
    for item in approvals:
        console.print(
            Panel(
                f"Task: {item.get('task_id', '')}\n"
                f"Action: {item.get('action', '')}\n"
                f"Risk: {item.get('risk_level', '')}\n"
                f"Status: {item.get('status', '')}\n"
                f"Reason: {item.get('reason', '')}\n"
                f"Updated: {item.get('updated_at', '')}",
                title=str(item.get("approval_id", "")),
            )
        )


@app.command("approve")
def approve(
    approval_id: str = typer.Argument(..., help="Approval request ID."),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repository path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Mark one pending request approved; this does not execute the action."""
    _resolve(path, approval_id, "approved", json_output)


@app.command("reject")
def reject(
    approval_id: str = typer.Argument(..., help="Approval request ID."),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repository path."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Mark one pending request rejected; no requested action is executed."""
    _resolve(path, approval_id, "rejected", json_output)


@app.command("execute")
def execute(
    approval_id: str = typer.Argument(..., help="Approved request ID."),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repository path."),
) -> None:
    """Execute one explicitly approved local commit request."""
    try:
        ApprovalManager(path).execute_approved_commit(approval_id)
    except (LookupError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    console.print("[bold green]Approved local commit created.[/bold green]")


def _resolve(path: Path, approval_id: str, status: str, json_output: bool) -> None:
    try:
        item = ApprovalManager(path).resolve(approval_id, status)
    except (LookupError, RuntimeError, ValueError) as exc:
        _fail(str(exc))
    if json_output:
        print(json.dumps(item, indent=2, default=str))
    else:
        console.print(
            f"[bold green]{item['approval_id']} marked {item['status']}.[/bold green] "
            "No action was executed automatically."
        )


def _fail(message: str) -> Any:
    error_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)
