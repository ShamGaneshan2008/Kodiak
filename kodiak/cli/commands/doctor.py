"""CLI presentation for sanitized Kodiak installation diagnostics."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from kodiak.cli.services.doctor_service import DoctorService

console = Console()


def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit sanitized JSON."),
) -> None:
    """Check local tools, repository state, and optional provider configuration."""
    service = DoctorService()
    checks = service.run()
    if json_output:
        console.print_json(json.dumps(service.as_dict()))
    else:
        table = Table(title="Kodiak diagnostics")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for item in checks:
            table.add_row(item.name, item.status.upper(), item.detail)
        console.print(table)
    if any(item.required and item.status == "fail" for item in checks):
        raise typer.Exit(code=1)
