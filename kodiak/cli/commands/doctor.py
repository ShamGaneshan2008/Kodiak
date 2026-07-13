"""CLI command for verifying the health of a Kodiak installation.

This module contains presentation logic only. All checks (Python version,
Git, Docker, API keys, connectivity, GitHub auth, configuration,
workspace permissions, and sandbox availability) are performed by
:class:`kodiak.services.doctor_service.DoctorService`.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kodiak.cli.schemas.doctor import CheckResult, CheckStatus, DoctorReport
from kodiak.cli.services.doctor_service import DoctorService

console = Console()
error_console = Console(stderr=True)

app = typer.Typer()

_STATUS_STYLES: dict[CheckStatus, tuple[str, str]] = {
    CheckStatus.PASSED: ("\u2713", "green"),
    CheckStatus.WARNING: ("\u26a0", "yellow"),
    CheckStatus.FAILED: ("\u2717", "red"),
}


@app.command("doctor")
def doctor() -> None:
    """Verify that the local Kodiak installation is healthy.

    Runs a series of environment checks, including Python version, Git
    and Docker availability, API key configuration, internet
    connectivity, GitHub authentication, the Kodiak configuration file,
    workspace permissions, and sandbox availability. Results are
    displayed in a Rich table, with recommendations shown for any check
    that did not pass cleanly.

    Example:
        kodiak doctor
    """
    service = DoctorService()

    try:
        report = asyncio.run(_run_checks(service))
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Failed to run health checks: {exc}")
        raise typer.Exit(code=2)

    _render_report(report)

    if report.has_failures:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


async def _run_checks(service: DoctorService) -> DoctorReport:
    """Run all health checks via DoctorService.

    Args:
        service: The service responsible for performing environment
            checks.

    Returns:
        A structured report containing the result of every check.
    """
    with console.status("[bold cyan]Running health checks...", spinner="dots"):
        return await service.run_checks()


def _render_report(report: DoctorReport) -> None:
    """Render a doctor report as a Rich table with a summary panel.

    Args:
        report: The structured result of all health checks.
    """
    table = Table(title="Kodiak Doctor")
    table.add_column("Status", justify="center", width=6)
    table.add_column("Check", style="bold")
    table.add_column("Message")

    for check in report.checks:
        symbol, style = _STATUS_STYLES[check.status]
        table.add_row(f"[{style}]{symbol}[/{style}]", check.name, check.message)

    console.print(table)

    failures = [c for c in report.checks if c.status == CheckStatus.FAILED]
    warnings = [c for c in report.checks if c.status == CheckStatus.WARNING]

    if failures or warnings:
        _render_recommendations(failures, warnings)
    else:
        console.print(
            Panel(
                "All checks passed. Kodiak is ready to use.",
                title="Summary",
                border_style="green",
            )
        )


def _render_recommendations(
    failures: list[CheckResult], warnings: list[CheckResult]
) -> None:
    """Render recommendations for failed and warning checks.

    Args:
        failures: Checks that failed outright.
        warnings: Checks that passed with a caveat.
    """
    lines: list[str] = []

    for check in failures:
        recommendation = check.recommendation or "No recommendation available."
        lines.append(f"[red]\u2717 {check.name}[/red]: {recommendation}")

    for check in warnings:
        recommendation = check.recommendation or "No recommendation available."
        lines.append(f"[yellow]\u26a0 {check.name}[/yellow]: {recommendation}")

    border_style = "red" if failures else "yellow"
    title = "Action Required" if failures else "Recommendations"

    console.print(
        Panel(
            "\n".join(lines),
            title=title,
            border_style=border_style,
            title_align="left",
        )
    )


def _render_error(message: str) -> None:
    """Render an error message inside a red Rich panel.

    Args:
        message: Human-readable error description.
    """
    error_console.print(
        Panel(
            message,
            title="Doctor Failed",
            border_style="red",
            title_align="left",
        )
    )