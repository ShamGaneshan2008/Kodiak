"""
kodiak review
=============

CLI entrypoint for reviewing Git changes before committing.

This module is presentation-only: it parses arguments, resolves them into a
``ReviewTarget``, delegates the actual analysis to
:class:`kodiak.review.service.ReviewService`, and renders the resulting
``ReviewResult`` either as a Rich report or as JSON. No bug/security/
performance/readability heuristics live here -- that logic belongs entirely
to the review service and its underlying analyzers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress as RichProgress
from rich.progress import SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from kodiak.review.errors import ReviewServiceError, ReviewTargetError
from kodiak.review.models import (
    Finding,
    ReviewResult,
    ReviewTarget,
    Severity,
)
from kodiak.review.service import ReviewService

logger = structlog.get_logger(__name__)
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="review",
    help="Review Git changes for bugs, security issues, performance, and style.",
    no_args_is_help=False,
    add_completion=False,
)

# Presentation constants


_SEVERITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
]

_SEVERITY_STYLE = {
    Severity.CRITICAL: ("🔴", "bold red"),
    Severity.HIGH: ("🟠", "bold dark_orange"),
    Severity.MEDIUM: ("🟡", "bold yellow"),
    Severity.LOW: ("🔵", "cyan"),
    Severity.INFO: ("⚪", "dim"),
}

_SECTION_SPECS: list[tuple[str, str, str]] = [
    # (attribute on ReviewResult, section title, empty-state message)
    ("bugs", "Bugs", "No bugs found."),
    ("security_issues", "Security Issues", "No security issues found."),
    ("performance_issues", "Performance", "No performance issues found."),
    ("readability_issues", "Readability", "No readability issues found."),
    ("best_practices_issues", "Best Practices", "No best-practice violations found."),
]

# Command


@app.callback(invoke_without_command=True)
def review(
    diff: Path | None = typer.Option(
        None,
        "--diff",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Review a unified diff read from FILE instead of live Git changes.",
    ),
    commit: str | None = typer.Option(
        None,
        "--commit",
        help="Review the changes introduced by a single commit (SHA or ref).",
    ),
    branch: str | None = typer.Option(
        None,
        "--branch",
        help="Review changes on BRANCH relative to its merge base with HEAD.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON instead of a Rich report.",
    ),
    fail_under: int | None = typer.Option(
        None,
        "--fail-under",
        min=0,
        max=100,
        help="Exit with a non-zero status if the AI score falls below this threshold.",
    ),
) -> None:
    """
    Review Git changes before committing.

    With no options, reviews the current working tree (staged + unstaged
    changes). Use --diff, --commit, or --branch to review a specific source
    of changes instead; these are mutually exclusive.
    """
    selected = [
        name for name, val in (("--diff", diff), ("--commit", commit), ("--branch", branch)) if val
    ]
    if len(selected) > 1:
        err_console.print(
            f"[bold red]Error:[/bold red] {', '.join(selected)} are mutually exclusive. "
            "Pick at most one change source."
        )
        raise typer.Exit(code=2)

    try:
        target = _build_target(diff=diff, commit=commit, branch=branch)
    except ReviewTargetError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        result = asyncio.run(_run_review(target, quiet=json_output))
    except ReviewServiceError as exc:
        err_console.print(f"[bold red]Review failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        err_console.print("[yellow]Review cancelled.[/yellow]")
        raise typer.Exit(code=130) from None

    if json_output:
        print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    else:
        _render_report(result)

    if fail_under is not None and result.score < fail_under:
        err_console.print(
            f"\n[bold red]AI score {result.score}/100 is below threshold {fail_under}.[/bold red]"
        )
        raise typer.Exit(code=1)

    if not json_output and result.blocking_findings:
        raise typer.Exit(code=1)


# Helpers


def _build_target(*, diff: Path | None, commit: str | None, branch: str | None) -> ReviewTarget:
    """Translate CLI flags into a ReviewTarget for the service layer."""
    if diff is not None:
        return ReviewTarget.from_diff_file(diff)
    if commit is not None:
        return ReviewTarget.from_commit(commit)
    if branch is not None:
        return ReviewTarget.from_branch(branch)
    return ReviewTarget.working_tree()


async def _run_review(target: ReviewTarget, *, quiet: bool) -> ReviewResult:
    """Run the review via ReviewService, showing a spinner unless quiet."""
    service = ReviewService()

    if quiet:
        return await service.review(target)

    with RichProgress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        transient=True,
    ) as progress:
        progress.add_task(description="Reviewing changes...", total=None)
        return await service.review(target)


def _render_report(result: ReviewResult) -> None:
    """Render a ReviewResult as a Rich, markdown-friendly report."""
    console.print()
    console.print(_render_header(result))
    console.print()

    for attr, title, empty_message in _SECTION_SPECS:
        findings: list[Finding] = getattr(result, attr)
        console.print(_render_section(title, findings, empty_message))
        console.print()

    if result.suggestions:
        console.print(_render_suggestions(result.suggestions))
        console.print()

    if result.summary:
        console.print(Panel(Markdown(result.summary), title="Summary", border_style="blue"))
        console.print()


def _render_header(result: ReviewResult) -> Panel:
    """Render the top-of-report banner with the overall AI score."""
    score = result.score
    score_style = "bold green" if score >= 80 else "bold yellow" if score >= 50 else "bold red"

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_row(
        Text(f"Kodiak Review — {result.target_description}", style="bold"),
        Text(f"{score}/100", style=score_style),
    )

    counts = _count_by_severity(result)
    badges = "  ".join(
        f"{_SEVERITY_STYLE[sev][0]} {counts[sev]} {sev.value.lower()}"
        for sev in _SEVERITY_ORDER
        if counts[sev]
    )
    footer = Text(badges or "No issues found.", style="dim")

    return Panel(
        Group(table, footer),
        title="AI Score",
        border_style=score_style,
        padding=(1, 2),
    )


def _render_section(title: str, findings: list[Finding], empty_message: str) -> Panel:
    """Render one findings section (bugs, security, performance, etc.) as a table."""
    if not findings:
        body: Table | Text = Text(f"✓ {empty_message}", style="green")
        border_style = "green"
    else:
        table = Table(show_header=True, header_style="bold", expand=True, box=None)
        table.add_column("", width=2, no_wrap=True)
        table.add_column("Location", style="cyan", no_wrap=True)
        table.add_column("Description", ratio=1)

        for finding in sorted(findings, key=lambda f: _SEVERITY_ORDER.index(f.severity)):
            icon, style = _SEVERITY_STYLE[finding.severity]
            location = finding.location or "—"
            description = finding.message
            if finding.suggestion:
                description += f"\n[dim]→ {finding.suggestion}[/dim]"
            table.add_row(Text(icon, style=style), location, description)
        body = table
        border_style = (
            "red"
            if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings)
            else "yellow"
        )

    return Panel(
        body, title=f"{title} ({len(findings)})", border_style=border_style, padding=(1, 2)
    )


def _render_suggestions(suggestions: list[str]) -> Panel:
    """Render suggested improvements as a bulleted markdown list."""
    md = "\n".join(f"- {s}" for s in suggestions)
    return Panel(
        Markdown(md), title=f"Suggested Improvements ({len(suggestions)})", border_style="magenta"
    )


def _count_by_severity(result: ReviewResult) -> dict[Severity, int]:
    counts = {sev: 0 for sev in _SEVERITY_ORDER}
    for attr, _title, _empty in _SECTION_SPECS:
        for finding in getattr(result, attr):
            counts[finding.severity] += 1
    return counts


if __name__ == "__main__":  # pragma: no cover
    app()
