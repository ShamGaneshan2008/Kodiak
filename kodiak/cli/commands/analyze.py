"""CLI command for analyzing a Kodiak-managed repository.

This module contains presentation logic only. All analysis work is
delegated to :class:`kodiak.services.analyze_service.AnalyzeService`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


from kodiak.agents.repository import (
    RepositoryAnalyzerAgent,
    RepositoryAnalysis,
)
from kodiak.cli.services.analyze_service import (
    AnalyzeService,
    InvalidRepositoryPathError,
    RepositoryAnalysisFailedError,
)

console = Console()
error_console = Console(stderr=True)

app = typer.Typer()


@app.command("analyze")
def analyze(
    path: Path = typer.Argument(
        Path("."),
        help="Path to the repository to analyze.",
    ),
    deep: bool = typer.Option(
        False,
        "--deep",
        "-d",
        help="Perform a deep analysis, including dependency resolution "
        "and token estimation.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON instead of formatted tables.",
    ),
) -> None:
    """Analyze a repository and report structural statistics.

    Inspects the target repository and reports total file count, detected
    languages, dependencies, project size, current git branch, and RAG
    indexing statistics such as indexed file count and estimated token
    usage.

    Examples:
        kodiak analyze
        kodiak analyze .
        kodiak analyze ./repo
        kodiak analyze --deep
        kodiak analyze --json
    """
    resolved_path = path.resolve()

    if not resolved_path.exists():
        _render_error(f"Path does not exist: {resolved_path}")
        raise typer.Exit(code=1)

    if not resolved_path.is_dir():
        _render_error(f"Path is not a directory: {resolved_path}")
        raise typer.Exit(code=1)

    service = AnalyzeService()

    try:
        result = asyncio.run(_run_analysis(service, resolved_path, deep))
    except RepositoryAnalysis as exc:
        _render_error(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Unexpected error during analysis: {exc}")
        raise typer.Exit(code=2)

    if json_output:
        console.print_json(result.model_dump_json())
    else:
        _render_result(result, resolved_path)

    raise typer.Exit(code=0)


async def _run_analysis(
    service: AnalyzeService, path: Path, deep: bool
) -> RepositoryAnalysis:
    """Run the analysis while displaying a Rich spinner.

    Args:
        service: The analysis service to delegate business logic to.
        path: Absolute path to the repository being analyzed.
        deep: Whether to request a deep analysis pass.

    Returns:
        The structured result of the analysis.
    """
    description = (
        "Running deep analysis..." if deep else "Analyzing repository..."
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        return await service.analyze(path, deep=deep)


def _render_result(result: RepositoryAnalysis, path: Path) -> None:
    """Render an analysis result as Rich tables.

    Args:
        result: The analysis result returned by AnalyzeService.
        path: The repository path that was analyzed.
    """
    console.print(
        Panel(
            f"[bold]{path}[/bold]",
            title="Kodiak Repository Analysis",
            border_style="cyan",
        )
    )

    overview = Table(title="Overview", show_header=False, box=None, padding=(0, 2))
    overview.add_column("Metric", style="bold")
    overview.add_column("Value")
    overview.add_row("Total files", str(result.total_files))
    overview.add_row("Project size", result.project_size_human)
    overview.add_row(
        "Git branch", result.git_branch or "[dim]not a git repo[/dim]"
    )
    overview.add_row("Indexed files", str(result.indexed_files))
    overview.add_row("Estimated tokens", f"{result.estimated_tokens:,}")
    console.print(overview)

    if result.languages:
        lang_table = Table(title="Languages")
        lang_table.add_column("Language", style="bold")
        lang_table.add_column("Files", justify="right")
        lang_table.add_column("Percentage", justify="right")
        for lang in result.languages:
            lang_table.add_row(
                lang.name, str(lang.file_count), f"{lang.percentage:.1f}%"
            )
        console.print(lang_table)
    else:
        console.print("[dim]No languages detected.[/dim]")

    if result.dependencies:
        dep_table = Table(title="Dependencies")
        dep_table.add_column("Name", style="bold")
        dep_table.add_column("Version")
        dep_table.add_column("Source")
        for dep in result.dependencies:
            dep_table.add_row(
                dep.name, dep.version or "[dim]unknown[/dim]", dep.source
            )
        console.print(dep_table)
    else:
        console.print("[dim]No dependencies detected.[/dim]")


def _render_error(message: str) -> None:
    """Render an error message inside a red Rich panel.

    Args:
        message: Human-readable error description.
    """
    error_console.print(
        Panel(
            message,
            title="Analysis Failed",
            border_style="red",
            title_align="left",
        )
    )