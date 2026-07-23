from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kodiak.agents.repository import RepositoryAnalysis
from kodiak.cli.services.analyze_service import (
    AnalyzeService,
    InvalidRepositoryPathError,
    RepositoryAnalysisFailedError,
)

app = typer.Typer(help="Analyze a repository.")
console = Console()
error_console = Console(stderr=True)


@app.command()
def analyze(
    path: Path = typer.Argument(Path("."), help="Repository path"),
    deep: bool = typer.Option(False, "--deep", "-d"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    service = AnalyzeService()

    try:
        with console.status("[bold cyan]Analyzing repository..."):
            result = asyncio.run(service.analyze_repository(path, deep=deep))
    except InvalidRepositoryPathError as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except RepositoryAnalysisFailedError as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if json_output:
        payload = {
            "root_path": str(result.root_path),
            "file_count": result.file_count,
            "directory_count": result.directory_count,
            "total_size_bytes": result.total_size_bytes,
            "extension_counts": result.extension_counts,
            "language_stats": result.language_stats,
        }
        console.print_json(json.dumps(payload))
        return

    _render(result)


def _render(result: RepositoryAnalysis) -> None:
    console.print(
        Panel(
            f"[bold]{result.root_path}[/bold]",
            title="Repository Analysis",
            border_style="cyan",
        )
    )

    overview = Table(title="Overview")
    overview.add_column("Metric")
    overview.add_column("Value")

    overview.add_row("Files", str(result.file_count))
    overview.add_row("Directories", str(result.directory_count))
    overview.add_row("Size (bytes)", str(result.total_size_bytes))

    console.print(overview)

    if result.language_stats:
        langs = Table(title="Languages")
        langs.add_column("Language")
        langs.add_column("Files", justify="right")
        for name, count in sorted(result.language_stats.items()):
            langs.add_row(name, str(count))
        console.print(langs)

    if result.extension_counts:
        ext = Table(title="Extensions")
        ext.add_column("Extension")
        ext.add_column("Files", justify="right")
        for name, count in sorted(result.extension_counts.items()):
            ext.add_row(name, str(count))
        console.print(ext)
