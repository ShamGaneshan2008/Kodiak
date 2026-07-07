"""CLI command for explaining code using Kodiak AI.

This module contains presentation logic only. All language-model work,
including streaming, is delegated to
:class:`kodiak.services.explain_service.ExplainService`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from kodiak.schemas.explain import (
    ExplainChunk,
    ExplainMetadata,
    ExplainRequest,
    ExplainResult,
    ExplainTargetKind,
)
from kodiak.services.exceptions import ExplainError
from kodiak.services.explain_service import ExplainService

console = Console()
error_console = Console(stderr=True)

app = typer.Typer()


@app.command("explain")
def explain(
    target: Optional[Path] = typer.Argument(
        None,
        help="Path to a file to explain.",
    ),
    function: Optional[str] = typer.Option(
        None,
        "--function",
        "-f",
        help="Name of a function to explain.",
    ),
    class_name: Optional[str] = typer.Option(
        None,
        "--class",
        "-c",
        help="Name of a class to explain.",
    ),
    architecture: bool = typer.Option(
        False,
        "--architecture",
        "-a",
        help="Explain the overall project architecture.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable JSON result instead of streaming text.",
    ),
    markdown_output: bool = typer.Option(
        False,
        "--markdown",
        help="Emit raw markdown text as it streams, suitable for piping.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional metadata (model, tokens, sources) after the explanation.",
    ),
) -> None:
    """Explain code, functions, classes, or overall architecture.

    Exactly one target must be specified: a file path, ``--function``,
    ``--class``, or ``--architecture``. By default the explanation is
    streamed and rendered as formatted Markdown. Use ``--markdown`` to
    stream raw markdown text instead, or ``--json`` to receive a single
    structured result once generation completes.

    Examples:
        kodiak explain app.py
        kodiak explain auth.py
        kodiak explain --function login
        kodiak explain --class User
        kodiak explain --architecture
        kodiak explain app.py --json
        kodiak explain app.py --markdown
    """
    if json_output and markdown_output:
        _render_error("--json and --markdown cannot be used together.")
        raise typer.Exit(code=1)

    try:
        request = _build_request(target, function, class_name, architecture, verbose)
    except ValueError as exc:
        _render_error(str(exc))
        raise typer.Exit(code=1)

    service = ExplainService()

    try:
        if json_output:
            result = _run(service.explain(request))
            console.print_json(result.model_dump_json())
        elif markdown_output:
            _run(_stream_raw_markdown(service, request))
        else:
            content, metadata = _run(_stream_rendered(service, request))
            if verbose and metadata is not None:
                _render_metadata(metadata)
    except ExplainError as exc:
        _render_error(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        _render_error(f"Unexpected error while generating explanation: {exc}")
        raise typer.Exit(code=2)

    raise typer.Exit(code=0)


def _build_request(
    target: Optional[Path],
    function: Optional[str],
    class_name: Optional[str],
    architecture: bool,
    verbose: bool,
) -> ExplainRequest:
    """Build a validated ExplainRequest from CLI arguments.

    Exactly one of file, function, class, or architecture must be given.

    Args:
        target: File path, if provided.
        function: Function name, if provided.
        class_name: Class name, if provided.
        architecture: Whether the architecture flag was set.
        verbose: Whether verbose metadata was requested.

    Returns:
        A fully populated ExplainRequest.

    Raises:
        ValueError: If zero or more than one target was specified, or if
            a given file path does not exist.
    """
    provided = [
        value
        for value in (target is not None, function is not None, class_name is not None, architecture)
        if value
    ]
    if len(provided) == 0:
        raise ValueError(
            "No target specified. Provide a file path, --function, --class, "
            "or --architecture."
        )
    if len(provided) > 1:
        raise ValueError(
            "Multiple targets specified. Provide only one of: file path, "
            "--function, --class, --architecture."
        )

    if target is not None:
        resolved = target.resolve()
        if not resolved.exists():
            raise ValueError(f"File does not exist: {resolved}")
        return ExplainRequest(
            kind=ExplainTargetKind.FILE, target=str(resolved), verbose=verbose
        )
    if function is not None:
        return ExplainRequest(
            kind=ExplainTargetKind.FUNCTION, target=function, verbose=verbose
        )
    if class_name is not None:
        return ExplainRequest(
            kind=ExplainTargetKind.CLASS, target=class_name, verbose=verbose
        )
    return ExplainRequest(
        kind=ExplainTargetKind.ARCHITECTURE, target=None, verbose=verbose
    )


async def _stream_rendered(
    service: ExplainService, request: ExplainRequest
) -> tuple[str, Optional[ExplainMetadata]]:
    """Stream an explanation and render it as live-updating Markdown.

    Args:
        service: The service to delegate streaming to.
        request: The validated explanation request.

    Returns:
        A tuple of the full accumulated markdown text and, if present,
        the metadata reported by the final chunk.
    """
    buffer = ""
    metadata: Optional[ExplainMetadata] = None
    status = console.status("[bold cyan]Analyzing...", spinner="dots")
    status.start()

    live: Optional[Live] = None
    try:
        async for chunk in service.stream(request):
            if chunk.metadata is not None:
                metadata = chunk.metadata
                continue
            if live is None:
                status.stop()
                live = Live(console=console, refresh_per_second=8)
                live.start()
            buffer += chunk.text or ""
            live.update(Markdown(buffer))
    finally:
        status.stop()
        if live is not None:
            live.stop()

    return buffer, metadata


async def _stream_raw_markdown(service: ExplainService, request: ExplainRequest) -> None:
    """Stream an explanation as raw markdown text, suitable for piping.

    Args:
        service: The service to delegate streaming to.
        request: The validated explanation request.
    """
    async for chunk in service.stream(request):
        if chunk.text:
            console.file.write(chunk.text)
    console.file.write("\n")


_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine to completion from a sync Typer command.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's result.
    """
    return asyncio.run(coro)


def _render_metadata(metadata: ExplainMetadata) -> None:
    """Render verbose explanation metadata in a Rich table.

    Args:
        metadata: Metadata reported alongside the explanation, such as
            model, token usage, and source references.
    """
    table = Table(title="Explanation Details", show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Model", metadata.model)
    table.add_row("Tokens used", f"{metadata.tokens_used:,}")
    if metadata.sources:
        table.add_row("Sources", ", ".join(metadata.sources))
    console.print(table)


def _render_error(message: str) -> None:
    """Render an error message inside a red Rich panel.

    Args:
        message: Human-readable error description.
    """
    error_console.print(
        Panel(
            message,
            title="Explain Failed",
            border_style="red",
            title_align="left",
        )
    )