"""
kodiak memory
=============

CLI entrypoint for managing Kodiak's long-term memory store.

This module is presentation-only: it parses arguments, delegates every
read/write operation to :class:`kodiak.memory.service.MemoryService`, and
renders the results either as Rich tables/panels or as JSON. No storage,
embedding, ranking, or retention logic lives here -- that belongs entirely
to the memory service and its underlying backends.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kodiak.memory.errors import MemoryNotFoundError, MemoryServiceError
from kodiak.memory.models import Memory, SearchResult
from kodiak.memory.service import MemoryService

logger = structlog.get_logger(__name__)
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="memory",
    help="Manage Kodiak's long-term memory.",
    no_args_is_help=True,
    add_completion=False,
)

_TAG_OPTION = typer.Option(
    None,
    "--tag",
    "-t",
    help="Filter/tag by label. Repeatable (e.g. --tag infra --tag postgres).",
)
_LIMIT_OPTION = typer.Option(
    20,
    "--limit",
    "-n",
    min=1,
    max=1000,
    help="Maximum number of memories to return.",
)
_JSON_OPTION = typer.Option(
    False,
    "--json",
    help="Emit machine-readable JSON instead of a Rich table.",
)


# --------------------------------------------------------------------------- #
# kodiak memory add
# --------------------------------------------------------------------------- #


@app.command("add")
def add(
    content: str = typer.Argument(..., help="The text content to remember."),
    tag: Optional[list[str]] = _TAG_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Add a new memory."""
    try:
        memory = asyncio.run(_add(content=content, tags=tag))
    except MemoryServiceError as exc:
        _fail(f"Could not add memory: {exc}")

    if json_output:
        _print_json(memory.model_dump(mode="json"))
        return

    console.print(
        Panel(
            _memory_body(memory),
            title="[bold green]Memory added[/bold green]",
            border_style="green",
        )
    )


async def _add(*, content: str, tags: Optional[list[str]]) -> Memory:
    service = MemoryService()
    return await service.add(content=content, tags=tags or [])


# --------------------------------------------------------------------------- #
# kodiak memory search
# --------------------------------------------------------------------------- #


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Natural-language query to search memories for."),
    tag: Optional[list[str]] = _TAG_OPTION,
    limit: int = _LIMIT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Search memories by semantic similarity, optionally scoped to tags."""
    try:
        results = asyncio.run(_search(query=query, tags=tag, limit=limit))
    except MemoryServiceError as exc:
        _fail(f"Search failed: {exc}")

    if json_output:
        _print_json([r.model_dump(mode="json") for r in results])
        return

    if not results:
        console.print(f"[dim]No memories found for query:[/dim] [italic]{query}[/italic]")
        return

    table = Table(
        title=f'Search results for "{query}"',
        show_header=True,
        header_style="bold",
        expand=True,
    )
    table.add_column("ID", style="dim", no_wrap=True, max_width=8)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Content", ratio=1)
    table.add_column("Tags", no_wrap=True)
    table.add_column("Created", style="dim", no_wrap=True)

    for result in results:
        table.add_row(
            _short_id(result.memory.id),
            _score_text(result.score),
            _truncate(result.memory.content),
            _tags_text(result.memory.tags),
            _format_dt(result.memory.created_at),
        )

    console.print(table)


async def _search(*, query: str, tags: Optional[list[str]], limit: int) -> list[SearchResult]:
    service = MemoryService()
    return await service.search(query=query, tags=tags or [], limit=limit)


# --------------------------------------------------------------------------- #
# kodiak memory list
# --------------------------------------------------------------------------- #


@app.command("list")
def list_memories(
    tag: Optional[list[str]] = _TAG_OPTION,
    limit: int = _LIMIT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """List stored memories, most recent first, optionally filtered by tag."""
    try:
        memories = asyncio.run(_list(tags=tag, limit=limit))
    except MemoryServiceError as exc:
        _fail(f"Could not list memories: {exc}")

    if json_output:
        _print_json([m.model_dump(mode="json") for m in memories])
        return

    if not memories:
        console.print("[dim]No memories stored yet.[/dim]")
        return

    tag_suffix = f" (tags: {', '.join(tag)})" if tag else ""
    table = Table(
        title=f"Memories{tag_suffix}",
        show_header=True,
        header_style="bold",
        expand=True,
    )
    table.add_column("ID", style="dim", no_wrap=True, max_width=8)
    table.add_column("Content", ratio=1)
    table.add_column("Tags", no_wrap=True)
    table.add_column("Created", style="dim", no_wrap=True)
    table.add_column("Updated", style="dim", no_wrap=True)

    for memory in memories:
        table.add_row(
            _short_id(memory.id),
            _truncate(memory.content),
            _tags_text(memory.tags),
            _format_dt(memory.created_at),
            _format_dt(memory.updated_at),
        )

    console.print(table)
    console.print(f"[dim]{len(memories)} memor{'y' if len(memories) == 1 else 'ies'} shown.[/dim]")


async def _list(*, tags: Optional[list[str]], limit: int) -> list[Memory]:
    service = MemoryService()
    return await service.list(tags=tags or [], limit=limit)


# --------------------------------------------------------------------------- #
# kodiak memory delete
# --------------------------------------------------------------------------- #


@app.command("delete")
def delete(
    memory_id: Optional[str] = typer.Argument(
        None, help="ID of the memory to delete. Omit when using --tag to bulk-delete."
    ),
    tag: Optional[list[str]] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Delete all memories matching this tag instead of a single ID. Repeatable.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Delete a memory by ID, or bulk-delete by tag."""
    if not memory_id and not tag:
        _fail("Provide a memory ID or at least one --tag to delete.")
    if memory_id and tag:
        _fail("Provide either a memory ID or --tag, not both.")

    if not yes:
        target = f"memory {memory_id}" if memory_id else f"all memories tagged {', '.join(tag)}"
        confirmed = typer.confirm(f"Delete {target}? This cannot be undone.")
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=0)

    try:
        if memory_id:
            asyncio.run(_delete_one(memory_id))
            deleted_count = 1
        else:
            deleted_count = asyncio.run(_delete_by_tag(tag))
    except MemoryNotFoundError as exc:
        _fail(f"Not found: {exc}")
    except MemoryServiceError as exc:
        _fail(f"Delete failed: {exc}")

    if json_output:
        _print_json({"deleted": deleted_count})
        return

    console.print(f"[bold green]Deleted {deleted_count} memor{'y' if deleted_count == 1 else 'ies'}.[/bold green]")


async def _delete_one(memory_id: str) -> None:
    service = MemoryService()
    await service.delete(memory_id)


async def _delete_by_tag(tags: list[str]) -> int:
    service = MemoryService()
    return await service.delete_by_tags(tags)


# --------------------------------------------------------------------------- #
# Shared rendering helpers
# --------------------------------------------------------------------------- #


def _memory_body(memory: Memory) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", no_wrap=True)
    grid.add_column()
    grid.add_row("ID", _short_id(memory.id))
    grid.add_row("Content", memory.content)
    grid.add_row("Tags", _tags_text(memory.tags))
    grid.add_row("Created", _format_dt(memory.created_at))
    return grid


def _tags_text(tags: list[str]) -> Text:
    if not tags:
        return Text("—", style="dim")
    return Text(" ".join(f"#{t}" for t in tags), style="cyan")


def _score_text(score: float) -> Text:
    style = "bold green" if score >= 0.8 else "yellow" if score >= 0.5 else "dim"
    return Text(f"{score:.2f}", style=style)


def _truncate(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _short_id(memory_id: str, width: int = 8) -> str:
    return memory_id[:width]


def _format_dt(value: Optional[datetime]) -> str:
    if value is None:
        return "—"
    return value.strftime("%Y-%m-%d %H:%M")


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _fail(message: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()