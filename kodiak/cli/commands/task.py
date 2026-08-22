"""CLI commands for Kodiak autonomous task workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from kodiak.agents.manager import AgentManager
from kodiak.memory.service import MemoryService
from kodiak.orchestration.autonomous_loop import AutonomousTaskLoop, initialize_autonomous_loop
from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.task_planner import TaskPlanner

app = typer.Typer(name="task", help="Kodiak autonomous task workflows.")

console = Console()


class _StubAgent:
    """Minimal agent used for CLI execution when no custom agents are configured."""

    def __init__(self, name: str, capabilities: set[str]) -> None:
        self.agent_id = name
        self.name = name
        self.capabilities = frozenset(capabilities)

    async def execute(self, task: Any) -> dict[str, Any]:
        return {
            "agent": self.name,
            "summary": f"Completed {getattr(task, 'task_type', 'task')} via {self.name}",
            "verification_status": "verified",
        }

    async def health_check(self) -> bool:
        return True


async def _build_default_loop(max_attempts: int) -> AutonomousTaskLoop:
    manager = AgentManager()
    for agent in (
        _StubAgent("coder", {"code_generation", "file_editing"}),
        _StubAgent("tester", {"test_execution"}),
        _StubAgent("reviewer", {"code_review"}),
        _StubAgent("research", {"research", "context_retrieval"}),
    ):
        await manager.register(agent)

    return AutonomousTaskLoop(
        task_planner=TaskPlanner(),
        memory_service=MemoryService(),
        agent_manager=manager,
        execution_engine=ExecutionEngine(manager, default_timeout_seconds=60.0),
        max_loop_attempts=max_attempts,
    )


@app.command("run")
def run(
    task: str = typer.Argument(..., help="Engineering task description."),
    workspace: Path = typer.Option(
        Path.cwd(), "--workspace", "-w", help="Repository workspace path."
    ),
    max_attempts: int = typer.Option(
        3, "--max-attempts", min=1, help="Maximum autonomous loop attempts."
    ),
) -> None:
    """Run an engineering task through the autonomous orchestration loop."""

    async def _execute() -> None:
        loop = await _build_default_loop(max_attempts)
        await initialize_autonomous_loop(loop)
        result = await loop.run(task, workspace=workspace)

        status_style = "green" if result.success else "red"
        console.print(
            Panel.fit(
                f"[bold]Status:[/bold] [{status_style}]"
                f"{result.task_state.status.value}[/{status_style}]\n"
                f"[bold]Attempts:[/bold] {result.attempts}\n"
                f"[bold]Agent:[/bold] {result.selected_agent or 'n/a'}\n"
                f"[bold]Elapsed:[/bold] {result.elapsed_seconds:.2f}s\n"
                f"[bold]Memory stored:[/bold] {result.memory_stored}",
                title="[bold cyan]Autonomous Task Result[/bold cyan]",
                border_style="cyan",
            )
        )
        if result.task_state.result:
            console.print(f"\n[bold]Result:[/bold] {result.task_state.result}")
        if result.task_state.error:
            console.print(f"\n[bold red]Error:[/bold red] {result.task_state.error}")
        if not result.success:
            raise typer.Exit(code=1)

    try:
        asyncio.run(_execute())
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(
            Panel.fit(
                f"[bold red]{exc}[/bold red]",
                title="[bold red]Task Execution Failed[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
