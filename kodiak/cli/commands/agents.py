"""List the concrete agent implementations shipped in the repository."""

from __future__ import annotations

import json
from dataclasses import asdict

import typer
from rich.console import Console
from rich.panel import Panel

from kodiak.orchestration.local_agents import AgentDescriptor, local_agent_descriptors

app = typer.Typer(help="Inspect Kodiak's available agent implementations.", no_args_is_help=True)
console = Console()


def available_agents() -> list[AgentDescriptor]:
    """Return only descriptors whose implementation module is importable."""
    return local_agent_descriptors()


@app.command("list")
def list_agents(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List implemented agents, their roles, status, and execution mode."""
    agents = available_agents()
    if json_output:
        print(json.dumps([asdict(agent) for agent in agents], indent=2))
        return
    console.print("[bold]Kodiak agents[/bold]")
    for agent in agents:
        console.print(
            Panel(
                f"Role: {agent.role}\n"
                f"Status: {agent.status}\n"
                f"Backend: {agent.backend}\n"
                f"Description: {agent.description}",
                title=agent.name,
            )
        )
