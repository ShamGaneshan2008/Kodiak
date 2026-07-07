"""CLI command for generating an implementation plan.

This module defines the ``kodiak plan`` command. It is strictly a
presentation-layer component: it performs no AI reasoning itself. All
business logic is delegated to
:class:`kodiak.services.plan_service.PlanService`.

Typical usage example:

    $ kodiak plan --issue 123
    $ kodiak plan --task-id abc123
    $ kodiak plan --issue 123 --json
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from kodiak.services.plan_service import (
    PlanGenerationError,
    PlanService,
    PlanServiceError,
)

app = typer.Typer(name="plan", help="Generate an implementation plan for an issue or task.")

console = Console()

_COMPLEXITY_STYLES = {
    "low": "green",
    "medium": "yellow",
    "high": "bold red",
}


def _render_error_panel(title: str, message: str) -> None:
    """Render a friendly error panel.

    Args:
        title: Short title describing the failure category.
        message: Human-readable explanation of what went wrong.
    """
    console.print(
        Panel.fit(
            f"[bold red]{message}[/bold red]",
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
        )
    )


def _render_plan(plan: "Plan") -> None:  # noqa: F821
    """Render a generated plan using Rich panels and trees.

    Args:
        plan: The plan data returned by :class:`PlanService`.
    """
    console.print(
        Panel.fit(
            f"[bold]{plan.goal}[/bold]",
            title="[bold cyan]Goal[/bold cyan]",
            border_style="cyan",
        )
    )

    files_tree = Tree("[bold cyan]Files to Modify[/bold cyan]")
    if plan.files_to_modify:
        for file_path in plan.files_to_modify:
            files_tree.add(file_path)
    else:
        files_tree.add("[dim]None[/dim]")
    console.print(files_tree)

    new_files_tree = Tree("[bold cyan]New Files[/bold cyan]")
    if plan.new_files:
        for file_path in plan.new_files:
            new_files_tree.add(file_path)
    else:
        new_files_tree.add("[dim]None[/dim]")
    console.print(new_files_tree)

    risks_tree = Tree("[bold yellow]Risks[/bold yellow]")
    if plan.risks:
        for risk in plan.risks:
            risks_tree.add(risk)
    else:
        risks_tree.add("[dim]None identified[/dim]")
    console.print(risks_tree)

    complexity_style = _COMPLEXITY_STYLES.get(plan.complexity.lower(), "white")
    console.print(
        Panel.fit(
            f"Complexity: [{complexity_style}]{plan.complexity}[/{complexity_style}]\n"
            f"Estimated time: [bold]{plan.estimated_time}[/bold]",
            title="[bold cyan]Estimate[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print(
        Panel(
            plan.reasoning,
            title="[bold cyan]AI Reasoning[/bold cyan]",
            border_style="cyan",
        )
    )


@app.command()
def plan(
    issue: Optional[str] = typer.Option(
        None, "--issue", help="GitHub issue number or URL to plan against."
    ),
    task_id: Optional[str] = typer.Option(
        None, "--task-id", help="ID of an existing Kodiak task to plan against."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output the plan as raw JSON instead of Rich display."
    ),
) -> None:
    """Generate an implementation plan for a GitHub issue or task.

    Exactly one of ``--issue`` or ``--task-id`` must be provided. All
    plan generation is delegated to :class:`PlanService`.

    Args:
        issue: GitHub issue number or URL to plan against.
        task_id: ID of an existing Kodiak task to plan against.
        json_output: If ``True``, prints the plan as raw JSON.

    Raises:
        typer.Exit: With code ``0`` on success, ``1`` on invalid input or
            planning failure, or ``2`` on any unexpected error.
    """
    if not issue and not task_id:
        _render_error_panel(
            "Missing Input", "You must provide either --issue or --task-id."
        )
        raise typer.Exit(code=1)

    if issue and task_id:
        _render_error_panel(
            "Conflicting Input", "Provide only one of --issue or --task-id, not both."
        )
        raise typer.Exit(code=1)

    plan_service = PlanService()

    try:
        with console.status(
            "[bold cyan]Generating implementation plan...[/bold cyan]", spinner="dots"
        ):
            if issue:
                generated_plan = plan_service.generate_plan_from_issue(issue)
            else:
                generated_plan = plan_service.generate_plan_from_task(task_id)

    except PlanGenerationError as exc:
        _render_error_panel("Plan Generation Failed", str(exc))
        raise typer.Exit(code=1) from exc

    except PlanServiceError as exc:
        _render_error_panel("Plan Service Error", str(exc))
        raise typer.Exit(code=1) from exc

    except Exception as exc:  # noqa: BLE001 - surfaced deliberately as a CLI panel
        _render_error_panel("Unexpected Error", f"An unexpected error occurred.\n{exc}")
        raise typer.Exit(code=2) from exc

    if json_output:
        console.print_json(json.dumps(generated_plan.to_dict()))
    else:
        _render_plan(generated_plan)

    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()