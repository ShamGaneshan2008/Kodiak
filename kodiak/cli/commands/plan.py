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

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from kodiak.agents.base import AgentInput
from kodiak.agents.planner import PlannerAgent, TaskPlan
from kodiak.cli.services.planner_service import (
    InvalidIssueError,
    PlanGenerationFailedError,
    PlannerService,
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


def _render_plan(plan: TaskPlan) -> None:
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

    likely_files = list(
        dict.fromkeys(file_path for subtask in plan.subtasks for file_path in subtask.likely_files)
    )
    files_tree = Tree("[bold cyan]Likely Files[/bold cyan]")
    if likely_files:
        for file_path in likely_files:
            files_tree.add(file_path)
    else:
        files_tree.add("[dim]None[/dim]")
    console.print(files_tree)

    subtasks_tree = Tree("[bold cyan]Subtasks[/bold cyan]")
    if plan.subtasks:
        for subtask in plan.subtasks:
            subtasks_tree.add(f"[bold]{subtask.title}[/bold]: {subtask.description}")
    else:
        subtasks_tree.add("[dim]None[/dim]")
    console.print(subtasks_tree)

    criteria_tree = Tree("[bold cyan]Acceptance Criteria[/bold cyan]")
    if plan.acceptance_criteria:
        for criterion in plan.acceptance_criteria:
            criteria_tree.add(criterion)
    else:
        criteria_tree.add("[dim]None specified[/dim]")
    console.print(criteria_tree)

    complexity = plan.estimated_total_complexity
    complexity_style = _COMPLEXITY_STYLES.get(complexity.lower(), "white")
    console.print(
        Panel.fit(
            f"Complexity: [{complexity_style}]{complexity}[/{complexity_style}]\n"
            f"Architecture review: "
            f"[bold]{'required' if plan.requires_architecture_review else 'not required'}[/bold]",
            title="[bold cyan]Estimate[/bold cyan]",
            border_style="cyan",
        )
    )


async def _generate_plan(issue: str | None, task_id: str | None) -> TaskPlan:
    from kodiak.llm.client import get_llm_client

    reference = issue if issue is not None else task_id
    if reference is None:
        raise InvalidIssueError("A planning reference is required.")

    plan_task_id = task_id or f"issue-{issue}"
    instruction = (
        f"Plan the implementation for GitHub issue {issue}"
        if issue is not None
        else f"Plan the implementation for Kodiak task {task_id}"
    )
    service = PlannerService(PlannerAgent(get_llm_client(plan_task_id)))
    return await service.create_plan(
        AgentInput(
            task_id=plan_task_id,
            project_id="local",
            instruction=instruction,
            context={"issue": issue, "task_id": task_id},
        )
    )


@app.command()
def plan(
    issue: str | None = typer.Option(
        None, "--issue", help="GitHub issue number or URL to plan against."
    ),
    task_id: str | None = typer.Option(
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
        _render_error_panel("Missing Input", "You must provide either --issue or --task-id.")
        raise typer.Exit(code=1)

    if issue and task_id:
        _render_error_panel(
            "Conflicting Input", "Provide only one of --issue or --task-id, not both."
        )
        raise typer.Exit(code=1)

    try:
        with console.status(
            "[bold cyan]Generating implementation plan...[/bold cyan]", spinner="dots"
        ):
            generated_plan = asyncio.run(_generate_plan(issue, task_id))

    except PlanGenerationFailedError as exc:
        _render_error_panel("Plan Generation Failed", str(exc))
        raise typer.Exit(code=1) from exc

    except InvalidIssueError as exc:
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
