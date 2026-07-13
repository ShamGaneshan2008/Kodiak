from __future__ import annotations

import structlog

from kodiak.agents.base import AgentInput
from kodiak.agents.planning_agent import PlannerAgent, SubTask, TaskPlan

logger = structlog.get_logger(__name__)


class InvalidIssueError(Exception):
    """Raised when the provided AgentInput is missing required information."""


class PlanGenerationFailedError(Exception):
    """Raised when the PlannerAgent fails to produce an implementation plan."""


class PlannerService:
    """Coordinates implementation planning by delegating to the PlannerAgent.

    This service is a thin orchestration layer. It validates the incoming
    AgentInput, invokes the planning agent via its standard BaseAgent.run()
    contract, and returns the resulting TaskPlan. It performs no planning
    logic itself.

    Attributes:
        _planning_agent: The agent responsible for generating implementation plans.
    """

    def __init__(self, planning_agent: PlannerAgent) -> None:
        """Initializes the PlannerService.

        Args:
            planning_agent: An instance of PlannerAgent used to generate
                implementation plans.
        """
        self._planning_agent = planning_agent

    async def create_plan(self, input_: AgentInput) -> TaskPlan:
        """Validates an AgentInput and coordinates generation of its implementation plan.

        Args:
            input_: The AgentInput describing the planning task.

        Returns:
            A TaskPlan produced by the PlannerAgent.

        Raises:
            InvalidIssueError: If the input is missing required information.
            PlanGenerationFailedError: If the planning agent fails to
                complete plan generation.
        """
        self._validate_input(input_)

        logger.info(
            "planner_service.plan_generation_started",
            task_id=input_.task_id,
            project_id=input_.project_id,
        )

        output = await self._planning_agent.run(input_)

        if not output.success:
            logger.error(
                "planner_service.plan_generation_failed",
                task_id=input_.task_id,
                project_id=input_.project_id,
                error=output.error,
            )
            raise PlanGenerationFailedError(
                f"Plan generation failed for task {input_.task_id}: {output.error}"
            )

        plan_dict = output.result.get("plan")
        if plan_dict is None:
            logger.error(
                "planner_service.plan_generation_failed",
                task_id=input_.task_id,
                project_id=input_.project_id,
                error="Agent output did not contain a plan.",
            )
            raise PlanGenerationFailedError(
                f"Plan generation failed for task {input_.task_id}: "
                "agent output did not contain a plan."
            )

        plan = self._dict_to_plan(plan_dict)

        logger.info(
            "planner_service.plan_generation_completed",
            task_id=input_.task_id,
            project_id=input_.project_id,
            step_count=len(plan.subtasks),
        )

        return plan

    def _validate_input(self, input_: AgentInput) -> None:
        """Validates that the AgentInput contains the information required for planning.

        Args:
            input_: The input to validate.

        Raises:
            InvalidIssueError: If the task id, project id, or instruction is missing.
        """
        if not input_.task_id:
            raise InvalidIssueError("AgentInput must have a valid task_id.")

        if not input_.project_id:
            raise InvalidIssueError("AgentInput must specify a project_id.")

        if not input_.instruction or not input_.instruction.strip():
            raise InvalidIssueError("AgentInput must have a non-empty instruction.")

    def _dict_to_plan(self, plan_dict: dict) -> TaskPlan:
        """Reconstructs a TaskPlan from the dict produced by PlannerAgent._plan_to_dict.

        Args:
            plan_dict: The plan dictionary from AgentOutput.result["plan"].

        Returns:
            The equivalent TaskPlan dataclass instance.
        """
        subtasks = [
            SubTask(
                id=st["id"],
                title=st["title"],
                description=st["description"],
                type=st["type"],
                complexity=st["complexity"],
                depends_on=st["depends_on"],
                likely_files=st["likely_files"],
            )
            for st in plan_dict.get("subtasks", [])
        ]
        return TaskPlan(
            goal=plan_dict.get("goal", ""),
            acceptance_criteria=plan_dict.get("acceptance_criteria", []),
            subtasks=subtasks,
            estimated_total_complexity=plan_dict.get("estimated_total_complexity", "medium"),
            requires_architecture_review=plan_dict.get("requires_architecture_review", False),
        )