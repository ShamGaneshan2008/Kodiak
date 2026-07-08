from __future__ import annotations

import structlog

from kodiak.agents.planning_agent import PlanningAgent
from kodiak.agents.models.issue import GitHubIssue
from kodiak.agents.models.plan import ImplementationPlan

logger = structlog.get_logger(__name__)


class InvalidIssueError(Exception):
    """Raised when the provided GitHub issue is missing required information."""


class PlanGenerationFailedError(Exception):
    """Raised when the PlanningAgent fails to produce an implementation plan."""


class PlannerService:
    """Coordinates implementation planning by delegating to the PlanningAgent.

    This service is a thin orchestration layer. It validates the incoming
    GitHub issue, invokes the planning agent, and returns the resulting
    ordered implementation plan. It performs no planning logic itself.

    Attributes:
        _planning_agent: The agent responsible for generating implementation plans.
    """

    def __init__(self, planning_agent: PlanningAgent) -> None:
        """Initializes the PlannerService.

        Args:
            planning_agent: An instance of PlanningAgent used to generate
                implementation plans from GitHub issues.
        """
        self._planning_agent = planning_agent

    async def create_plan(self, issue: GitHubIssue) -> ImplementationPlan:
        """Validates a GitHub issue and coordinates generation of its implementation plan.

        Args:
            issue: The GitHub issue to plan an implementation for.

        Returns:
            An ImplementationPlan model produced by the PlanningAgent.

        Raises:
            InvalidIssueError: If the issue is missing required information.
            PlanGenerationFailedError: If the planning agent fails to
                complete plan generation.
        """
        self._validate_issue(issue)

        logger.info(
            "planner_service.plan_generation_started",
            issue_number=issue.number,
            repository=issue.repository,
        )

        try:
            plan = await self._planning_agent.plan(issue)
        except Exception as exc:
            logger.error(
                "planner_service.plan_generation_failed",
                issue_number=issue.number,
                repository=issue.repository,
                error=str(exc),
            )
            raise PlanGenerationFailedError(
                f"Plan generation failed for issue #{issue.number} in {issue.repository}"
            ) from exc

        logger.info(
            "planner_service.plan_generation_completed",
            issue_number=issue.number,
            repository=issue.repository,
            step_count=len(plan.steps),
        )

        return plan

    def _validate_issue(self, issue: GitHubIssue) -> None:
        """Validates that the GitHub issue contains the information required for planning.

        Args:
            issue: The issue to validate.

        Raises:
            InvalidIssueError: If the issue number, repository, or title is missing.
        """
        if not issue.number:
            raise InvalidIssueError("GitHub issue must have a valid issue number.")

        if not issue.repository:
            raise InvalidIssueError("GitHub issue must specify a repository.")

        if not issue.title or not issue.title.strip():
            raise InvalidIssueError("GitHub issue must have a non-empty title.")