from __future__ import annotations

from pathlib import Path

import structlog

from kodiak.agents.execution_agent import ExecutionAgent
from kodiak.agents.models.analysis import RepositoryAnalysis
from kodiak.agents.models.execution import ExecutionResult
from kodiak.agents.models.issue import GitHubIssue
from kodiak.agents.models.plan import ImplementationPlan
from kodiak.agents.models.review import ReviewResult
from kodiak.agents.models.task import TaskOutcome
from kodiak.agents.planning_agent import PlanningAgent
from kodiak.agents.repository_analyzer_agent import RepositoryAnalyzerAgent
from kodiak.agents.review_agent import ReviewAgent

logger = structlog.get_logger(__name__)


class InvalidTaskInputError(Exception):
    """Raised when the provided repository path or GitHub issue is invalid."""


class TaskWorkflowFailedError(Exception):
    """Raised when any stage of the autonomous task workflow fails."""


class TaskService:
    """Coordinates the end-to-end autonomous task workflow.

    This service orchestrates the full lifecycle of an autonomous task by
    delegating each stage to the appropriate agent: repository analysis,
    planning, review, and execution. It contains no business logic of its
    own beyond validating inputs and sequencing agent calls.

    Attributes:
        _analyzer_agent: Agent responsible for repository analysis.
        _planning_agent: Agent responsible for implementation planning.
        _review_agent: Agent responsible for reviewing proposed plans.
        _execution_agent: Agent responsible for executing approved plans.
    """

    def __init__(
        self,
        analyzer_agent: RepositoryAnalyzerAgent,
        planning_agent: PlanningAgent,
        review_agent: ReviewAgent,
        execution_agent: ExecutionAgent,
    ) -> None:
        """Initializes the TaskService.

        Args:
            analyzer_agent: An instance of RepositoryAnalyzerAgent.
            planning_agent: An instance of PlanningAgent.
            review_agent: An instance of ReviewAgent.
            execution_agent: An instance of ExecutionAgent.
        """
        self._analyzer_agent = analyzer_agent
        self._planning_agent = planning_agent
        self._review_agent = review_agent
        self._execution_agent = execution_agent

    async def run_task(self, repository_path: str | Path, issue: GitHubIssue) -> TaskOutcome:
        """Runs the full autonomous workflow for a given repository and issue.

        Args:
            repository_path: The filesystem path to the repository to work on.
            issue: The GitHub issue describing the requested change.

        Returns:
            A TaskOutcome model summarizing the analysis, plan, review, and
            execution results.

        Raises:
            InvalidTaskInputError: If the repository path or issue is invalid.
            TaskWorkflowFailedError: If any stage of the workflow fails.
        """
        validated_path = self._validate_repository_path(repository_path)
        self._validate_issue(issue)

        logger.info(
            "task_service.workflow_started",
            path=str(validated_path),
            issue_number=issue.number,
            repository=issue.repository,
        )

        analysis = await self._run_analysis(validated_path)
        plan = await self._run_planning(issue, analysis)
        review = await self._run_review(plan)
        execution = await self._run_execution(plan, review)

        logger.info(
            "task_service.workflow_completed",
            path=str(validated_path),
            issue_number=issue.number,
            repository=issue.repository,
        )

        return TaskOutcome(
            analysis=analysis,
            plan=plan,
            review=review,
            execution=execution,
        )

    async def _run_analysis(self, repository_path: Path) -> RepositoryAnalysis:
        """Delegates repository analysis to the RepositoryAnalyzerAgent.

        Args:
            repository_path: The validated repository path.

        Returns:
            A RepositoryAnalysis model.

        Raises:
            TaskWorkflowFailedError: If the analysis stage fails.
        """
        try:
            return await self._analyzer_agent.analyze(repository_path)
        except Exception as exc:
            logger.error(
                "task_service.analysis_failed",
                path=str(repository_path),
                error=str(exc),
            )
            raise TaskWorkflowFailedError(
                f"Repository analysis failed for path: {repository_path}"
            ) from exc

    async def _run_planning(
        self, issue: GitHubIssue, analysis: RepositoryAnalysis
    ) -> ImplementationPlan:
        """Delegates implementation planning to the PlanningAgent.

        Args:
            issue: The GitHub issue to plan an implementation for.
            analysis: The repository analysis informing the plan.

        Returns:
            An ImplementationPlan model.

        Raises:
            TaskWorkflowFailedError: If the planning stage fails.
        """
        try:
            return await self._planning_agent.plan(issue, analysis)
        except Exception as exc:
            logger.error(
                "task_service.planning_failed",
                issue_number=issue.number,
                repository=issue.repository,
                error=str(exc),
            )
            raise TaskWorkflowFailedError(
                f"Plan generation failed for issue #{issue.number} in {issue.repository}"
            ) from exc

    async def _run_review(self, plan: ImplementationPlan) -> ReviewResult:
        """Delegates plan review to the ReviewAgent.

        Args:
            plan: The implementation plan to review.

        Returns:
            A ReviewResult model.

        Raises:
            TaskWorkflowFailedError: If the review stage fails.
        """
        try:
            return await self._review_agent.review(plan)
        except Exception as exc:
            logger.error(
                "task_service.review_failed",
                plan_id=plan.id,
                error=str(exc),
            )
            raise TaskWorkflowFailedError(f"Plan review failed for plan: {plan.id}") from exc

    async def _run_execution(
        self, plan: ImplementationPlan, review: ReviewResult
    ) -> ExecutionResult:
        """Delegates plan execution to the ExecutionAgent.

        Args:
            plan: The implementation plan to execute.
            review: The review result determining whether execution proceeds.

        Returns:
            An ExecutionResult model.

        Raises:
            TaskWorkflowFailedError: If the execution stage fails.
        """
        try:
            return await self._execution_agent.execute(plan, review)
        except Exception as exc:
            logger.error(
                "task_service.execution_failed",
                plan_id=plan.id,
                error=str(exc),
            )
            raise TaskWorkflowFailedError(f"Plan execution failed for plan: {plan.id}") from exc

    def _validate_repository_path(self, repository_path: str | Path) -> Path:
        """Validates that the given repository path exists and is a directory.

        Args:
            repository_path: The path to validate.

        Returns:
            The validated path resolved to an absolute Path instance.

        Raises:
            InvalidTaskInputError: If the path does not exist or is not a directory.
        """
        resolved_path = Path(repository_path).resolve()

        if not resolved_path.exists():
            raise InvalidTaskInputError(f"Repository path does not exist: {resolved_path}")

        if not resolved_path.is_dir():
            raise InvalidTaskInputError(f"Repository path is not a directory: {resolved_path}")

        return resolved_path

    def _validate_issue(self, issue: GitHubIssue) -> None:
        """Validates that the GitHub issue contains the information required for the workflow.

        Args:
            issue: The issue to validate.

        Raises:
            InvalidTaskInputError: If the issue number, repository, or title is missing.
        """
        if not issue.number:
            raise InvalidTaskInputError("GitHub issue must have a valid issue number.")

        if not issue.repository:
            raise InvalidTaskInputError("GitHub issue must specify a repository.")

        if not issue.title or not issue.title.strip():
            raise InvalidTaskInputError("GitHub issue must have a non-empty title.")
