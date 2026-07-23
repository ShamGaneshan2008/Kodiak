from __future__ import annotations

from pathlib import Path

import structlog

from kodiak.agents.models.review import ReviewResult
from kodiak.agents.review_agent import ReviewAgent

logger = structlog.get_logger(__name__)


class InvalidRepositoryPathError(Exception):
    """Raised when the provided repository path does not exist or is not a directory."""


class ReviewFailedError(Exception):
    """Raised when the ReviewAgent fails to complete a review."""


class ReviewService:
    """Coordinates repository review by delegating to the ReviewAgent.

    This service is a thin orchestration layer. It validates the target
    repository path, invokes the review agent, and returns the resulting
    review. It performs no scanning, smell detection, or review logic
    itself.

    Attributes:
        _review_agent: The agent responsible for performing reviews.
    """

    def __init__(self, review_agent: ReviewAgent) -> None:
        """Initializes the ReviewService.

        Args:
            review_agent: An instance of ReviewAgent used to perform
                repository reviews.
        """
        self._review_agent = review_agent

    async def review_repository(self, repository_path: str | Path) -> ReviewResult:
        """Validates a repository path and coordinates its review.

        Args:
            repository_path: The filesystem path to the repository to review.

        Returns:
            A ReviewResult model produced by the ReviewAgent.

        Raises:
            InvalidRepositoryPathError: If the path does not exist or is not
                a directory.
            ReviewFailedError: If the review agent fails to complete the
                review.
        """
        validated_path = self._validate_repository_path(repository_path)

        logger.info("review_service.review_started", path=str(validated_path))

        try:
            result = await self._review_agent.review(validated_path)
        except Exception as exc:
            logger.error(
                "review_service.review_failed",
                path=str(validated_path),
                error=str(exc),
            )
            raise ReviewFailedError(f"Repository review failed for path: {validated_path}") from exc

        logger.info("review_service.review_completed", path=str(validated_path))

        return result

    def _validate_repository_path(self, repository_path: str | Path) -> Path:
        """Validates that the given path exists and is a directory.

        Args:
            repository_path: The path to validate.

        Returns:
            The validated path resolved to an absolute Path instance.

        Raises:
            InvalidRepositoryPathError: If the path does not exist or is not
                a directory.
        """
        resolved_path = Path(repository_path).resolve()

        if not resolved_path.exists():
            raise InvalidRepositoryPathError(f"Repository path does not exist: {resolved_path}")

        if not resolved_path.is_dir():
            raise InvalidRepositoryPathError(f"Repository path is not a directory: {resolved_path}")

        return resolved_path
