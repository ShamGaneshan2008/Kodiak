
from __future__ import annotations

from pathlib import Path

import structlog

from kodiak.agents.explain_agent import ExplainAgent
from kodiak.agents.models.explanation import ExplanationResult

logger = structlog.get_logger(__name__)


class InvalidFilePathError(Exception):
    """Raised when the provided file path does not exist or is not a file."""


class ExplanationFailedError(Exception):
    """Raised when the ExplainAgent fails to produce an explanation."""


class ExplainService:
    """Coordinates code explanation by delegating to the ExplainAgent.

    This service is a thin orchestration layer. It validates the target
    file path, invokes the explain agent, and returns the resulting
    explanation. It performs no parsing, analysis, or prompt generation
    itself.

    Attributes:
        _explain_agent: The agent responsible for generating explanations.
    """

    def __init__(self, explain_agent: ExplainAgent) -> None:
        """Initializes the ExplainService.

        Args:
            explain_agent: An instance of ExplainAgent used to generate
                explanations for source files.
        """
        self._explain_agent = explain_agent

    async def explain_file(self, file_path: str | Path) -> ExplanationResult:
        """Validates a file path and coordinates generation of its explanation.

        Args:
            file_path: The filesystem path to the file to explain.

        Returns:
            An ExplanationResult model produced by the ExplainAgent.

        Raises:
            InvalidFilePathError: If the path does not exist or is not a file.
            ExplanationFailedError: If the explain agent fails to complete
                the explanation.
        """
        validated_path = self._validate_file_path(file_path)

        logger.info("explain_service.explanation_started", path=str(validated_path))

        try:
            explanation = await self._explain_agent.explain(validated_path)
        except Exception as exc:
            logger.error(
                "explain_service.explanation_failed",
                path=str(validated_path),
                error=str(exc),
            )
            raise ExplanationFailedError(
                f"Explanation generation failed for file: {validated_path}"
            ) from exc

        logger.info("explain_service.explanation_completed", path=str(validated_path))

        return explanation

    def _validate_file_path(self, file_path: str | Path) -> Path:
        """Validates that the given path exists and is a file.

        Args:
            file_path: The path to validate.

        Returns:
            The validated path resolved to an absolute Path instance.

        Raises:
            InvalidFilePathError: If the path does not exist or is not a file.
        """
        resolved_path = Path(file_path).resolve()

        if not resolved_path.exists():
            raise InvalidFilePathError(f"File path does not exist: {resolved_path}")

        if not resolved_path.is_file():
            raise InvalidFilePathError(f"File path is not a file: {resolved_path}")

        return resolved_path
