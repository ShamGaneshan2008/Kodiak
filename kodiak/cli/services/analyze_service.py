from __future__ import annotations

from pathlib import Path

import structlog

from kodiak.agents.repository_analyzer_agent import RepositoryAnalyzerAgent
from kodiak.agents.models.analysis import RepositoryAnalysis

logger = structlog.get_logger(__name__)


class InvalidRepositoryPathError(Exception):
    """Raised when the provided repository path is not a valid, accessible directory."""


class RepositoryAnalysisFailedError(Exception):
    """Raised when the RepositoryAnalyzerAgent fails to complete an analysis."""


class AnalyzeService:
    """Coordinates repository analysis by delegating to the RepositoryAnalyzerAgent.

    This service is a thin orchestration layer. It validates the repository
    path, invokes the analyzer agent, and returns the resulting model. It
    performs no filesystem scanning, parsing, or analysis logic itself.

    Attributes:
        _analyzer_agent: The agent responsible for performing repository analysis.
    """

    def __init__(self, analyzer_agent: RepositoryAnalyzerAgent) -> None:
        """Initializes the AnalyzeService.

        Args:
            analyzer_agent: An instance of RepositoryAnalyzerAgent used to
                perform the actual repository analysis.
        """
        self._analyzer_agent = analyzer_agent

    async def analyze_repository(self, repository_path: str | Path) -> RepositoryAnalysis:
        """Validates a repository path and coordinates its analysis.

        Args:
            repository_path: The filesystem path to the repository to analyze.

        Returns:
            A RepositoryAnalysis model produced by the RepositoryAnalyzerAgent.

        Raises:
            InvalidRepositoryPathError: If the path does not exist or is not
                a directory.
            RepositoryAnalysisFailedError: If the analyzer agent fails to
                complete the analysis.
        """
        validated_path = self._validate_repository_path(repository_path)

        logger.info("analyze_service.analysis_started", path=str(validated_path))

        try:
            analysis = await self._analyzer_agent.analyze(validated_path)
        except Exception as exc:
            logger.error(
                "analyze_service.analysis_failed",
                path=str(validated_path),
                error=str(exc),
            )
            raise RepositoryAnalysisFailedError(
                f"Repository analysis failed for path: {validated_path}"
            ) from exc

        logger.info("analyze_service.analysis_completed", path=str(validated_path))

        return analysis

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
            raise InvalidRepositoryPathError(
                f"Repository path does not exist: {resolved_path}"
            )

        if not resolved_path.is_dir():
            raise InvalidRepositoryPathError(
                f"Repository path is not a directory: {resolved_path}"
            )

        return resolved_path