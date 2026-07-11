from __future__ import annotations

from pathlib import Path

import structlog

from kodiak.agents.base import AgentInput
from kodiak.agents.repository import (
    RepositoryAnalysis,
    RepositoryAnalyzerAgent,
)

logger = structlog.get_logger(__name__)


class InvalidRepositoryPathError(Exception):
    """Raised when a repository path is invalid."""


class RepositoryAnalysisFailedError(Exception):
    """Raised when repository analysis fails."""


class AnalyzeService:
    """CLI service that orchestrates repository analysis."""

    def __init__(self) -> None:
        self._agent = RepositoryAnalyzerAgent()

    async def analyze_repository(
        self,
        repository_path: str | Path,
        *,
        deep: bool = False,
    ) -> RepositoryAnalysis:
        path = self._validate_repository_path(repository_path)

        logger.info(
            "repository_analysis.started",
            path=str(path),
            deep=deep,
        )

        agent_input = AgentInput(
            task_id="cli-analyze",
            project_id="local",
            instruction="Analyze repository",
            context={
                "repository_path": path,
                "deep": deep,
            },
        )

        output = await self._agent.run(agent_input)

        if not output.success:
            raise RepositoryAnalysisFailedError(
                output.error or "Repository analysis failed."
            )

        analysis = output.result.get("analysis")

        if not isinstance(analysis, RepositoryAnalysis):
            raise RepositoryAnalysisFailedError(
                "Repository agent returned an invalid analysis object."
            )

        logger.info(
            "repository_analysis.completed",
            files=analysis.file_count,
            directories=analysis.directory_count,
        )

        return analysis

    @staticmethod
    def _validate_repository_path(
        repository_path: str | Path,
    ) -> Path:
        path = Path(repository_path).expanduser().resolve()

        if not path.exists():
            raise InvalidRepositoryPathError(
                f"Repository path does not exist: {path}"
            )

        if not path.is_dir():
            raise InvalidRepositoryPathError(
                f"Repository path is not a directory: {path}"
            )

        return path
