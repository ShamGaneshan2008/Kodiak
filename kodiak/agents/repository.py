import json
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class FileNode(BaseModel):
    path: str
    type: str
    size: int = 0


class RepoOutput(AgentOutput):
    result: list[FileNode] = Field(default_factory=list)
    summary: str = ""


class RepositoryAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="repository", description="Understands repository structure")

    async def execute(self, input_data: AgentInput) -> RepoOutput:
        self._logger.info("analyzing_repository", task=input_data.task)
        repo_path = input_data.context.get("repo_path", ".")
        tree = await self._scan_directory(Path(repo_path))

        prompt = (
            f"Analyze this repository structure:\n"
            f"{json.dumps([t.model_dump() for t in tree])}\n\nTask: {input_data.task}"
        )
        summary = await self._run_with_timing(prompt)
        return RepoOutput(success=True, result=tree, summary=summary)

    async def _scan_directory(self, path: Path, depth: int = 2) -> list[FileNode]:
        nodes: list[FileNode] = []
        if depth == 0 or not path.exists():
            return nodes
        for p in path.iterdir():
            if p.name.startswith("."):
                continue
            node_type = "dir" if p.is_dir() else "file"
            size = p.stat().st_size if p.is_file() else 0
            nodes.append(FileNode(path=str(p.relative_to(path)), type=node_type, size=size))
            if p.is_dir():
                nodes.extend(await self._scan_directory(p, depth - 1))
        return nodes