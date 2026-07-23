from __future__ import annotations

import asyncio
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from kodiak.agents.base import (
    AgentInput,
    AgentOutput,
    AgentRole,
    BaseAgent,
)

_LANGUAGE_EXTENSION_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".c": "C",
    ".h": "C",
    ".hpp": "C++",
    ".cs": "C#",
    ".php": "PHP",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".html": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".toml": "TOML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".md": "Markdown",
    ".sql": "SQL",
    ".sh": "Shell",
}

_IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
}


@dataclass(slots=True)
class FileNode:
    path: str
    extension: str
    size_bytes: int
    language: str | None


@dataclass(slots=True)
class RepositoryAnalysis:
    root_path: Path
    files: list[FileNode]
    extension_counts: dict[str, int]
    language_stats: dict[str, int]
    total_size_bytes: int
    file_count: int
    directory_count: int


class RepositoryAnalyzerAgent(BaseAgent):
    """Repository scanner agent compatible with BaseAgent."""

    role = AgentRole.REPOSITORY

    def __init__(self) -> None:
        super().__init__()

    async def _run(self, input_: AgentInput) -> AgentOutput:
        repository = input_.context.get("repository_path")

        if repository is None:
            return self._make_error(
                input_,
                "repository_path missing from AgentInput.context",
            )

        analysis = await asyncio.to_thread(lambda: self._scan(Path(repository).resolve()))

        return self._make_output(
            input_,
            {"analysis": analysis},
        )

    def _scan(self, root: Path) -> RepositoryAnalysis:
        if not root.exists():
            raise FileNotFoundError(root)
        if not root.is_dir():
            raise NotADirectoryError(root)

        files: list[FileNode] = []
        extensions: Counter[str] = Counter()
        languages: Counter[str] = Counter()
        total_size = 0
        directory_count = 0

        for current, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _IGNORED_DIRECTORIES]

            if Path(current) != root:
                directory_count += 1

            for filename in filenames:
                file_path = Path(current) / filename

                try:
                    size = file_path.stat().st_size
                except OSError:
                    size = 0

                extension = file_path.suffix.lower()
                language = _LANGUAGE_EXTENSION_MAP.get(extension)

                node = FileNode(
                    path=str(file_path.relative_to(root)),
                    extension=extension,
                    size_bytes=size,
                    language=language,
                )

                files.append(node)
                total_size += size
                extensions[extension or "<none>"] += 1

                if language:
                    languages[language] += 1

        return RepositoryAnalysis(
            root_path=root,
            files=files,
            extension_counts=dict(extensions),
            language_stats=dict(languages),
            total_size_bytes=total_size,
            file_count=len(files),
            directory_count=directory_count,
        )
