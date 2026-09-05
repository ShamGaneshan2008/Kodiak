from __future__ import annotations

import ast
import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kodiak.agents.base import (
    AgentInput,
    AgentOutput,
    AgentRole,
    BaseAgent,
)
from kodiak.agents.repository_intelligence import RepositoryIntelligenceService

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
    structure: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, str]] = field(default_factory=list)
    llm: dict[str, Any] = field(default_factory=dict)


class RepositoryAnalyzerAgent(BaseAgent):
    """Repository scanner agent compatible with BaseAgent."""

    role = AgentRole.REPOSITORY

    def __init__(
        self,
        tool_router: Any | None = None,
        llm_client: Any | None = None,
        intelligence: RepositoryIntelligenceService | None = None,
    ) -> None:
        super().__init__(tool_router=tool_router)
        self._llm = llm_client
        self._intelligence = intelligence

    async def _run(self, input_: AgentInput) -> AgentOutput:
        repository = input_.context.get("repository_path")

        if repository is None:
            return self._make_error(
                input_,
                "repository_path missing from AgentInput.context",
            )

        analysis = await asyncio.to_thread(lambda: self._scan(Path(repository).resolve()))
        deep = bool(input_.context.get("deep", False))
        token_usage: dict[str, int] = {}
        if deep:
            token_usage = await self._enrich_with_llm(analysis, input_.task_id)

        result_payload: dict[str, Any] = {"analysis": analysis}
        if bool(input_.context.get("discover_issues", False)):
            intelligence = self._intelligence or RepositoryIntelligenceService(
                tool_router=self._tool_router
            )
            snapshot = await intelligence.scan(
                str(input_.context.get("repository_id", input_.task_id)),
                analysis.root_path,
                incremental=bool(input_.context.get("incremental", True)),
                run_tests=bool(input_.context.get("run_tests", False)),
                test_target=str(input_.context.get("test_target", "tests")),
                ci_failures=list(input_.context.get("ci_failures", [])),
            )
            result_payload["repository_intelligence"] = {
                "scan_id": snapshot.scan_id,
                "findings": [finding.to_dict() for finding in snapshot.findings],
                "files_processed": list(snapshot.files_processed),
                "files_unchanged": list(snapshot.files_unchanged),
                "dimensions": snapshot.dimensions,
            }
        if self._tool_router is not None:
            listing = await self.invoke_tool(
                "list_dir",
                {"path": "."},
                task_id=input_.task_id,
            )
            if listing.success:
                result_payload["tool_listing"] = listing.output

        return self._make_output(
            input_,
            result_payload,
            token_usage=token_usage,
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

        analysis = RepositoryAnalysis(
            root_path=root,
            files=files,
            extension_counts=dict(extensions),
            language_stats=dict(languages),
            total_size_bytes=total_size,
            file_count=len(files),
            directory_count=directory_count,
        )
        analysis.structure = self._structure(root, analysis)
        analysis.findings = self._static_findings(analysis)
        return analysis

    def _structure(self, root: Path, analysis: RepositoryAnalysis) -> dict[str, Any]:
        modules: list[dict[str, Any]] = []
        routes: list[str] = []
        for node in sorted(analysis.files, key=lambda item: item.path):
            if node.extension != ".py" or len(modules) >= 100:
                continue
            try:
                tree = ast.parse((root / node.path).read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            imports = sorted(
                {
                    alias.name.split(".")[0]
                    for item in ast.walk(tree)
                    if isinstance(item, (ast.Import, ast.ImportFrom))
                    for alias in (item.names if isinstance(item, ast.Import) else [item])
                    if getattr(alias, "name", None)
                }
            )
            functions = [
                item.name
                for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes = [item.name for item in tree.body if isinstance(item, ast.ClassDef)]
            modules.append(
                {
                    "path": node.path,
                    "imports": imports[:30],
                    "functions": functions[:30],
                    "classes": classes[:30],
                }
            )
            if "APIRouter" in (root / node.path).read_text(encoding="utf-8", errors="replace"):
                routes.append(node.path)
        return {
            "python_modules": modules,
            "api_route_files": routes,
            "test_files": [f.path for f in analysis.files if "test" in f.path.lower()][:100],
            "config_files": [
                f.path
                for f in analysis.files
                if f.path in {"pyproject.toml", "docker-compose.yml", "alembic.ini"}
            ],
        }

    def _static_findings(self, analysis: RepositoryAnalysis) -> list[dict[str, str]]:
        findings = [
            {
                "category": "architecture",
                "severity": "info",
                "message": (
                    "Detected "
                    f"{len(analysis.structure['python_modules'])} parseable Python modules."
                ),
            }
        ]
        if not analysis.structure["test_files"]:
            findings.append(
                {
                    "category": "testing_gaps",
                    "severity": "warning",
                    "message": "No test files were detected.",
                }
            )
        if not analysis.structure["config_files"]:
            findings.append(
                {
                    "category": "maintainability",
                    "severity": "warning",
                    "message": "No standard project configuration file was detected.",
                }
            )
        return findings

    async def _enrich_with_llm(self, analysis: RepositoryAnalysis, task_id: str) -> dict[str, int]:
        context = self._build_context(analysis)
        client = self._llm
        if client is None:
            try:
                from kodiak.llm.client import get_llm_client

                client = get_llm_client(task_id)
            except Exception as exc:
                analysis.llm = {"status": "unavailable", "error": f"LLM client unavailable: {exc}"}
                return {}
        try:
            response = await client.complete(
                messages=[{"role": "user", "content": context}],
                system=(
                    "Analyze this repository context. Return JSON with summary and findings "
                    "(category, severity, message)."
                ),
                model_preference="strong",
                max_tokens=1500,
                temperature=0.1,
            )
            data = json.loads(response.get("content", ""))
            findings = data.get("findings")
            if not isinstance(findings, list):
                raise ValueError("response missing findings list")
            analysis.findings.extend([item for item in findings if isinstance(item, dict)])
            usage = response.get("usage", {})
            analysis.llm = {
                "status": "completed",
                "provider": response.get("provider"),
                "model": response.get("model"),
                "summary": data.get("summary", ""),
            }
            return {
                key: int(value)
                for key, value in usage.items()
                if key in {"input_tokens", "output_tokens"} and isinstance(value, int)
            }
        except Exception as exc:
            analysis.llm = {"status": "unavailable", "error": str(exc)}
            return {}

    def _build_context(self, analysis: RepositoryAnalysis) -> str:
        payload = {
            "files": analysis.file_count,
            "languages": analysis.language_stats,
            "structure": analysis.structure,
            "static_findings": analysis.findings,
        }
        return json.dumps(payload, sort_keys=True, default=str)[:48_000]
