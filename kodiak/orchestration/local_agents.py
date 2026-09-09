"""Deterministic local agents and their CLI inventory."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from kodiak.orchestration.v1_models import (
    CheckResult,
    FileChange,
    GitDiffSummary,
    RepositoryContext,
    ReviewReport,
    TaskPlan,
    TaskStep,
)

_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".kodiak",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)


def _find_executable(name: str) -> str | None:
    """Find a local check runner without requiring provider configuration."""
    if name == "pytest" and importlib.util.find_spec("pytest") is not None:
        return sys.executable
    return shutil.which(name)


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(["git", *arguments], 1, "", str(exc))


class LocalGitAgent:
    """Inspect local Git state and create explicitly approved commits."""

    def inspect(self, repository_path: str | Path) -> GitDiffSummary:
        root = Path(repository_path).expanduser().resolve()
        probe = _run_git(root, "rev-parse", "--is-inside-work-tree")
        if probe.returncode != 0 or probe.stdout.strip().casefold() != "true":
            return GitDiffSummary(
                repository_path=str(root),
                is_git_repo=False,
                branch=None,
                dirty=False,
                warning="Path is not a Git repository.",
            )

        branch_result = _run_git(root, "branch", "--show-current")
        branch = branch_result.stdout.strip() or None
        status_result = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        if status_result.returncode == 0:
            for line in status_result.stdout.splitlines():
                if len(line) < 3:
                    continue
                code = line[:2]
                path = line[3:].split(" -> ")[-1].strip('"').replace("\\", "/")
                if code == "??":
                    untracked.append(path)
                else:
                    if code[0] not in {" ", "?"}:
                        staged.append(path)
                    if code[1] not in {" ", "?"}:
                        unstaged.append(path)
        diff_result = _run_git(root, "diff", "--stat", "HEAD")
        if diff_result.returncode != 0:
            diff_result = _run_git(root, "diff", "--stat")
        changed = sorted(set(staged) | set(unstaged) | set(untracked))
        return GitDiffSummary(
            repository_path=str(root),
            is_git_repo=True,
            branch=branch,
            dirty=bool(changed),
            changed_files=changed,
            staged_files=sorted(staged),
            unstaged_files=sorted(unstaged),
            untracked_files=sorted(untracked),
            diff_stat=diff_result.stdout.strip() if diff_result.returncode == 0 else "",
        )

    def commit(self, repository_path: str | Path, files: list[str], message: str) -> str:
        root = Path(repository_path).expanduser().resolve()
        summary = self.inspect(root)
        if not summary.is_git_repo:
            raise RuntimeError(summary.warning or "Path is not a Git repository.")
        unrelated = sorted(set(summary.staged_files) - set(files))
        if unrelated:
            raise RuntimeError(
                "Refusing to commit with unrelated staged files: " + ", ".join(unrelated)
            )
        add = _run_git(root, "add", "--", *files)
        if add.returncode != 0:
            raise RuntimeError(add.stderr.strip() or "git add failed")
        commit = _run_git(root, "commit", "-m", message, "--", *files)
        if commit.returncode != 0:
            raise RuntimeError(
                commit.stderr.strip() or commit.stdout.strip() or "git commit failed"
            )
        return commit.stdout.strip()


class RepositoryAnalyzer:
    """Collect the repository context consumed by deterministic v1 planning."""

    def analyze(self, repository_path: str | Path) -> RepositoryContext:
        root = Path(repository_path).expanduser().resolve()
        python_files = list(root.rglob("*.py"))
        package_files = [
            name
            for name in ("pyproject.toml", "requirements.txt", "setup.py", "package.json")
            if (root / name).is_file()
        ]
        source_dirs = [
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name not in _IGNORED_DIRECTORIES and any(path.glob("*.py"))
        ]
        test_dirs = [name for name in ("tests", "test") if (root / name).is_dir()]
        important_files = [
            name
            for name in ("README.md", "pyproject.toml", "requirements.txt")
            if (root / name).is_file()
        ]
        git = LocalGitAgent().inspect(root)
        warnings = [git.warning] if git.warning else []
        return RepositoryContext(
            path=str(root),
            language="python" if python_files else "unknown",
            framework="fastapi"
            if any(
                "FastAPI" in path.read_text(encoding="utf-8", errors="ignore")
                for path in python_files
            )
            else None,
            source_dirs=sorted(source_dirs),
            test_dirs=test_dirs,
            package_files=package_files,
            important_files=important_files,
            git_branch=git.branch,
            has_uncommitted_changes=git.dirty,
            warnings=warnings,
        )


class DeterministicPlannerAgent:
    """Build bounded plans for the local task templates supported by v1."""

    def plan(self, instruction: str, context: RepositoryContext) -> TaskPlan:
        normalized = " ".join(instruction.casefold().split())
        task_type = "manual"
        summary = "Manual implementation is required for this task."
        files: list[str] = []
        checks = ["pytest", "ruff check"]
        if "readme" in normalized:
            if "README.md" not in context.important_files:
                task_type = "readme_missing"
                summary = "README.md is missing; manual action is required."
            else:
                task_type = "readme_quickstart"
                summary = "Add or update the README quickstart section."
                files = ["README.md"]
        elif "health" in normalized and "test" in normalized:
            task_type = "health_check_test"
            summary = "Add a deterministic health-check test."
            files = ["tests/test_health_check.py"]
        elif "project structure" in normalized:
            task_type = "project_structure"
            summary = "Generate a project structure summary."
            files = ["PROJECT_STRUCTURE.md"]
        elif "basic unit test" in normalized:
            task_type = "basic_unit_test"
            summary = "Add a basic unit test file."
            files = ["tests/test_kodiak_smoke.py"]
        elif "missing import" in normalized:
            task_type = "missing_import"
            summary = "Add a clearly requested missing import."
            match = next(
                (token for token in instruction.split() if token.casefold().endswith(".py")),
                "",
            )
            files = [match] if match else []

        task_id = "local-plan"
        requires_approval = bool(
            __import__(
                "kodiak.orchestration.approval_manager", fromlist=["ApprovalManager"]
            ).ApprovalManager.risk_reasons(action="edit", paths=files)
        )
        return TaskPlan(
            task_id=task_id,
            instruction=instruction,
            repository_path=context.path,
            summary=summary,
            steps=[
                TaskStep("analyze", "Analyze repository", "repository"),
                TaskStep("implement", summary, "coder"),
                TaskStep("verify", "Run local checks", "tester"),
            ],
            files_to_inspect=list(files),
            files_to_modify=list(files),
            risk_level="high" if requires_approval else "low",
            requires_approval=requires_approval,
            expected_checks=checks,
            task_type=task_type,
        )


class TesterAgent:
    """Run pytest and Ruff and preserve complete check results."""

    def run_checks(self, repository_path: str | Path) -> list[CheckResult]:
        root = Path(repository_path).expanduser().resolve()
        commands = (
            ("pytest", [sys.executable, "-m", "pytest", "-q"]),
            ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        )
        results: list[CheckResult] = []
        for name, command in commands:
            if _find_executable(name) is None:
                results.append(
                    CheckResult(
                        name=name,
                        command=command,
                        success=None,
                        exit_code=None,
                        stdout="",
                        stderr="",
                        duration_seconds=0.0,
                        skipped_reason=f"{name} is not installed.",
                        skipped=True,
                    )
                )
                continue
            started = monotonic()
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
            results.append(
                CheckResult(
                    name=name,
                    command=command,
                    success=completed.returncode == 0,
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    duration_seconds=monotonic() - started,
                )
            )
        return results


class ReviewerAgent:
    """Summarize deterministic changes and local verification results."""

    def review(
        self,
        plan: TaskPlan,
        changes: list[FileChange],
        checks: list[CheckResult],
    ) -> ReviewReport:
        git = LocalGitAgent().inspect(plan.repository_path)
        failed = [check.name for check in checks if check.success is False]
        warnings = [git.warning] if git.warning else []
        recommendations: list[str] = []
        if plan.task_type in {"manual", "readme_missing"}:
            recommendations.append("Complete this task manually after reviewing the plan.")
        if failed:
            recommendations.append("Fix failed checks before accepting the changes.")
        return ReviewReport(
            task_id=plan.task_id,
            completed=not failed and plan.task_type not in {"manual", "readme_missing"},
            summary=f"Reviewed {len(changes)} change(s) and {len(checks)} check(s).",
            changed_files=[change.path for change in changes],
            checks=checks,
            risks=[],
            recommendations=recommendations,
            approval_required=plan.requires_approval,
            git_diff_stat=git.diff_stat,
            git_changed_files=git.changed_files,
            warnings=[f"Git review unavailable: {warning}" for warning in warnings],
            plan_summary=plan.summary,
            approval_status="pending" if plan.requires_approval else "not_required",
        )


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    name: str
    role: str
    status: str
    description: str
    backend: str
    module: str


_AGENTS = (
    AgentDescriptor(
        "PlannerAgent",
        "planner",
        "available",
        "Builds structured plans; the advanced implementation needs an LLM client.",
        "LLM-backed",
        "kodiak.agents.planner",
    ),
    AgentDescriptor(
        "RepositoryAgent",
        "repository analysis",
        "available",
        "Scans repository structure and project signals locally.",
        "deterministic",
        "kodiak.agents.repository",
    ),
    AgentDescriptor(
        "CoderAgent",
        "coder",
        "available",
        "Builds and validates patches; the advanced implementation needs an LLM client.",
        "LLM-backed",
        "kodiak.agents.coder",
    ),
    AgentDescriptor(
        "TesterAgent",
        "tester",
        "available",
        "Discovers and executes tests; optional test generation is LLM-backed.",
        "deterministic",
        "kodiak.agents.tester",
    ),
    AgentDescriptor(
        "ReviewerAgent",
        "reviewer",
        "available",
        "Reviews patches with deterministic checks and optional LLM analysis.",
        "LLM-backed",
        "kodiak.agents.reviewer",
    ),
    AgentDescriptor(
        "GitAgent",
        "git",
        "available",
        "Summarizes local changes and prepares commit metadata.",
        "deterministic",
        "kodiak.agents.git",
    ),
    AgentDescriptor(
        "MemoryAgent",
        "memory",
        "available",
        "Queries advanced memory stores through an injected LLM client.",
        "LLM-backed",
        "kodiak.agents.memory_agent",
    ),
)


def local_agent_descriptors() -> list[AgentDescriptor]:
    """Return descriptors backed by importable implementation modules."""
    return [agent for agent in _AGENTS if importlib.util.find_spec(agent.module) is not None]
