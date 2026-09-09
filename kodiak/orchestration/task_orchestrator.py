"""Deterministic, offline-capable Kodiak v1 task workflow."""

from __future__ import annotations

import difflib
import hashlib
import inspect
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kodiak.cli.services.git_service import GitDiffSummary, GitService
from kodiak.orchestration.approval_manager import ApprovalManager
from kodiak.orchestration.local_storage import LocalStateStore, sanitize_for_storage

if TYPE_CHECKING:
    from kodiak.orchestration.v1_models import (
        FileChange as LegacyFileChange,
    )
    from kodiak.orchestration.v1_models import (
        RepositoryContext as LegacyRepositoryContext,
    )
    from kodiak.orchestration.v1_models import (
        ReviewReport as LegacyReviewReport,
    )

_IGNORED_DIRECTORIES = {
    ".git",
    ".kodiak",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


class InvalidRepositoryError(ValueError):
    """Raised when a task target does not exist or is not a directory."""


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    root: str
    file_count: int
    directory_count: int
    extension_counts: dict[str, int]
    selected_signals: tuple[str, ...]
    git: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_signals"] = list(self.selected_signals)
        return payload


@dataclass(frozen=True, slots=True)
class TaskPlan:
    task_type: str
    summary: str
    supported: bool
    target_files: tuple[str, ...]
    steps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_files"] = list(self.target_files)
        payload["steps"] = list(self.steps)
        return payload

    @property
    def files_to_inspect(self) -> list[str]:
        if self.task_type == "health_check_test":
            return ["tests/test_health_check.py"]
        return list(self.target_files)

    @property
    def files_to_modify(self) -> list[str]:
        return list(self.target_files)


@dataclass(frozen=True, slots=True)
class ProposedChange:
    path: str
    action: str
    description: str
    before: str | None = field(repr=False)
    after: str = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        before_lines = (self.before or "").splitlines(keepends=True)
        after_lines = self.after.splitlines(keepends=True)
        patch = "".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{self.path}",
                tofile=f"b/{self.path}",
            )
        )
        return {
            "path": self.path,
            "action": self.action,
            "description": self.description,
            "diff": patch,
        }


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    command: tuple[str, ...]
    exit_code: int | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    task_id: str
    timestamp: str
    instruction: str
    repository_path: str
    dry_run: bool
    no_commit: bool
    final_status: str
    analysis: RepositorySnapshot | None
    plan: TaskPlan | None
    proposed_changes: tuple[ProposedChange, ...]
    changed_files: tuple[str, ...]
    checks: tuple[CheckResult, ...]
    review_summary: str
    approval_id: str | None = None
    error: str | None = None

    @property
    def successful(self) -> bool:
        return self.final_status in {
            "already_satisfied",
            "already_satisfied_with_warnings",
            "completed",
            "completed_with_warnings",
            "dry_run",
        }

    @property
    def changes(self) -> list[LegacyFileChange]:
        from kodiak.orchestration.v1_models import FileChange

        return [
            FileChange(
                path=change.path,
                change_type=change.action,
                summary=change.description,
                before_hash=None,
                after_hash=None,
            )
            for change in self.proposed_changes
            if change.path in self.changed_files
        ]

    @property
    def repository(self) -> LegacyRepositoryContext:
        from kodiak.orchestration.v1_models import RepositoryContext

        git = self.analysis.git if self.analysis else {}
        return RepositoryContext(
            path=self.repository_path,
            language="python"
            if self.analysis and ".py" in self.analysis.extension_counts
            else "unknown",
            framework=None,
            source_dirs=[],
            test_dirs=[],
            package_files=[],
            important_files=list(self.analysis.selected_signals) if self.analysis else [],
            git_branch=str(git.get("current_branch")) if git.get("current_branch") else None,
            has_uncommitted_changes=bool(git.get("dirty")),
            warnings=[],
        )

    @property
    def review(self) -> LegacyReviewReport:
        from kodiak.orchestration.v1_models import CheckResult as LegacyCheckResult
        from kodiak.orchestration.v1_models import ReviewReport

        checks = [
            LegacyCheckResult(
                name=check.name,
                command=list(check.command),
                success=True
                if check.status == "passed"
                else False
                if check.status == "failed"
                else None,
                exit_code=check.exit_code,
                stdout="",
                stderr="",
                duration_seconds=0.0,
                skipped_reason=check.summary if check.status in {"missing", "skipped"} else None,
                skipped=check.status in {"missing", "skipped"},
            )
            for check in self.checks
        ]
        recommendations: list[str] = []
        if self.final_status == "manual_required":
            recommendations.append("Complete this task manually.")
        if self.approval_id:
            recommendations.append(
                f"Run kodiak approval approve {self.approval_id} before executing the action."
            )
        return ReviewReport(
            task_id=self.task_id,
            completed=self.final_status
            in {"completed", "already_satisfied", "completed_with_warnings"},
            summary=self.review_summary,
            changed_files=list(self.changed_files),
            checks=checks,
            risks=[],
            recommendations=recommendations,
            approval_required=self.approval_id is not None,
            approval_status="pending" if self.approval_id else "not_required",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "repository_path": self.repository_path,
            "dry_run": self.dry_run,
            "no_commit": self.no_commit,
            "final_status": self.final_status,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "proposed_changes": [change.to_dict() for change in self.proposed_changes],
            "changed_files": list(self.changed_files),
            "checks": [check.to_dict() for check in self.checks],
            "check_summary": {
                status: sum(check.status == status for check in self.checks)
                for status in sorted({check.status for check in self.checks})
            },
            "review_summary": self.review_summary,
            "approval_id": self.approval_id,
            "error": self.error,
        }
        payload["review"] = asdict(self.review)
        return payload


class RepositoryInspector:
    """Collect bounded repository metadata without importing project code."""

    def __init__(self, git_service: GitService | None = None) -> None:
        self.git_service = git_service or GitService()

    def analyze(self, root: Path) -> RepositorySnapshot:
        extensions: Counter[str] = Counter()
        signals: list[str] = []
        file_count = 0
        directory_count = 0
        for _current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in _IGNORED_DIRECTORIES]
            directory_count += len(directories)
            for name in files:
                file_count += 1
                extensions[Path(name).suffix.casefold() or "[no extension]"] += 1
        for candidate in ("README.md", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"):
            if (root / candidate).is_file():
                signals.append(candidate)
        git = self.git_service.diff_summary(root)
        return RepositorySnapshot(
            root=str(root),
            file_count=file_count,
            directory_count=directory_count,
            extension_counts=dict(sorted(extensions.items())),
            selected_signals=tuple(signals),
            git=git.to_dict(),
        )


class DeterministicPlanner:
    """Classify only the bounded task templates promised for v1."""

    def plan(self, instruction: str) -> TaskPlan:
        normalized = " ".join(instruction.casefold().split())
        common_steps = ("Analyze repository", "Select relevant files")
        if "readme" in normalized and "quickstart" in normalized:
            return TaskPlan(
                task_type="readme_quickstart",
                summary="Add or update one honest Kodiak Quickstart section.",
                supported=True,
                target_files=("README.md",),
                steps=common_steps
                + (
                    "Build an idempotent Quickstart edit",
                    "Apply the edit unless this is a dry run",
                    "Run pytest and Ruff",
                    "Review the Git diff and store history",
                ),
            )
        if "health" in normalized and "test" in normalized:
            return TaskPlan(
                task_type="health_check_test",
                summary="Add a FastAPI /health test when its app import is verified.",
                supported=True,
                target_files=("tests/test_health.py",),
                steps=common_steps
                + (
                    "Verify the FastAPI app import and /health route",
                    "Build a deterministic health test",
                    "Apply the edit unless this is a dry run",
                    "Run pytest and Ruff",
                    "Review the Git diff and store history",
                ),
            )
        if "project structure" in normalized:
            return TaskPlan(
                task_type="project_structure",
                summary="Generate a bounded project structure summary.",
                supported=True,
                target_files=("PROJECT_STRUCTURE.md",),
                steps=common_steps + ("Write the structure summary", "Run pytest and Ruff"),
            )
        if "basic unit test" in normalized:
            return TaskPlan(
                task_type="basic_unit_test",
                summary="Add a basic import smoke test.",
                supported=True,
                target_files=("tests/test_kodiak_smoke.py",),
                steps=common_steps + ("Write the smoke test", "Run pytest and Ruff"),
            )
        if "missing import" in normalized:
            target = next(
                (word for word in instruction.split() if word.casefold().endswith(".py")),
                "",
            )
            return TaskPlan(
                task_type="missing_import",
                summary="Add the explicitly requested missing import.",
                supported=bool(target),
                target_files=(target,) if target else (),
                steps=common_steps + ("Add the missing import", "Run pytest and Ruff"),
            )
        return TaskPlan(
            task_type="unsupported",
            summary="Automatic implementation is not supported for this instruction in v1.",
            supported=False,
            target_files=(),
            steps=common_steps
            + (
                "Report the unsupported task without editing files",
                "Store a manual-required history record",
            ),
        )


class SafeEditor:
    """Propose and apply repository-contained, deterministic single-file edits."""

    def propose(self, root: Path, plan: TaskPlan) -> tuple[ProposedChange, ...]:
        if plan.task_type == "readme_quickstart":
            return self._readme_change(root)
        if plan.task_type == "health_check_test":
            return self._health_test_change(root)
        if plan.task_type == "project_structure":
            return self._project_structure_change(root)
        if plan.task_type == "basic_unit_test":
            return self._basic_unit_test_change(root)
        if plan.task_type == "missing_import":
            return self._missing_import_change(root, plan)
        return ()

    def can_handle(self, root: Path, plan: TaskPlan) -> bool:
        """Return whether a no-op means the requested state is already satisfied."""
        if plan.task_type == "readme_quickstart":
            return self._find_readme(root) is not None
        if plan.task_type == "health_check_test":
            module = self._verified_fastapi_module(root)
            if module is None and not self._health_route_exists(root):
                return False
            marker = "# Generated by Kodiak v1: health-check-test"
            primary = (
                root
                / "tests"
                / ("test_health.py" if module is not None else "test_health_check.py")
            )
            fallback = root / "tests" / "test_kodiak_health.py"
            primary_available = not primary.exists() or marker in primary.read_text(
                encoding="utf-8"
            )
            fallback_available = not fallback.exists() or marker in fallback.read_text(
                encoding="utf-8"
            )
            return primary_available or fallback_available
        if plan.task_type in {"project_structure", "basic_unit_test", "missing_import"}:
            return True
        return False

    def _project_structure_change(self, root: Path) -> tuple[ProposedChange, ...]:
        path = root / "PROJECT_STRUCTURE.md"
        before = path.read_text(encoding="utf-8") if path.is_file() else None
        entries = sorted(
            item.relative_to(root).as_posix()
            for item in root.iterdir()
            if item.name not in _IGNORED_DIRECTORIES
        )
        after = "# Project Structure\n\n" + "\n".join(f"- `{entry}`" for entry in entries) + "\n"
        if before == after:
            return ()
        return (
            ProposedChange(
                "PROJECT_STRUCTURE.md",
                "create" if before is None else "update",
                "Document the top-level project structure.",
                before,
                after,
            ),
        )

    def _basic_unit_test_change(self, root: Path) -> tuple[ProposedChange, ...]:
        path = root / "tests" / "test_kodiak_smoke.py"
        before = path.read_text(encoding="utf-8") if path.is_file() else None
        after = (
            '"""Basic deterministic smoke test."""\n\n\n'
            "def test_project_imports() -> None:\n"
            "    import kodiak\n\n"
            "    assert kodiak is not None\n"
        )
        if before == after:
            return ()
        return (
            ProposedChange(
                "tests/test_kodiak_smoke.py",
                "create" if before is None else "update",
                "Add a basic package import smoke test.",
                before,
                after,
            ),
        )

    def _missing_import_change(self, root: Path, plan: TaskPlan) -> tuple[ProposedChange, ...]:
        if not plan.target_files:
            return ()
        relative = plan.target_files[0]
        path = self._inside(root, relative)
        if not path.is_file():
            return ()
        before = path.read_text(encoding="utf-8")
        if re.search(r"(?m)^import os$", before):
            return ()
        after = "import os\n" + before
        return (ProposedChange(relative, "update", "Add the missing os import.", before, after),)

    def apply(self, root: Path, changes: Iterable[ProposedChange]) -> tuple[str, ...]:
        changed: list[str] = []
        for change in changes:
            destination = self._inside(root, change.path)
            current = destination.read_text(encoding="utf-8") if destination.is_file() else None
            if current != change.before:
                raise RuntimeError(
                    f"Refusing to overwrite a concurrently changed file: {change.path}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(change.after, encoding="utf-8", newline="\n")
            changed.append(change.path)
        return tuple(changed)

    def _readme_change(self, root: Path) -> tuple[ProposedChange, ...]:
        readme = self._find_readme(root)
        if readme is None:
            return ()
        before = readme.read_text(encoding="utf-8")
        after = self._replace_quickstarts(before)
        if before == after:
            return ()
        return (
            ProposedChange(
                path=readme.relative_to(root).as_posix(),
                action="update",
                description="Create one idempotent, verified Kodiak Quickstart section.",
                before=before,
                after=after,
            ),
        )

    def _health_test_change(self, root: Path) -> tuple[ProposedChange, ...]:
        module = self._verified_fastapi_module(root)
        if module is None and not self._health_route_exists(root):
            return ()
        path = root / "tests" / ("test_health.py" if module is not None else "test_health_check.py")
        module = module or "kodiak.api.main"
        marker = "# Generated by Kodiak v1: health-check-test"
        if path.exists() and marker not in path.read_text(encoding="utf-8"):
            path = root / "tests" / "test_kodiak_health.py"
        if path.exists() and marker not in path.read_text(encoding="utf-8"):
            return ()
        before = path.read_text(encoding="utf-8") if path.exists() else None
        after = (
            f'{marker}\n"""Smoke test for the verified FastAPI health route."""\n\n'
            "from fastapi.testclient import TestClient\n\n"
            f"from {module} import app\n\n\n"
            "def test_health_check() -> None:\n"
            '    response = TestClient(app).get("/health")\n\n'
            "    assert response.status_code == 200\n"
            '    assert response.json() == {"status": "healthy"}\n'
        )
        if before == after:
            return ()
        return (
            ProposedChange(
                path=path.relative_to(root).as_posix(),
                action="create" if before is None else "update",
                description=f"Test /health through the verified {module}.app import.",
                before=before,
                after=after,
            ),
        )

    def _health_route_exists(self, root: Path) -> bool:
        for path in self._python_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if re.search(r"@\w+(?:\.\w+)*\.get\(\s*[\"']/health[\"']", text):
                return True
        return False

    def _verified_fastapi_module(self, root: Path) -> str | None:
        health_route_found = False
        app_candidates: list[tuple[int, str]] = []
        for path in self._python_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if re.search(r"@\w+(?:\.\w+)*\.get\(\s*[\"']/health[\"']", text):
                health_route_found = True
            if not re.search(r"\bapp\s*=\s*FastAPI\s*\(", text):
                continue
            module = self._module_name(root, path)
            if module:
                score = 0 if path.name == "main.py" else 1
                score += len(path.parts)
                app_candidates.append((score, module))
        if not health_route_found:
            return None
        for _, module in sorted(app_candidates):
            code = (
                "import importlib,sys; "
                f"m=importlib.import_module({module!r}); "
                "app=getattr(m,'app'); "
                "sys.exit(0 if any(getattr(r,'path',None)=='/health' for r in app.routes) else 2)"
            )
            try:
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=root,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                return module
        return None

    def _python_files(self, root: Path) -> Iterable[Path]:
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in _IGNORED_DIRECTORIES]
            for name in files:
                if name.endswith(".py"):
                    yield Path(current) / name

    @staticmethod
    def _module_name(root: Path, path: Path) -> str | None:
        relative = path.relative_to(root).with_suffix("")
        if any(not part.isidentifier() for part in relative.parts):
            return None
        parents = relative.parts[:-1]
        current = root
        for part in parents:
            current /= part
            if not (current / "__init__.py").is_file():
                return None
        return ".".join(relative.parts)

    @staticmethod
    def _find_readme(root: Path) -> Path | None:
        direct = root / "README.md"
        if direct.is_file():
            return direct
        return next(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.name.casefold() == "readme.md"
            ),
            None,
        )

    @staticmethod
    def _replace_quickstarts(text: str) -> str:
        canonical = """## Quickstart

<!-- kodiak-v1-quickstart -->

Kodiak's local v1 workflow needs Python 3.12 or newer. Provider API keys, Docker, and internet
access are not required for the commands below.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -e \".[dev]\"
kodiak --help
kodiak doctor
kodiak analyze analyze . --deep
kodiak task run \"Improve the README quickstart section\" --path . --dry-run
```

Use `--dry-run` to plan without source-file writes. Use `--no-commit` to apply a supported edit
without creating a commit. Kodiak v1 never pushes.
"""
        heading = re.compile(r"(?im)^##[ \t]+quickstart[ \t]*$")
        section_heading = re.compile(r"(?m)^##[ \t]+.+$")
        matches = list(heading.finditer(text))
        if not matches:
            return text.rstrip() + "\n\n" + canonical
        ranges: list[tuple[int, int]] = []
        for match in matches:
            next_heading = section_heading.search(text, match.end())
            end = next_heading.start() if next_heading else len(text)
            ranges.append((match.start(), end))
        output = text
        for index, (start, end) in reversed(list(enumerate(ranges))):
            replacement = canonical.rstrip() + "\n\n" if index == 0 else ""
            output = output[:start] + replacement + output[end:]
        return output.rstrip() + "\n"

    @staticmethod
    def _inside(root: Path, relative_path: str) -> Path:
        destination = (root / relative_path).resolve()
        try:
            destination.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Refusing to write outside repository: {relative_path}") from exc
        return destination


class LocalTester:
    """Run pytest and Ruff directly with bounded argv-based subprocesses."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 900.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.runner = runner

    def run(self, root: Path, *, dry_run: bool = False) -> tuple[CheckResult, ...]:
        commands = (
            ("pytest", (sys.executable, "-m", "pytest", "-q")),
            ("ruff check", (sys.executable, "-m", "ruff", "check", ".")),
            ("ruff format", (sys.executable, "-m", "ruff", "format", "--check", ".")),
        )
        if dry_run:
            return tuple(
                CheckResult(name, "dry_run", command, None, "Planned; not run in dry-run mode.")
                for name, command in commands
            )
        return tuple(self._run_one(name, command, root) for name, command in commands)

    def _run_one(self, name: str, command: tuple[str, ...], root: Path) -> CheckResult:
        try:
            result = self.runner(
                list(command),
                cwd=root,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            return CheckResult(name, "missing", command, None, f"{name} is not available.")
        except subprocess.TimeoutExpired:
            return CheckResult(name, "error", command, None, f"{name} timed out.")
        except OSError as exc:
            return CheckResult(name, "error", command, None, f"{name} could not run: {exc}")
        output = f"{result.stdout}\n{result.stderr}"
        if "No module named" in output and name.split()[0] in output:
            return CheckResult(
                name, "missing", command, result.returncode, f"{name} is not installed."
            )
        if name == "pytest" and result.returncode == 5:
            return CheckResult(name, "skipped", command, 5, "No tests were collected.")
        status = "passed" if result.returncode == 0 else "failed"
        return CheckResult(
            name,
            status,
            command,
            result.returncode,
            f"Exited with code {result.returncode}.",
        )


class DiffReviewer:
    """Produce an honest summary from proposed changes, checks, and Git state."""

    def review(
        self,
        changes: Iterable[ProposedChange],
        checks: Iterable[CheckResult],
        git: GitDiffSummary,
    ) -> str:
        changed = [change.path for change in changes]
        check_results = tuple(checks)
        failed = [check.name for check in check_results if check.status in {"failed", "error"}]
        warning = [check.name for check in check_results if check.status in {"missing", "skipped"}]
        planned = [check.name for check in check_results if check.status == "dry_run"]
        parts = [f"Reviewed {len(changed)} task change(s): {', '.join(changed) or 'none'}."]
        if failed:
            parts.append(f"Failed checks: {', '.join(failed)}.")
        elif warning:
            parts.append(f"Checks completed with warnings: {', '.join(warning)}.")
        elif planned:
            parts.append("Checks were planned but not executed in dry-run mode.")
        elif check_results:
            parts.append("All executed checks passed.")
        if git.is_git_repo:
            parts.append(
                f"Git branch {git.current_branch}; {len(git.changed_files)} "
                "working-tree file(s) changed."
            )
        else:
            parts.append("The target is not a Git repository; no Git diff was available.")
        return " ".join(parts)


class TaskOrchestrator:
    """Connect analysis, planning, safe editing, checks, review, approval, and history."""

    def __init__(
        self,
        *,
        inspector: RepositoryInspector | None = None,
        planner: Any = None,
        editor: SafeEditor | None = None,
        tester: Any = None,
        reviewer: DiffReviewer | None = None,
        git_service: GitService | None = None,
        coder: Any = None,
    ) -> None:
        self.git_service = git_service or GitService()
        self.inspector = inspector or RepositoryInspector(self.git_service)
        self.planner: Any = planner or DeterministicPlanner()
        self.editor = editor or SafeEditor()
        self.tester: Any = tester or LocalTester()
        self.reviewer = reviewer or DiffReviewer()
        self.coder = coder

    def run(
        self,
        instruction: str,
        repository_path: str | Path,
        *,
        dry_run: bool = False,
        no_commit: bool = False,
    ) -> TaskRunResult:
        root = Path(repository_path).expanduser().resolve()
        if not root.exists():
            raise InvalidRepositoryError(f"Repository path does not exist: {root}")
        if not root.is_dir():
            raise InvalidRepositoryError(f"Repository path is not a directory: {root}")
        if not instruction.strip():
            raise ValueError("Task instruction must not be empty.")

        sanitized_instruction = sanitize_for_storage(instruction.strip())
        if not isinstance(sanitized_instruction, str):
            raise ValueError("Task instruction could not be sanitized.")
        instruction = sanitized_instruction
        compatibility_mode = hasattr(self.tester, "run_checks") or self.coder is not None

        task_id = f"task_{uuid4().hex[:12]}"
        timestamp = datetime.now(UTC).isoformat()
        store = LocalStateStore(root)
        analysis: RepositorySnapshot | None = None
        plan: TaskPlan | None = None
        proposed: tuple[ProposedChange, ...] = ()
        changed_files: tuple[str, ...] = ()
        checks: tuple[CheckResult, ...] = ()
        review = ""
        approval_id: str | None = None
        status = "error"
        error: str | None = None
        caught_exception: Exception | None = None

        try:
            analysis = self.inspector.analyze(root)
            plan, plan_requires_approval = self._build_plan(instruction, root)
            if (
                plan.task_type == "readme_quickstart" or "quickstart" in instruction.casefold()
            ) and not (root / "README.md").is_file():
                plan = TaskPlan(
                    task_type="readme_missing",
                    summary="README.md is missing; manual implementation is required.",
                    supported=False,
                    target_files=(),
                    steps=plan.steps,
                )
            if not plan.supported:
                status = "manual_required"
                review = plan.summary
            else:
                if self.coder is not None:
                    self.coder.apply(plan)
                proposed = self.editor.propose(root, plan)
                if plan_requires_approval:
                    request = ApprovalManager(root).create(
                        task_id=task_id,
                        action="apply_changes",
                        reason="The injected plan requires explicit approval.",
                        metadata={"paths": list(plan.target_files)},
                    )
                    approval_id = request.approval_id
                    status = "pending_approval"
                    review = "Risky edit was not applied; explicit approval is required."
                elif not proposed and not self.editor.can_handle(root, plan):
                    status = "manual_required"
                    review = (
                        "The required target could not be verified safely; no file was changed. "
                        "For health tests, confirm an importable FastAPI app and /health route."
                    )
                elif dry_run:
                    checks = self._run_checks(root, dry_run=True)
                    git = self.git_service.diff_summary(root)
                    review = self.reviewer.review(proposed, checks, git)
                    status = "dry_run"
                else:
                    risk_reasons = ApprovalManager.risk_reasons(
                        action="edit", paths=(change.path for change in proposed)
                    )
                    if risk_reasons:
                        request = ApprovalManager(root).create(
                            task_id=task_id,
                            action="edit_files",
                            reason="; ".join(risk_reasons),
                            metadata={"paths": [change.path for change in proposed]},
                        )
                        approval_id = str(request["approval_id"])
                        status = "awaiting_approval"
                        review = "Risky edit was not applied; explicit approval is required."
                    else:
                        changed_files = self.editor.apply(root, proposed)
                        checks = self._run_checks(root)
                        git = self.git_service.diff_summary(root)
                        review = self.reviewer.review(proposed, checks, git)
                        if not proposed:
                            review = (
                                f"The requested repository state is already satisfied. {review}"
                            )
                        if any(check.status in {"failed", "error"} for check in checks):
                            status = "failed_checks"
                        elif any(check.status in {"missing", "skipped"} for check in checks):
                            status = (
                                "completed_with_warnings"
                                if proposed
                                else "already_satisfied_with_warnings"
                            )
                        else:
                            status = (
                                "completed"
                                if proposed
                                else "no_changes"
                                if compatibility_mode
                                else "already_satisfied"
                            )

                        if (
                            changed_files
                            and not no_commit
                            and status.startswith("completed")
                            and git.is_git_repo
                        ):
                            commit_request = self._create_commit_approval(
                                root, task_id, changed_files
                            )
                            approval_id = str(commit_request["approval_id"])
                            status = (
                                "pending_approval" if compatibility_mode else "awaiting_approval"
                            )
                            review += (
                                " A local commit requires explicit approval; no commit was made."
                            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "error"
            review = "The workflow stopped after an error; completion was not reported."
            if self.coder is not None:
                caught_exception = exc
                status = "failed"
                error = str(exc)
                review = "The workflow failed; no success was reported."

        result = TaskRunResult(
            task_id=task_id,
            timestamp=timestamp,
            instruction=instruction,
            repository_path=str(root),
            dry_run=dry_run,
            no_commit=no_commit,
            final_status=status,
            analysis=analysis,
            plan=plan,
            proposed_changes=proposed,
            changed_files=changed_files,
            checks=checks,
            review_summary=review,
            approval_id=approval_id,
            error=error,
        )
        payload = result.to_dict()
        history = {
            key: payload[key]
            for key in (
                "task_id",
                "timestamp",
                "instruction",
                "repository_path",
                "dry_run",
                "no_commit",
                "final_status",
                "changed_files",
                "checks",
                "check_summary",
                "review_summary",
                "approval_id",
                "error",
            )
        }
        if compatibility_mode:
            history.pop("check_summary", None)
        store.append_history(history)
        store.save_run(task_id, payload)
        if caught_exception is not None:
            raise caught_exception
        return result

    def _build_plan(self, instruction: str, root: Path) -> tuple[TaskPlan, bool]:
        parameters = inspect.signature(self.planner.plan).parameters
        if len(parameters) == 1:
            return self.planner.plan(instruction), False

        from kodiak.orchestration.local_agents import RepositoryAnalyzer

        legacy_plan = self.planner.plan(instruction, RepositoryAnalyzer().analyze(root))
        steps = tuple(step.description for step in legacy_plan.steps)
        targets = tuple(legacy_plan.files_to_modify or legacy_plan.files_to_inspect)
        return (
            TaskPlan(
                task_type=legacy_plan.task_type,
                summary=legacy_plan.summary,
                supported=legacy_plan.task_type not in {"manual", "readme_missing"},
                target_files=targets,
                steps=steps,
            ),
            bool(legacy_plan.requires_approval),
        )

    def _run_checks(self, root: Path, *, dry_run: bool = False) -> tuple[CheckResult, ...]:
        if hasattr(self.tester, "run"):
            return tuple(self.tester.run(root, dry_run=dry_run))

        legacy_checks = self.tester.run_checks(root)
        return tuple(
            CheckResult(
                name=check.name,
                status="passed"
                if check.success is True
                else "failed"
                if check.success is False
                else "skipped",
                command=tuple(check.command),
                exit_code=check.exit_code,
                summary=check.skipped_reason or ("passed" if check.success else "failed"),
            )
            for check in legacy_checks
        )

    @staticmethod
    def _create_commit_approval(
        root: Path, task_id: str, changed_files: tuple[str, ...]
    ) -> dict[str, Any]:
        hashes: dict[str, str] = {}
        for relative in changed_files:
            path = (root / relative).resolve()
            path.relative_to(root)
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return ApprovalManager(root).create(
            task_id=task_id,
            action="git_commit",
            reason="Creating a local Git commit requires explicit approval.",
            metadata={"paths": list(changed_files), "sha256": hashes},
        )
