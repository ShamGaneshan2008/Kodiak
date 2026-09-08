"""Deterministic, offline-capable Kodiak v1 task workflow."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from kodiak.cli.services.git_service import GitDiffSummary, GitService
from kodiak.orchestration.approval_manager import ApprovalManager
from kodiak.orchestration.local_storage import LocalStateStore

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
        return self.final_status in {"completed", "completed_with_warnings", "dry_run"}

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "review_summary": self.review_summary,
            "approval_id": self.approval_id,
            "error": self.error,
        }


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
        return ()

    def can_handle(self, root: Path, plan: TaskPlan) -> bool:
        """Return whether a no-op means the requested state is already satisfied."""
        if plan.task_type == "readme_quickstart":
            return self._find_readme(root) is not None
        if plan.task_type == "health_check_test":
            if self._verified_fastapi_module(root) is None:
                return False
            marker = "# Generated by Kodiak v1: health-check-test"
            primary = root / "tests" / "test_health.py"
            fallback = root / "tests" / "test_kodiak_health.py"
            primary_available = not primary.exists() or marker in primary.read_text(
                encoding="utf-8"
            )
            fallback_available = not fallback.exists() or marker in fallback.read_text(
                encoding="utf-8"
            )
            return primary_available or fallback_available
        return False

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
        if module is None:
            return ()
        path = root / "tests" / "test_health.py"
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
        planner: DeterministicPlanner | None = None,
        editor: SafeEditor | None = None,
        tester: LocalTester | None = None,
        reviewer: DiffReviewer | None = None,
        git_service: GitService | None = None,
    ) -> None:
        self.git_service = git_service or GitService()
        self.inspector = inspector or RepositoryInspector(self.git_service)
        self.planner = planner or DeterministicPlanner()
        self.editor = editor or SafeEditor()
        self.tester = tester or LocalTester()
        self.reviewer = reviewer or DiffReviewer()

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

        try:
            analysis = self.inspector.analyze(root)
            plan = self.planner.plan(instruction)
            if not plan.supported:
                status = "manual_required"
                review = plan.summary
            else:
                proposed = self.editor.propose(root, plan)
                if not proposed and not self.editor.can_handle(root, plan):
                    status = "manual_required"
                    review = (
                        "The required target could not be verified safely; no file was changed. "
                        "For health tests, confirm an importable FastAPI app and /health route."
                    )
                elif dry_run:
                    checks = self.tester.run(root, dry_run=True)
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
                        checks = self.tester.run(root)
                        git = self.git_service.diff_summary(root)
                        review = self.reviewer.review(proposed, checks, git)
                        if any(check.status in {"failed", "error"} for check in checks):
                            status = "failed_checks"
                        elif any(check.status in {"missing", "skipped"} for check in checks):
                            status = "completed_with_warnings"
                        else:
                            status = "completed"

                        if (
                            changed_files
                            and not no_commit
                            and status.startswith("completed")
                            and git.is_git_repo
                        ):
                            request = self._create_commit_approval(root, task_id, changed_files)
                            approval_id = str(request["approval_id"])
                            status = "awaiting_approval"
                            review += (
                                " A local commit requires explicit approval; no commit was made."
                            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "error"
            review = "The workflow stopped after an error; completion was not reported."

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
                "review_summary",
                "approval_id",
                "error",
            )
        }
        store.append_history(history)
        store.save_run(task_id, payload)
        return result

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
