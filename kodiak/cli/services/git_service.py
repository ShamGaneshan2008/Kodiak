"""Read-only Git inspection for the local v1 CLI."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GitDiffSummary:
    repository_path: str
    is_git_repo: bool
    current_branch: str | None
    dirty: bool
    staged_files: tuple[str, ...] = ()
    unstaged_files: tuple[str, ...] = ()
    untracked_files: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    diff_stat: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("staged_files", "unstaged_files", "untracked_files", "changed_files"):
            payload[key] = list(payload[key])
        return payload


class GitService:
    """Inspect repository state using a fixed set of non-destructive Git commands."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    def diff_summary(self, repository_path: str | Path) -> GitDiffSummary:
        root = Path(repository_path).expanduser().resolve()
        if not root.is_dir():
            return GitDiffSummary(
                str(root), False, None, False, error=f"Path is not a directory: {root}"
            )
        if shutil.which("git") is None:
            return GitDiffSummary(str(root), False, None, False, error="Git is not available.")

        probe = self._run(["git", "rev-parse", "--is-inside-work-tree"], root)
        if probe.returncode != 0 or probe.stdout.strip().casefold() != "true":
            return GitDiffSummary(str(root), False, None, False, error="Not a Git repository.")

        branch_result = self._run(["git", "branch", "--show-current"], root)
        branch = branch_result.stdout.strip() or "(detached HEAD)"
        status_result = self._run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], root
        )
        if status_result.returncode != 0:
            return GitDiffSummary(
                str(root),
                True,
                branch,
                False,
                error=self._error_detail(status_result, "git status failed"),
            )
        staged, unstaged, untracked = self._parse_status(status_result.stdout)
        diff = self._run(["git", "diff", "--stat", "HEAD"], root)
        if diff.returncode != 0:
            diff = self._run(["git", "diff", "--stat"], root)
        changed = tuple(sorted(set(staged) | set(unstaged) | set(untracked)))
        return GitDiffSummary(
            repository_path=str(root),
            is_git_repo=True,
            current_branch=branch,
            dirty=bool(changed),
            staged_files=tuple(staged),
            unstaged_files=tuple(unstaged),
            untracked_files=tuple(untracked),
            changed_files=changed,
            diff_stat=diff.stdout.strip() if diff.returncode == 0 else "",
        )

    def _run(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=cwd,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(command, 1, "", str(exc))

    @staticmethod
    def _parse_status(output: str) -> tuple[list[str], list[str], list[str]]:
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        for line in output.splitlines():
            if len(line) < 3:
                continue
            code = line[:2]
            path = line[3:].split(" -> ")[-1].strip('"').replace("\\", "/")
            if code == "??":
                untracked.append(path)
                continue
            if code[0] not in {" ", "?"}:
                staged.append(path)
            if code[1] not in {" ", "?"}:
                unstaged.append(path)
        return sorted(staged), sorted(unstaged), sorted(untracked)

    @staticmethod
    def _error_detail(result: subprocess.CompletedProcess[str], fallback: str) -> str:
        return (result.stderr or result.stdout).strip() or fallback
