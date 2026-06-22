from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GitFileChange:
    path: str
    status: str
    additions: int = 0
    deletions: int = 0

    @property
    def area(self) -> str:
        parts = Path(self.path).parts
        if not parts:
            return "repo"
        if parts[0] == "kodiak" and len(parts) > 1:
            return parts[1]
        return parts[0].lstrip(".") or "repo"


@dataclass(frozen=True, slots=True)
class GitChangeSet:
    root: Path
    branch: str
    files: tuple[GitFileChange, ...] = field(default_factory=tuple)

    @property
    def changed_paths(self) -> list[str]:
        return [change.path for change in self.files]

    @property
    def total_additions(self) -> int:
        return sum(change.additions for change in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(change.deletions for change in self.files)

    @property
    def areas(self) -> list[str]:
        return sorted({change.area for change in self.files})

    @property
    def is_empty(self) -> bool:
        return not self.files


def run_git(args: list[str], cwd: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(cwd),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def current_branch(cwd: str | Path = ".") -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "HEAD"


def repo_root(cwd: str | Path = ".") -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], cwd))


def read_changes(cwd: str | Path = ".") -> GitChangeSet:
    root = repo_root(cwd)
    branch = current_branch(root)
    status_by_path = _parse_status(run_git(["status", "--porcelain=v1"], root))
    stats_by_path = _parse_numstat(run_git(["diff", "--numstat", "HEAD"], root))
    paths = sorted(set(status_by_path) | set(stats_by_path))
    files = tuple(
        GitFileChange(
            path=path,
            status=status_by_path.get(path, "M"),
            additions=stats_by_path.get(path, (0, 0))[0],
            deletions=stats_by_path.get(path, (0, 0))[1],
        )
        for path in paths
    )
    return GitChangeSet(root=root, branch=branch, files=files)


def make_branch_name(title: str, prefix: str = "codex") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:64].strip("-")
    return f"{prefix}/{slug or 'work'}"


def stage_paths(paths: list[str], cwd: str | Path = ".") -> None:
    if paths:
        run_git(["add", "--", *paths], cwd)


def commit(message: str, cwd: str | Path = ".") -> str:
    run_git(["commit", "-m", message], cwd)
    return run_git(["rev-parse", "HEAD"], cwd)


def _parse_status(raw: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in raw.splitlines():
        if not line:
            continue
        status = line[:2].strip() or "M"
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        statuses[path] = status
    return statuses


def _parse_numstat(raw: str) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        additions = 0 if parts[0] == "-" else int(parts[0])
        deletions = 0 if parts[1] == "-" else int(parts[1])
        path = parts[2]
        if " => " in path:
            path = path.split(" => ", 1)[1].strip("{}")
        stats[path] = (additions, deletions)
    return stats
