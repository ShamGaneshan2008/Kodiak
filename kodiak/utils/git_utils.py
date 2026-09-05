from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
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


@dataclass(frozen=True, slots=True)
class GitRepositoryState:
    root: Path
    branch: str
    detached: bool
    staged: tuple[str, ...]
    unstaged: tuple[str, ...]
    untracked: tuple[str, ...]
    remotes: dict[str, str]
    tracking_branch: str | None

    @property
    def is_clean(self) -> bool:
        return not (self.staged or self.unstaged or self.untracked)

    @property
    def changed_paths(self) -> set[str]:
        return set(self.staged) | set(self.unstaged) | set(self.untracked)


@dataclass(frozen=True, slots=True)
class GitDiffReview:
    intended_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...] = ()
    sensitive_paths: tuple[str, ...] = ()
    binary_paths: tuple[str, ...] = ()

    @property
    def approved(self) -> bool:
        return not (self.unexpected_paths or self.sensitive_paths)


class GitOperationError(RuntimeError):
    """Actionable failure raised by the canonical local Git boundary."""


def summarize_changes(changes: GitChangeSet) -> list[dict[str, object]]:
    """Serialize a Git change set for agent and pull-request output."""
    return [
        {
            "path": change.path,
            "status": change.status,
            "additions": change.additions,
            "deletions": change.deletions,
            "area": change.area,
        }
        for change in changes.files
    ]


def human_summary(changes: GitChangeSet) -> str:
    """Return a concise, deterministic summary of repository changes."""
    count = len(changes.files)
    noun = "file" if count == 1 else "files"
    return (
        f"{count} {noun} changed, "
        f"{changes.total_additions} insertions, {changes.total_deletions} deletions."
    )


def run_git(args: list[str], cwd: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(cwd),
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        stderr = getattr(exc, "stderr", None)
        detail = str(stderr or exc).strip()
        raise GitOperationError(f"git {args[0]} failed: {detail}") from exc
    # Leading spaces are significant in porcelain status output (index/worktree columns).
    return result.stdout.rstrip()


def current_branch(cwd: str | Path = ".") -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "HEAD"


def repo_root(cwd: str | Path = ".") -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], cwd))


def inspect_repository(cwd: str | Path = ".") -> GitRepositoryState:
    root = repo_root(cwd).resolve()
    branch = current_branch(root)
    status = run_git(["status", "--porcelain=v1"], root)
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    for line in status.splitlines():
        code, path = line[:2], _status_path(line[3:])
        if code == "??":
            untracked.append(path)
            continue
        if code[0] not in {" ", "?"}:
            staged.append(path)
        if code[1] not in {" ", "?"}:
            unstaged.append(path)

    remotes: dict[str, str] = {}
    remote_output = run_git(["remote", "-v"], root)
    for line in remote_output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "(push)":
            remotes[parts[0]] = parts[1]
    tracking = (
        run_git(["rev-parse", "--abbrev-ref", "@{upstream}"], root) if _has_upstream(root) else None
    )
    return GitRepositoryState(
        root=root,
        branch=branch,
        detached=branch == "HEAD",
        staged=tuple(sorted(staged)),
        unstaged=tuple(sorted(unstaged)),
        untracked=tuple(sorted(untracked)),
        remotes=remotes,
        tracking_branch=tracking,
    )


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


def make_branch_name(title: str, prefix: str = "kodiak/task") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:64].strip("-")
    return f"{prefix}/{slug or 'work'}"


def branch_exists(branch: str, cwd: str | Path = ".") -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=Path(cwd),
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def ensure_work_branch(
    branch: str,
    cwd: str | Path = ".",
    *,
    protected_branches: Iterable[str] = ("main", "master"),
) -> str:
    state = inspect_repository(cwd)
    if state.detached:
        raise GitOperationError("Cannot create autonomous work from detached HEAD.")
    if state.branch == branch:
        return branch
    if state.branch not in set(protected_branches):
        raise GitOperationError(
            f"Refusing to switch from existing non-default branch {state.branch!r}."
        )
    if branch_exists(branch, state.root):
        run_git(["switch", branch], state.root)
    else:
        run_git(["switch", "-c", branch], state.root)
    return branch


def review_diff(
    intended_paths: Iterable[str],
    cwd: str | Path = ".",
    *,
    allowed_changed_paths: Iterable[str] | None = None,
) -> GitDiffReview:
    root = repo_root(cwd)
    intended = tuple(sorted({_validate_relative_path(path, root) for path in intended_paths}))
    actual = inspect_repository(root).changed_paths
    allowed = set(allowed_changed_paths or ()) | set(intended)
    sensitive = tuple(path for path in intended if _is_sensitive_path(path))
    binary = tuple(path for path in intended if _is_binary_diff(path, root))
    return GitDiffReview(
        intended_paths=intended,
        unexpected_paths=tuple(sorted(actual - allowed)),
        sensitive_paths=sensitive,
        binary_paths=binary,
    )


def stage_paths(paths: list[str], cwd: str | Path = ".") -> None:
    if paths:
        root = repo_root(cwd)
        safe_paths = [_validate_relative_path(path, root) for path in paths]
        if any(_is_sensitive_path(path) for path in safe_paths):
            raise GitOperationError("Refusing to stage a sensitive file.")
        run_git(["add", "--", *safe_paths], root)


def commit(message: str, cwd: str | Path = ".") -> str:
    if not run_git(["diff", "--cached", "--name-only"], cwd):
        raise GitOperationError("Refusing to create an empty commit.")
    run_git(["commit", "-m", message], cwd)
    return run_git(["rev-parse", "HEAD"], cwd)


def push_branch(branch: str, cwd: str | Path = ".", remote: str = "origin") -> None:
    state = inspect_repository(cwd)
    if branch in {"main", "master"}:
        raise GitOperationError("Refusing to push directly to a protected branch.")
    if state.branch != branch:
        raise GitOperationError(f"Current branch {state.branch!r} is not {branch!r}.")
    if remote not in state.remotes:
        raise GitOperationError(f"Git remote {remote!r} is not configured.")
    run_git(["push", "--set-upstream", remote, branch], state.root)


def _parse_status(raw: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in raw.splitlines():
        if not line:
            continue
        status = line[:2].strip() or "M"
        path = _status_path(line[3:])
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


def _status_path(path: str) -> str:
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"').replace("\\", "/")


def _has_upstream(cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "@{upstream}"],
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _validate_relative_path(path: str, root: Path) -> str:
    candidate = (root / path).resolve()
    try:
        relative = candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise GitOperationError(f"Path is outside repository: {path}") from exc
    return relative.as_posix()


def _is_sensitive_path(path: str) -> bool:
    lowered = path.lower()
    name = Path(lowered).name
    return (
        name == ".env"
        or name.startswith(".env.")
        or "credential" in name
        or name in {"id_rsa", "id_ed25519"}
        or lowered.endswith((".pem", ".key", ".p12", ".pfx"))
    )


def _is_binary_diff(path: str, cwd: Path) -> bool:
    output = run_git(["diff", "--numstat", "HEAD", "--", path], cwd)
    return any(line.startswith("-\t-") for line in output.splitlines())
