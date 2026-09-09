"""Typed data exchanged by Kodiak's local v1 workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RepositoryContext:
    path: str
    language: str
    framework: str | None
    source_dirs: list[str]
    test_dirs: list[str]
    package_files: list[str]
    important_files: list[str]
    git_branch: str | None
    has_uncommitted_changes: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskStep:
    id: str
    description: str
    agent_role: str
    status: str = "pending"


@dataclass(slots=True)
class TaskPlan:
    task_id: str
    instruction: str
    repository_path: str
    summary: str
    steps: list[TaskStep]
    files_to_inspect: list[str]
    files_to_modify: list[str]
    risk_level: str
    requires_approval: bool
    expected_checks: list[str]
    task_type: str = "manual"


@dataclass(slots=True)
class FileChange:
    path: str
    change_type: str
    summary: str
    before_hash: str | None
    after_hash: str | None


@dataclass(slots=True)
class CheckResult:
    name: str
    command: list[str]
    success: bool | None
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    skipped_reason: str | None = None
    skipped: bool = False


@dataclass(slots=True)
class ReviewReport:
    task_id: str
    completed: bool
    summary: str
    changed_files: list[str]
    checks: list[CheckResult]
    risks: list[str]
    recommendations: list[str]
    approval_required: bool
    git_diff_stat: str = ""
    git_changed_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan_summary: str = ""
    approval_status: str | None = None


@dataclass(slots=True)
class TaskRunResult:
    task_id: str
    instruction: str
    plan: TaskPlan
    changes: list[FileChange]
    checks: list[CheckResult]
    review: ReviewReport
    final_status: str
    repository: RepositoryContext
    approval_id: str | None = None
    dry_run: bool = False
    no_commit: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ApprovalRequest:
    approval_id: str
    task_id: str
    action: str
    reason: str
    risk_level: str
    created_at: str
    updated_at: str
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GitDiffSummary:
    repository_path: str
    is_git_repo: bool
    branch: str | None
    dirty: bool
    changed_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    unstaged_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    diff_stat: str = ""
    warning: str | None = None
