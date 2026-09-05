"""Safe branch-to-PR lifecycle wrapped around Kodiak's autonomous task loop."""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import structlog

from kodiak.db.models.task import Task, TaskPriority, TaskStatus
from kodiak.github.pr_manager import create_or_update_pull_request
from kodiak.orchestration.approval_gate import ApprovalGate, ApprovalStatus
from kodiak.orchestration.autonomous_loop import AutonomousLoopResult, AutonomousTaskLoop
from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.orchestration.reflection import ReflectionEngine, RepairStrategy
from kodiak.orchestration.verification import VerificationResult, VerificationStatus
from kodiak.security.secrets import SecretManager
from kodiak.utils.git_utils import (
    GitDiffReview,
    GitOperationError,
    GitRepositoryState,
    commit,
    ensure_work_branch,
    inspect_repository,
    make_branch_name,
    push_branch,
    repo_root,
    review_diff,
    run_git,
    stage_paths,
)

logger = structlog.get_logger(__name__)


class GitWorkflowStatus(enum.StrEnum):
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMMITTED = "committed"
    PUSHED = "pushed"
    WAITING_FOR_CI = "waiting_for_ci"
    READY_FOR_REVIEW = "ready_for_review"


class CIStatus(enum.StrEnum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    CANCELLED = "cancelled"
    ERROR = "error"


class GitHubWorkflowClient(Protocol):
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 30,
        head: str | None = None,
        base: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, body: str = ""
    ) -> dict[str, Any]: ...

    async def update_pull_request(
        self, owner: str, repo: str, pr_number: int, **kwargs: Any
    ) -> dict[str, Any]: ...

    async def list_check_runs(self, owner: str, repo: str, ref: str) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class GitWorkflowRequest:
    task_id: str
    title: str
    goal: str
    repository: Path
    intended_paths: tuple[str, ...] = ()
    default_branch: str = "main"
    remote: str = "origin"
    github_owner: str | None = None
    github_repo: str | None = None
    issue_number: int | None = None
    publish_remote: bool = False


@dataclass(slots=True)
class CIResult:
    status: CIStatus
    checks: tuple[dict[str, Any], ...] = ()

    @property
    def failure_evidence(self) -> dict[str, Any]:
        failed = [
            {
                "name": check.get("name"),
                "status": check.get("status"),
                "conclusion": check.get("conclusion"),
                "details_url": check.get("details_url"),
                "summary": (check.get("output") or {}).get("summary"),
            }
            for check in self.checks
            if _normalize_check(check) in {CIStatus.FAIL, CIStatus.CANCELLED, CIStatus.ERROR}
        ]
        return {"ci_status": self.status.value, "failed_checks": failed}


@dataclass(slots=True)
class GitWorkflowResult:
    status: GitWorkflowStatus
    task_id: str
    branch: str | None = None
    commit_sha: str | None = None
    changed_files: tuple[str, ...] = ()
    pr_number: int | None = None
    pr_url: str | None = None
    ci_status: CIStatus | None = None
    repair_attempts: int = 0
    verification: dict[str, Any] | None = None
    diff_review: GitDiffReview | None = None
    error: str | None = None
    audit: list[dict[str, Any]] = field(default_factory=list)


PushOperation = Callable[[str, Path, str], None]
SleepOperation = Callable[[float], Awaitable[None]]


class AutonomousGitWorkflow:
    """Coordinate safe local Git and bounded GitHub lifecycle side effects."""

    def __init__(
        self,
        autonomous_loop: AutonomousTaskLoop,
        *,
        approval_gate: ApprovalGate | None = None,
        github_client: GitHubWorkflowClient | None = None,
        reflection: ReflectionEngine | None = None,
        secret_manager: SecretManager | None = None,
        max_ci_polls: int = 6,
        ci_poll_interval_seconds: float = 30.0,
        max_ci_repairs: int = 2,
        push_operation: PushOperation = push_branch,
        sleep: SleepOperation = asyncio.sleep,
    ) -> None:
        self._autonomous_loop = autonomous_loop
        self._approval = approval_gate or ApprovalGate()
        self._github = github_client
        self._reflection = reflection or ReflectionEngine()
        self._secrets = secret_manager or SecretManager()
        self._max_ci_polls = max(1, max_ci_polls)
        self._poll_interval = max(0.0, ci_poll_interval_seconds)
        self._max_ci_repairs = max(0, max_ci_repairs)
        self._push = push_operation
        self._sleep = sleep
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self._autonomous_loop.cancel()

    async def run(self, request: GitWorkflowRequest) -> GitWorkflowResult:
        result = GitWorkflowResult(status=GitWorkflowStatus.FAILED, task_id=request.task_id)
        try:
            initial = inspect_repository(request.repository)
            self._record(result, "git.repository.inspected", branch=initial.branch)
            if initial.staged:
                raise GitOperationError(
                    "Refusing autonomous commit while pre-existing staged changes are present: "
                    f"{list(initial.staged)}"
                )
            branch = make_branch_name(f"{request.task_id}-{request.title}")
            ensure_work_branch(
                branch,
                initial.root,
                protected_branches=(request.default_branch,),
            )
            result.branch = branch
            self._record(result, "git.branch.created", branch=branch)

            engineering = await self._run_engineering(request, result, repair_evidence=None)
            if engineering is None:
                return result

            await self._commit_verified_changes(request, result, initial, engineering)
            if not request.publish_remote:
                result.status = GitWorkflowStatus.COMMITTED
                return result

            self._check_cancelled()
            await self._publish(request, result)
            ci = await self._wait_for_ci(request, result)

            while ci.status is CIStatus.FAIL and result.repair_attempts < self._max_ci_repairs:
                self._check_cancelled()
                result.repair_attempts += 1
                reflection = await self._reflect_ci_failure(request, result, ci)
                self._record(
                    result,
                    "github.repair.started",
                    attempt=result.repair_attempts,
                    reflection_action=reflection.strategy.value,
                )
                if reflection.strategy is RepairStrategy.STOP:
                    result.status = GitWorkflowStatus.FAILED
                    result.error = reflection.root_cause
                    return result

                before_repair = inspect_repository(request.repository)
                engineering = await self._run_engineering(
                    request,
                    result,
                    repair_evidence=ci.failure_evidence,
                )
                if engineering is None:
                    return result
                await self._commit_verified_changes(request, result, before_repair, engineering)
                await self._push_approved(request, result)
                ci = await self._wait_for_ci(request, result)

            result.ci_status = ci.status
            if ci.status is CIStatus.PASS:
                result.status = GitWorkflowStatus.READY_FOR_REVIEW
            elif ci.status is CIStatus.FAIL:
                result.status = GitWorkflowStatus.FAILED
                result.error = "CI repair budget exhausted; human attention required."
            else:
                result.status = GitWorkflowStatus.FAILED
                result.error = f"CI ended with status {ci.status.value}."
            return result
        except _WorkflowCancelled:
            result.status = GitWorkflowStatus.CANCELLED
            result.error = "Workflow cancelled before further side effects."
            return result
        except (GitOperationError, RuntimeError) as exc:
            result.error = str(exc)
            return result

    async def _run_engineering(
        self,
        request: GitWorkflowRequest,
        result: GitWorkflowResult,
        repair_evidence: dict[str, Any] | None,
    ) -> AutonomousLoopResult | None:
        self._check_cancelled()
        output = await self._autonomous_loop.run(
            request.goal,
            workspace=request.repository,
            title=request.title,
            extra_context={
                "git_task_id": request.task_id,
                "issue_number": request.issue_number,
                "expected_files": list(request.intended_paths),
                "ci_failure_evidence": repair_evidence,
            },
        )
        result.verification = (
            output.verification_result.to_dict() if output.verification_result else None
        )
        if not output.success:
            result.error = output.task_state.error or "Autonomous engineering did not verify."
            return None
        return output

    async def _commit_verified_changes(
        self,
        request: GitWorkflowRequest,
        result: GitWorkflowResult,
        baseline: GitRepositoryState,
        engineering: AutonomousLoopResult,
    ) -> None:
        self._check_cancelled()
        current = inspect_repository(request.repository)
        newly_changed = current.changed_paths - baseline.changed_paths
        intended = set(request.intended_paths) if request.intended_paths else newly_changed
        if not intended:
            raise GitOperationError(
                "Verified execution produced no new changes; no commit created."
            )
        if intended & baseline.changed_paths:
            raise GitOperationError(
                "Refusing to commit a path that contained pre-existing user work."
            )
        review = review_diff(
            intended,
            request.repository,
            allowed_changed_paths=baseline.changed_paths,
        )
        result.diff_review = review
        self._record(result, "git.diff.reviewed", changed_files=list(review.intended_paths))
        if not review.approved:
            raise GitOperationError(
                "Diff review rejected changes: "
                f"unexpected={list(review.unexpected_paths)}, "
                f"sensitive={list(review.sensitive_paths)}"
            )

        secret_scan_text = run_git(
            ["diff", "--", *review.intended_paths], request.repository
        ) + _read_text_files(request.repository, review.intended_paths)
        if not await self._secrets.validate_secret(secret_scan_text):
            raise GitOperationError("Diff contains secret-like content; commit blocked.")
        stage_paths(list(review.intended_paths), request.repository)
        subject = _commit_subject(request.title)
        result.commit_sha = commit(subject, request.repository)
        result.changed_files = review.intended_paths
        self._record(
            result,
            "git.commit.created",
            commit_sha=result.commit_sha,
            changed_files=list(result.changed_files),
            verification=result.verification,
            agents=[engineering.selected_agent] if engineering.selected_agent else [],
        )

    async def _publish(self, request: GitWorkflowRequest, result: GitWorkflowResult) -> None:
        if self._github is None or not request.github_owner or not request.github_repo:
            raise RuntimeError("GitHub client and repository identity are required for publishing.")
        await self._push_approved(request, result)
        result.status = GitWorkflowStatus.PUSHED
        self._check_cancelled()
        approval = await self._approval.request_approval(
            "create_pr",
            {
                "repository": f"{request.github_owner}/{request.github_repo}",
                "branch": result.branch or "",
            },
        )
        if approval.status is not ApprovalStatus.APPROVED:
            raise RuntimeError("Pull request creation was not approved.")

        body = _pull_request_body(request, result)
        pr, created = await create_or_update_pull_request(
            self._github,
            owner=request.github_owner,
            repo=request.github_repo,
            title=request.title,
            body=body,
            head=result.branch or "",
            base=request.default_branch,
        )
        event = "github.pr.created" if created else "github.pr.updated"
        result.pr_number = int(pr["number"])
        result.pr_url = pr.get("html_url")
        self._record(result, event, pr_number=result.pr_number)

    async def _push_approved(self, request: GitWorkflowRequest, result: GitWorkflowResult) -> None:
        self._check_cancelled()
        approval = await self._approval.request_approval(
            "git_push",
            {"branch": result.branch or "", "remote": request.remote},
        )
        if approval.status is not ApprovalStatus.APPROVED:
            raise RuntimeError("Git push was not approved.")
        self._push(result.branch or "", repo_root(request.repository), request.remote)
        self._record(result, "git.push.completed", commit_sha=result.commit_sha)

    async def _wait_for_ci(
        self, request: GitWorkflowRequest, result: GitWorkflowResult
    ) -> CIResult:
        assert self._github is not None
        result.status = GitWorkflowStatus.WAITING_FOR_CI
        for poll in range(1, self._max_ci_polls + 1):
            self._check_cancelled()
            checks = tuple(
                await self._github.list_check_runs(
                    request.github_owner or "", request.github_repo or "", result.branch or ""
                )
            )
            ci = _aggregate_ci(checks)
            result.ci_status = ci.status
            self._record(result, f"github.ci.{ci.status.value}", poll=poll)
            if ci.status is not CIStatus.PENDING:
                return ci
            if poll < self._max_ci_polls:
                await self._sleep(self._poll_interval)
        return CIResult(CIStatus.ERROR, checks)

    async def _reflect_ci_failure(
        self, request: GitWorkflowRequest, result: GitWorkflowResult, ci: CIResult
    ) -> Any:
        evidence = ci.failure_evidence
        verification = VerificationResult(
            status=VerificationStatus.FAILED,
            message="Remote CI failed.",
            evidence={
                "evidence": [
                    {
                        "verifier": "ci",
                        "status": "failed",
                        "message": "Remote CI failed.",
                        "metadata": evidence,
                    }
                ]
            },
            retry_recommended=True,
        )
        execution = ExecutionResult(
            task_id=request.task_id,
            outcome=ExecutionOutcome.FAILURE,
            attempts=result.repair_attempts,
            duration_seconds=0,
            error={"type": "CIFailure", "message": "Remote CI failed", **evidence},
            final_status=TaskStatus.FAILED,
            verification=verification.to_dict(),
        )
        task = Task(
            id=str(uuid.uuid4()),
            repository_id=str(uuid.uuid4()),
            title=request.title,
            description=request.goal,
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.MEDIUM,
            max_retries=self._max_ci_repairs,
            context={"ci": evidence, "branch": result.branch},
        )
        return await self._reflection.reflect(
            task,
            execution,
            verification_result=verification,
            attempt=result.repair_attempts,
            max_attempts=self._max_ci_repairs + 1,
        )

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise _WorkflowCancelled

    @staticmethod
    def _record(result: GitWorkflowResult, event: str, **data: Any) -> None:
        result.audit.append({"event": event, "timestamp": time.time(), **data})
        logger.info(event.replace(".", "_"), task_id=result.task_id, **data)


class _WorkflowCancelled(Exception):
    pass


def _normalize_check(check: dict[str, Any]) -> CIStatus:
    if check.get("status") != "completed":
        return CIStatus.PENDING
    conclusion = str(check.get("conclusion") or "").lower()
    if conclusion in {"success", "neutral", "skipped"}:
        return CIStatus.PASS
    if conclusion in {"cancelled", "stale"}:
        return CIStatus.CANCELLED
    if conclusion in {"failure", "timed_out", "action_required"}:
        return CIStatus.FAIL
    return CIStatus.ERROR


def _aggregate_ci(checks: tuple[dict[str, Any], ...]) -> CIResult:
    if not checks:
        return CIResult(CIStatus.PENDING, checks)
    statuses = {_normalize_check(check) for check in checks}
    for status in (CIStatus.FAIL, CIStatus.ERROR, CIStatus.CANCELLED, CIStatus.PENDING):
        if status in statuses:
            return CIResult(status, checks)
    return CIResult(CIStatus.PASS, checks)


def _commit_subject(title: str) -> str:
    clean = " ".join(title.strip().split())
    if clean.lower().startswith(("fix:", "feat:", "test:", "docs:", "chore:", "refactor:")):
        return clean[:100]
    return f"fix: {clean[:93]}"


def _pull_request_body(request: GitWorkflowRequest, result: GitWorkflowResult) -> str:
    issue = f"\nIssue: #{request.issue_number}" if request.issue_number else ""
    files = "\n".join(f"- `{path}`" for path in result.changed_files)
    verification = (result.verification or {}).get("summary", "Verified by Kodiak")
    return (
        f"## Summary\n\n{request.goal}\n\n"
        f"## Verification\n\n- {verification}\n\n"
        f"## Files\n\n{files or '- No files recorded'}\n\n"
        "## Risks\n\n- Awaiting human review; no automatic merge was performed."
        f"{issue}"
    )


def _read_text_files(repository: Path, paths: tuple[str, ...]) -> str:
    contents: list[str] = []
    for path in paths:
        candidate = repository / path
        if not candidate.is_file() or candidate.stat().st_size > 1_000_000:
            continue
        try:
            contents.append(candidate.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(contents)


__all__ = [
    "AutonomousGitWorkflow",
    "CIResult",
    "CIStatus",
    "GitWorkflowRequest",
    "GitWorkflowResult",
    "GitWorkflowStatus",
]
