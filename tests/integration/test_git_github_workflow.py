from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kodiak.orchestration.approval_gate import ApprovalRequest, ApprovalStatus
from kodiak.orchestration.autonomous_loop import AutonomousLoopResult
from kodiak.orchestration.git_workflow import (
    AutonomousGitWorkflow,
    CIStatus,
    GitWorkflowRequest,
    GitWorkflowStatus,
)
from kodiak.orchestration.state import TaskState, TaskStatus
from kodiak.orchestration.verification import VerificationResult, VerificationStatus
from kodiak.utils.git_utils import (
    GitOperationError,
    ensure_work_branch,
    inspect_repository,
    make_branch_name,
    push_branch,
    run_git,
)


class ScriptedLoop:
    def __init__(self, repo: Path, contents: list[str], *, verified: bool = True) -> None:
        self.repo = repo
        self.contents = contents
        self.verified = verified
        self.calls = 0
        self.cancelled = False
        self.on_run = None

    async def run(self, goal: str, **kwargs: Any) -> AutonomousLoopResult:
        index = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        (self.repo / "app.py").write_text(self.contents[index], encoding="utf-8")
        if self.on_run:
            self.on_run(self.calls)
        state = TaskState(title=goal, objective=goal)
        state.status = TaskStatus.COMPLETED if self.verified else TaskStatus.FAILED
        state.error = None if self.verified else "verification failed"
        verification = VerificationResult(
            status=VerificationStatus.VERIFIED if self.verified else VerificationStatus.FAILED,
            message="tests passed" if self.verified else "tests failed",
        )
        return AutonomousLoopResult(
            task_state=state,
            plan=None,
            execution_result=None,
            verification_result=verification,
            selected_agent="coder",
        )

    def cancel(self) -> None:
        self.cancelled = True


class DecidingApprovalGate:
    def __init__(self, approved: bool = True) -> None:
        self.approved = approved
        self.operations: list[str] = []

    async def request_approval(
        self, operation: str, details: dict[str, str] | None = None
    ) -> ApprovalRequest:
        self.operations.append(operation)
        return ApprovalRequest(
            operation=operation,
            details=details or {},
            status=ApprovalStatus.APPROVED if self.approved else ApprovalStatus.DENIED,
        )


class FakeGitHub:
    def __init__(
        self,
        ci_cycles: list[list[dict[str, Any]]] | None = None,
        existing_prs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ci_cycles = ci_cycles or [[_check("success")]]
        self.existing_prs = existing_prs or []
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.check_calls = 0

    async def list_pull_requests(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self.existing_prs

    async def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, body: str = ""
    ) -> dict[str, Any]:
        data = {
            "number": 7,
            "html_url": "https://example.test/pr/7",
            "state": "open",
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        }
        self.created.append(data)
        return data

    async def update_pull_request(
        self, owner: str, repo: str, pr_number: int, **kwargs: Any
    ) -> dict[str, Any]:
        data = {"number": pr_number, "html_url": f"https://example.test/pr/{pr_number}", **kwargs}
        self.updated.append(data)
        return data

    async def list_check_runs(self, owner: str, repo: str, ref: str) -> list[dict[str, Any]]:
        index = min(self.check_calls, len(self.ci_cycles) - 1)
        self.check_calls += 1
        return self.ci_cycles[index]


def _check(conclusion: str, name: str = "tests") -> dict[str, Any]:
    return {
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "details_url": "https://example.test/check",
        "output": {"summary": f"{name} {conclusion}"},
    }


def _repo(path: Path) -> Path:
    path.mkdir()
    run_git(["init", "-b", "main"], path)
    run_git(["config", "user.email", "kodiak@example.test"], path)
    run_git(["config", "user.name", "Kodiak Test"], path)
    (path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "user.txt").write_text("original\n", encoding="utf-8")
    run_git(["add", "--", "app.py", "user.txt"], path)
    run_git(["commit", "-m", "initial"], path)
    return path


def _request(
    repo: Path,
    *,
    publish: bool = False,
    paths: tuple[str, ...] = ("app.py",),
) -> GitWorkflowRequest:
    return GitWorkflowRequest(
        task_id="task-123",
        title="repair application value",
        goal="Repair the application value and verify it",
        repository=repo,
        intended_paths=paths,
        github_owner="owner",
        github_repo="repo",
        publish_remote=publish,
    )


def _workflow(
    loop: ScriptedLoop,
    *,
    github: FakeGitHub | None = None,
    approval: DecidingApprovalGate | None = None,
    pushes: list[tuple[str, str]] | None = None,
    max_repairs: int = 2,
) -> AutonomousGitWorkflow:
    push_log = pushes if pushes is not None else []
    return AutonomousGitWorkflow(
        loop,  # type: ignore[arg-type]
        github_client=github,
        approval_gate=approval or DecidingApprovalGate(),  # type: ignore[arg-type]
        max_ci_repairs=max_repairs,
        max_ci_polls=2,
        ci_poll_interval_seconds=0,
        push_operation=lambda branch, repo, remote: push_log.append((branch, remote)),
    )


def test_clean_repository_inspection(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    state = inspect_repository(repo)
    assert state.root == repo.resolve()
    assert state.branch == "main"
    assert state.is_clean


@pytest.mark.asyncio
async def test_dirty_user_work_is_preserved_and_selective_staging_commits_only_task(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "user.txt").write_text("user work\n", encoding="utf-8")
    workflow = _workflow(ScriptedLoop(repo, ["VALUE = 2\n"]))

    result = await workflow.run(_request(repo))

    assert result.status is GitWorkflowStatus.COMMITTED
    assert run_git(["show", "--pretty=", "--name-only", "HEAD"], repo) == "app.py"
    assert (repo / "user.txt").read_text(encoding="utf-8") == "user work\n"
    assert "user.txt" in inspect_repository(repo).unstaged


@pytest.mark.asyncio
async def test_preexisting_staged_user_work_blocks_commit_without_altering_it(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "user.txt").write_text("staged user work\n", encoding="utf-8")
    run_git(["add", "--", "user.txt"], repo)
    before = run_git(["diff", "--cached", "--", "user.txt"], repo)

    result = await _workflow(ScriptedLoop(repo, ["VALUE = 2\n"])).run(_request(repo))

    assert result.status is GitWorkflowStatus.FAILED
    assert "pre-existing staged changes" in (result.error or "")
    assert run_git(["diff", "--cached", "--", "user.txt"], repo) == before


@pytest.mark.asyncio
async def test_feature_branch_and_verified_commit_metadata(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    original_main = run_git(["rev-parse", "main"], repo)
    result = await _workflow(ScriptedLoop(repo, ["VALUE = 2\n"])).run(_request(repo))

    assert result.status is GitWorkflowStatus.COMMITTED
    assert result.branch == make_branch_name("task-123-repair application value")
    assert result.commit_sha == run_git(["rev-parse", "HEAD"], repo)
    assert result.changed_files == ("app.py",)
    assert run_git(["rev-parse", "main"], repo) == original_main
    assert any(event["event"] == "git.diff.reviewed" for event in result.audit)


@pytest.mark.asyncio
async def test_failed_verification_does_not_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    before = run_git(["rev-parse", "HEAD"], repo)
    result = await _workflow(ScriptedLoop(repo, ["VALUE = 0\n"], verified=False)).run(
        _request(repo)
    )
    assert result.status is GitWorkflowStatus.FAILED
    assert run_git(["rev-parse", "HEAD"], repo) == before


@pytest.mark.asyncio
async def test_sensitive_path_and_secret_content_are_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    loop = ScriptedLoop(repo, ["VALUE = 2\n"])
    request = _request(repo, paths=(".env",))
    loop.repo = repo
    original_run = loop.run

    async def write_env(*args: Any, **kwargs: Any) -> AutonomousLoopResult:
        output = await original_run(*args, **kwargs)
        (repo / ".env").write_text("GITHUB_TOKEN=ghp_" + "x" * 40, encoding="utf-8")
        return output

    loop.run = write_env  # type: ignore[method-assign]
    result = await _workflow(loop).run(request)
    assert result.status is GitWorkflowStatus.FAILED
    assert "sensitive" in (result.error or "").lower()
    assert not run_git(["diff", "--cached", "--name-only"], repo)


@pytest.mark.asyncio
async def test_secret_content_in_normal_file_is_blocked(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    token = "ghp_" + "x" * 40
    result = await _workflow(ScriptedLoop(repo, [f"TOKEN = '{token}'\n"])).run(_request(repo))
    assert result.status is GitWorkflowStatus.FAILED
    assert "secret-like content" in (result.error or "")
    assert not run_git(["diff", "--cached", "--name-only"], repo)


@pytest.mark.asyncio
async def test_denied_push_approval_blocks_remote_side_effect(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    pushes: list[tuple[str, str]] = []
    result = await _workflow(
        ScriptedLoop(repo, ["VALUE = 2\n"]),
        github=FakeGitHub(),
        approval=DecidingApprovalGate(False),
        pushes=pushes,
    ).run(_request(repo, publish=True))
    assert result.status is GitWorkflowStatus.FAILED
    assert pushes == []
    assert result.pr_number is None


@pytest.mark.asyncio
async def test_pr_creation_ci_pass_reaches_ready_for_review(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    github = FakeGitHub()
    pushes: list[tuple[str, str]] = []
    result = await _workflow(ScriptedLoop(repo, ["VALUE = 2\n"]), github=github, pushes=pushes).run(
        _request(repo, publish=True)
    )
    assert result.status is GitWorkflowStatus.READY_FOR_REVIEW
    assert result.ci_status is CIStatus.PASS
    assert result.pr_number == 7
    assert len(github.created) == 1
    assert pushes and "--force" not in str(pushes)
    assert "## Verification" in github.created[0]["body"]


@pytest.mark.asyncio
async def test_existing_open_pr_is_updated_not_duplicated(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    github = FakeGitHub(existing_prs=[{"number": 4, "state": "open"}])
    result = await _workflow(ScriptedLoop(repo, ["VALUE = 2\n"]), github=github).run(
        _request(repo, publish=True)
    )
    assert result.status is GitWorkflowStatus.READY_FOR_REVIEW
    assert github.created == []
    assert [item["number"] for item in github.updated] == [4]


@pytest.mark.asyncio
async def test_push_failure_remains_terminal_remote_failure(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")

    def reject_push(branch: str, root: Path, remote: str) -> None:
        raise GitOperationError("push rejected")

    workflow = AutonomousGitWorkflow(
        ScriptedLoop(repo, ["VALUE = 2\n"]),  # type: ignore[arg-type]
        github_client=FakeGitHub(),
        approval_gate=DecidingApprovalGate(),  # type: ignore[arg-type]
        push_operation=reject_push,
    )
    result = await workflow.run(_request(repo, publish=True))
    assert result.status is GitWorkflowStatus.FAILED
    assert result.commit_sha
    assert result.pr_number is None
    assert result.error == "push rejected"


@pytest.mark.asyncio
async def test_ci_failure_repairs_commits_pushes_and_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    github = FakeGitHub(ci_cycles=[[_check("failure")], [_check("success")]])
    pushes: list[tuple[str, str]] = []
    result = await _workflow(
        ScriptedLoop(repo, ["VALUE = 2\n", "VALUE = 3\n"]),
        github=github,
        pushes=pushes,
    ).run(_request(repo, publish=True))
    assert result.status is GitWorkflowStatus.READY_FOR_REVIEW
    assert result.repair_attempts == 1
    assert len(pushes) == 2
    assert run_git(["rev-list", "--count", "main..HEAD"], repo) == "2"
    repair_event = next(item for item in result.audit if item["event"] == "github.repair.started")
    assert repair_event["reflection_action"] in {"retry", "repair", "replan"}


@pytest.mark.asyncio
async def test_ci_retry_exhaustion_is_bounded(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    github = FakeGitHub(ci_cycles=[[_check("failure")]])
    pushes: list[tuple[str, str]] = []
    result = await _workflow(
        ScriptedLoop(repo, ["VALUE = 2\n", "VALUE = 3\n", "VALUE = 4\n"]),
        github=github,
        pushes=pushes,
        max_repairs=1,
    ).run(_request(repo, publish=True))
    assert result.status is GitWorkflowStatus.FAILED
    assert result.repair_attempts == 1
    assert len(pushes) == 2
    assert "budget exhausted" in (result.error or "")


@pytest.mark.asyncio
async def test_cancellation_during_repair_stops_further_commits_and_pushes(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    github = FakeGitHub(ci_cycles=[[_check("failure")]])
    pushes: list[tuple[str, str]] = []
    loop = ScriptedLoop(repo, ["VALUE = 2\n", "VALUE = 3\n"])
    workflow = _workflow(loop, github=github, pushes=pushes)
    loop.on_run = lambda calls: workflow.cancel() if calls == 2 else None
    result = await workflow.run(_request(repo, publish=True))
    assert result.status is GitWorkflowStatus.CANCELLED
    assert len(pushes) == 1
    assert run_git(["rev-list", "--count", "main..HEAD"], repo) == "1"


def test_detached_head_and_protected_push_are_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    with pytest.raises(GitOperationError, match="protected branch"):
        push_branch("main", repo)
    run_git(["checkout", "--detach"], repo)
    with pytest.raises(GitOperationError, match="detached HEAD"):
        ensure_work_branch("kodiak/task/work", repo)


@pytest.mark.asyncio
async def test_ci_polling_is_bounded_when_checks_never_arrive(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    github = FakeGitHub(ci_cycles=[[]])
    result = await _workflow(ScriptedLoop(repo, ["VALUE = 2\n"]), github=github).run(
        _request(repo, publish=True)
    )
    assert result.status is GitWorkflowStatus.FAILED
    assert result.ci_status is CIStatus.ERROR
    assert github.check_calls == 2
