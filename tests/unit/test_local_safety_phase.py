"""Safety, approval, Git inspection, history, and doctor regression tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kodiak.cli.app import app
from kodiak.cli.services.doctor_service import DoctorService
from kodiak.orchestration.approval_manager import ApprovalManager
from kodiak.orchestration.local_agents import LocalGitAgent
from kodiak.orchestration.local_storage import LocalStateStore
from kodiak.orchestration.task_orchestrator import TaskOrchestrator
from kodiak.orchestration.v1_models import CheckResult

runner = CliRunner()


class PassingTester:
    def run_checks(self, repository_path: Path) -> list[CheckResult]:
        return [CheckResult("pytest", ["pytest"], True, 0, "passed", "", 0.01)]


def _repo(path: Path) -> Path:
    (path / "kodiak").mkdir()
    (path / "kodiak" / "__init__.py").write_text("", encoding="utf-8")
    (path / "tests").mkdir()
    (path / "README.md").write_text("# Demo\n", encoding="utf-8")
    return path


def _git(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )


def _git_repo(path: Path) -> Path:
    repo = _repo(path)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Kodiak Test")
    _git(repo, "add", "README.md", "kodiak/__init__.py")
    _git(repo, "commit", "-m", "baseline")
    return repo


def test_approval_manager_persists_complete_schema(tmp_path: Path) -> None:
    manager = ApprovalManager(_repo(tmp_path))
    approval = manager.create(
        task_id="task-1", action="git_commit", reason="commit", risk_level="medium"
    )
    payload = json.loads((tmp_path / ".kodiak" / "approvals.json").read_text())
    assert payload[0] == approval.to_dict()
    assert set(payload[0]) == {
        "approval_id",
        "task_id",
        "action",
        "reason",
        "risk_level",
        "status",
        "created_at",
        "updated_at",
        "metadata",
    }


def test_approval_cli_lists_and_resolves_request(tmp_path: Path) -> None:
    manager = ApprovalManager(_repo(tmp_path))
    approval = manager.create(
        task_id="task-1", action="git_commit", reason="commit", risk_level="medium"
    )
    listed = runner.invoke(app, ["approval", "list", "--path", str(tmp_path)])
    approved = runner.invoke(
        app, ["approval", "approve", approval.approval_id, "--path", str(tmp_path)]
    )
    assert listed.exit_code == approved.exit_code == 0
    assert approval.approval_id in listed.stdout
    assert manager.get(approval.approval_id)["status"] == "approved"  # type: ignore[index]
    assert "No action was executed automatically" in approved.stdout


def test_approval_cli_handles_unknown_and_resolved_ids(tmp_path: Path) -> None:
    manager = ApprovalManager(_repo(tmp_path))
    approval = manager.create(
        task_id="task-1", action="delete_file", reason="delete", risk_level="high"
    )
    assert (
        runner.invoke(
            app, ["approval", "reject", approval.approval_id, "--path", str(tmp_path)]
        ).exit_code
        == 0
    )
    resolved = runner.invoke(
        app, ["approval", "approve", approval.approval_id, "--path", str(tmp_path)]
    )
    unknown = runner.invoke(app, ["approval", "approve", "missing", "--path", str(tmp_path)])
    assert resolved.exit_code == 1
    assert "already rejected" in resolved.stderr
    assert unknown.exit_code == 1
    assert "Approval not found" in unknown.stderr


def test_policy_covers_required_risky_operations() -> None:
    manager = ApprovalManager
    assert manager.requires_approval("git_commit")
    assert manager.requires_approval("delete_file")
    assert manager.requires_approval("git_push")
    assert manager.requires_approval("risky_shell", command=["tool", "dangerous"])
    assert manager.requires_approval("apply_changes", paths=[f"f{i}.py" for i in range(6)])
    for path in (
        "uv.lock",
        "requirements.txt",
        ".github/workflows/test.yml",
        "src/auth/login.py",
        "alembic/versions/001.py",
    ):
        assert manager.requires_approval("apply_changes", paths=[path])


def test_task_stops_for_commit_approval_and_does_not_commit(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    before = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    result = TaskOrchestrator(tester=PassingTester()).run("Improve README quickstart", repo)
    after = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    assert result.final_status == "pending_approval"
    assert result.approval_id
    assert before == after == "1"
    assert "kodiak approval approve" in " ".join(result.review.recommendations)


def test_git_diff_summary_reports_file_categories(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("# Changed\n", encoding="utf-8")
    (repo / "staged.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "untracked.py").write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "staged.py")
    summary = LocalGitAgent().inspect(repo)
    assert summary.is_git_repo and summary.dirty
    assert "staged.py" in summary.staged_files
    assert "README.md" in summary.unstaged_files
    assert "untracked.py" in summary.untracked_files
    assert set(summary.changed_files) >= {"README.md", "staged.py", "untracked.py"}


def test_git_diff_summary_handles_non_git_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "kodiak.orchestration.local_agents._run_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(["git"], 128, "", "not a repo"),
    )
    summary = LocalGitAgent().inspect(tmp_path)
    assert not summary.is_git_repo
    assert summary.warning == "Path is not a Git repository."


def test_history_has_requested_fields_and_latest_first(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    orchestrator = TaskOrchestrator(tester=PassingTester())
    first = orchestrator.run("Add a simple health check test", repo, dry_run=True)
    second = orchestrator.run("Improve README quickstart", repo, no_commit=True)
    history = LocalStateStore(repo).history(10)
    assert [record["task_id"] for record in history[:2]] == [second.task_id, first.task_id]
    assert set(history[0]) == {
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
    }


def test_doctor_runs_and_does_not_expose_provider_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "never-print-this-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    checks = DoctorService().run(tmp_path)
    assert any(item.name == "OPENAI_API_KEY" and item.status == "present" for item in checks)
    assert secret not in repr(checks)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert secret not in result.stdout


def test_approved_commit_requires_explicit_execute(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run("Improve README quickstart", repo)
    assert result.approval_id
    manager = ApprovalManager(repo)
    manager.resolve(result.approval_id, "approved")
    executed = runner.invoke(app, ["approval", "execute", result.approval_id, "--path", str(repo)])
    assert executed.exit_code == 0
    assert "Approved local commit created" in executed.stdout
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "2"
    approval = manager.get(result.approval_id)
    assert approval and approval["metadata"]["executed_at"]


def test_approved_commit_rejects_changed_snapshot(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run("Improve README quickstart", repo)
    assert result.approval_id
    manager = ApprovalManager(repo)
    manager.resolve(result.approval_id, "approved")
    (repo / "README.md").write_text("changed after approval\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after"):
        manager.execute_approved_commit(result.approval_id)
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_local_git_commit_rejects_unrelated_staged_files(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("approved change\n", encoding="utf-8")
    (repo / "other.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "other.py")
    with pytest.raises(RuntimeError, match="unrelated staged"):
        LocalGitAgent().commit(repo, ["README.md"], "safe commit")


def test_additional_deterministic_task_types(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    structure = TaskOrchestrator(tester=PassingTester()).run(
        "Generate a project structure summary", repo, no_commit=True
    )
    unit_test = TaskOrchestrator(tester=PassingTester()).run(
        "Add a basic unit test file", repo, no_commit=True
    )
    (repo / "module.py").write_text("VALUE = os.getcwd()\n", encoding="utf-8")
    import_fix = TaskOrchestrator(tester=PassingTester()).run(
        "Fix missing import os in module.py", repo, no_commit=True
    )
    assert (
        structure.final_status == unit_test.final_status == import_fix.final_status == "completed"
    )
    assert (repo / "PROJECT_STRUCTURE.md").is_file()
    assert (repo / "tests" / "test_kodiak_smoke.py").is_file()
    assert (repo / "module.py").read_text(encoding="utf-8").startswith("import os\n")


def test_agents_list_describes_runnable_local_agents() -> None:
    result = runner.invoke(app, ["agents", "list"])
    assert result.exit_code == 0
    for name in ("planner", "repository", "coder", "tester", "reviewer", "git", "memory"):
        assert name in result.stdout
    assert "deterministic" in result.stdout
    assert "Description" in result.stdout


def test_malformed_local_storage_is_handled_safely(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    state = repo / ".kodiak"
    (state / "runs").mkdir(parents=True)
    (state / "approvals.json").write_text("{not json", encoding="utf-8")
    (state / "task_history.jsonl").write_text("bad\n[]\n", encoding="utf-8")
    (state / "runs" / "broken.json").write_text("{bad", encoding="utf-8")
    store = LocalStateStore(repo)
    assert store.approvals() == []
    assert store.history() == []
    assert store.get_run("broken") is None


def test_local_task_instruction_is_redacted_before_output_and_storage(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    secret = "sk-" + "Z" * 48
    result = TaskOrchestrator(tester=PassingTester()).run(
        f"Unsupported work token={secret}", repo, no_commit=True
    )
    assert secret not in result.instruction
    assert "[REDACTED]" in result.instruction
    history_text = (repo / ".kodiak" / "task_history.jsonl").read_text(encoding="utf-8")
    run_text = (repo / ".kodiak" / "runs" / f"{result.task_id}.json").read_text(encoding="utf-8")
    assert secret not in history_text + run_text


def test_task_file_error_is_recorded_without_false_success(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    class FailingCoder:
        def apply(self, plan: object) -> list[object]:
            raise PermissionError("write denied")

    with pytest.raises(PermissionError, match="write denied"):
        TaskOrchestrator(coder=FailingCoder(), tester=PassingTester()).run(  # type: ignore[arg-type]
            "Improve README quickstart", repo, no_commit=True
        )
    history = LocalStateStore(repo).history(1)[0]
    assert history["final_status"] == "failed"
    assert history["error"] == "write denied"
    assert "no success" in history["review_summary"]
