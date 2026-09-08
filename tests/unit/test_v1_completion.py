"""Completion coverage for Kodiak's deterministic local v1 workflow."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kodiak.cli.app import app
from kodiak.cli.services.doctor_service import PROVIDER_KEYS, DoctorService
from kodiak.cli.services.git_service import GitService
from kodiak.cli.services.task_service import TaskService
from kodiak.orchestration.approval_manager import ApprovalManager
from kodiak.orchestration.local_agents import local_agent_descriptors
from kodiak.orchestration.local_storage import LocalStateStore
from kodiak.orchestration.task_orchestrator import (
    CheckResult,
    InvalidRepositoryError,
    LocalTester,
    SafeEditor,
    TaskOrchestrator,
)
from kodiak.utils.git_utils import GitOperationError, push_branch

runner = CliRunner()


class PassingTester:
    def run(self, root: Path, *, dry_run: bool = False) -> tuple[CheckResult, ...]:
        status = "dry_run" if dry_run else "passed"
        return (
            CheckResult("pytest", status, ("python", "-m", "pytest"), 0, status),
            CheckResult("ruff check", status, ("ruff", "check", "."), 0, status),
            CheckResult("ruff format", status, ("ruff", "format", "--check", "."), 0, status),
        )


def _readme_repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# Demo\n\n## Notes\n\nSafe project.\n", encoding="utf-8")
    return tmp_path


def _health_repo(tmp_path: Path) -> Path:
    package = tmp_path / "sample"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health')\n"
        "async def health():\n"
        "    return {'status': 'healthy'}\n",
        encoding="utf-8",
    )
    return tmp_path


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_cli_help_loads_all_required_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("analyze", "task", "agents", "approval", "memory", "git", "doctor"):
        assert command in result.stdout


def test_analyze_command_still_works(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = runner.invoke(app, ["analyze", "analyze", str(tmp_path), "--deep", "--json"])
    assert result.exit_code == 0
    assert '"file_count": 1' in result.stdout


def test_task_dry_run_plans_without_source_writes_and_records_history(tmp_path: Path) -> None:
    root = _readme_repo(tmp_path)
    before = (root / "README.md").read_bytes()
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve the README quickstart section", root, dry_run=True
    )
    assert result.final_status == "dry_run"
    assert result.plan is not None and result.plan.supported
    assert result.proposed_changes[0].path == "README.md"
    assert result.changed_files == ()
    assert (root / "README.md").read_bytes() == before
    assert "not executed" in result.review_summary
    assert LocalStateStore(root).history(1)[0]["final_status"] == "dry_run"


def test_readme_no_commit_is_idempotent_and_never_duplicates_quickstart(tmp_path: Path) -> None:
    root = _readme_repo(tmp_path)
    orchestrator = TaskOrchestrator(tester=PassingTester())
    first = orchestrator.run("Improve the README quickstart section", root, no_commit=True)
    second = orchestrator.run("Improve the README quickstart section", root, no_commit=True)
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert first.final_status == "completed"
    assert first.changed_files == ("README.md",)
    assert second.final_status == "completed"
    assert second.changed_files == ()
    assert readme.casefold().count("## quickstart") == 1
    assert first.approval_id is None


def test_health_check_dry_run_verifies_import_and_proposes_test(tmp_path: Path) -> None:
    root = _health_repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Add a simple health check test", root, dry_run=True
    )
    assert result.final_status == "dry_run"
    assert result.proposed_changes[0].path == "tests/test_health.py"
    assert "from sample.main import app" in result.proposed_changes[0].after
    assert not (root / "tests").exists()


def test_unsupported_task_is_manual_and_makes_no_source_change(tmp_path: Path) -> None:
    root = _readme_repo(tmp_path)
    before = (root / "README.md").read_bytes()
    result = TaskOrchestrator(tester=PassingTester()).run("Invent a moon database", root)
    assert result.final_status == "manual_required"
    assert result.changed_files == ()
    assert (root / "README.md").read_bytes() == before


def test_invalid_repository_path_has_clean_service_and_cli_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(InvalidRepositoryError, match="does not exist"):
        TaskOrchestrator().run("Improve the README quickstart section", missing)
    result = runner.invoke(
        app,
        ["task", "run", "Improve the README quickstart section", "--path", str(missing)],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_memory_history_reads_latest_task_runs_and_limit(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    for index in range(3):
        store.append_history({"task_id": str(index), "final_status": "completed"})
    assert [item["task_id"] for item in store.history(2)] == ["2", "1"]
    result = runner.invoke(app, ["memory", "history", "--path", str(tmp_path), "--limit", "2"])
    assert result.exit_code == 0
    assert "Kodiak task history" in result.stdout


def test_approval_manager_create_approve_and_reject(tmp_path: Path) -> None:
    manager = ApprovalManager(tmp_path)
    approved = manager.create(task_id="one", action="git_commit", reason="commit")
    rejected = manager.create(task_id="two", action="delete_files", reason="delete")
    assert set(approved) >= {
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
    assert manager.resolve(approved["approval_id"], "approved")["status"] == "approved"
    assert manager.resolve(rejected["approval_id"], "rejected")["status"] == "rejected"


def test_approval_cli_lists_and_resolves_requests(tmp_path: Path) -> None:
    item = ApprovalManager(tmp_path).create(task_id="one", action="git_commit", reason="commit")
    listed = runner.invoke(app, ["approval", "list", "--path", str(tmp_path)])
    approved = runner.invoke(
        app, ["approval", "approve", item["approval_id"], "--path", str(tmp_path)]
    )
    assert listed.exit_code == approved.exit_code == 0
    assert item["approval_id"] in listed.stdout
    assert "No action was executed" in approved.stdout


def test_unknown_or_already_resolved_approval_is_clean_error(tmp_path: Path) -> None:
    manager = ApprovalManager(tmp_path)
    with pytest.raises(LookupError, match="Unknown approval ID"):
        manager.resolve("missing", "approved")
    item = manager.create(task_id="one", action="git_commit", reason="commit")
    manager.resolve(item["approval_id"], "approved")
    with pytest.raises(RuntimeError, match="already approved"):
        manager.resolve(item["approval_id"], "rejected")
    result = runner.invoke(app, ["approval", "reject", "missing", "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert "Unknown approval ID" in result.output


@pytest.mark.skipif(not os.environ.get("PATH"), reason="Git executable lookup needs PATH")
def test_git_diff_summary_in_repository(tmp_path: Path) -> None:
    if _git(tmp_path, "--version").returncode != 0:
        pytest.skip("Git is not installed")
    assert _git(tmp_path, "init").returncode == 0
    (tmp_path / "new.txt").write_text("new\n", encoding="utf-8")
    summary = GitService().diff_summary(tmp_path)
    assert summary.is_git_repo
    assert summary.dirty
    assert "new.txt" in summary.untracked_files
    assert "new.txt" in summary.changed_files


def test_git_diff_summary_handles_non_repository(tmp_path: Path) -> None:
    summary = GitService().diff_summary(tmp_path)
    assert not summary.is_git_repo
    assert summary.error == "Not a Git repository."
    result = runner.invoke(app, ["git", "diff-summary", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert "Not a Git repository" in result.output


def test_real_remote_push_is_disabled(tmp_path: Path) -> None:
    if _git(tmp_path, "--version").returncode != 0:
        pytest.skip("Git is not installed")
    assert _git(tmp_path, "init").returncode == 0
    assert _git(tmp_path, "config", "user.email", "kodiak@example.invalid").returncode == 0
    assert _git(tmp_path, "config", "user.name", "Kodiak Test").returncode == 0
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    assert _git(tmp_path, "add", "README.md").returncode == 0
    assert _git(tmp_path, "commit", "-m", "initial").returncode == 0
    assert _git(tmp_path, "switch", "-c", "feature").returncode == 0
    assert (
        _git(tmp_path, "remote", "add", "origin", "https://example.invalid/repo.git").returncode
        == 0
    )
    with pytest.raises(GitOperationError, match="disabled in Kodiak v1"):
        push_branch("feature", tmp_path)


def test_local_tester_handles_success_and_failure(tmp_path: Path) -> None:
    exit_codes = iter((0, 1, 0))

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], next(exit_codes), "", "")

    results = LocalTester(runner=fake_run).run(tmp_path)
    assert [result.status for result in results] == ["passed", "failed", "passed"]


def test_local_tester_default_timeout_supports_the_full_release_suite() -> None:
    assert LocalTester().timeout_seconds >= 900


def test_local_tester_handles_missing_pytest_and_ruff(tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        module = command[2]
        return subprocess.CompletedProcess(command, 1, "", f"No module named {module}")

    results = LocalTester(runner=fake_run).run(tmp_path)
    assert [result.status for result in results] == ["missing", "missing", "missing"]


def test_doctor_runs_and_never_prints_secret_values(tmp_path: Path, monkeypatch: Any) -> None:
    secret = "sk-" + "Z" * 48
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    payload = DoctorService(tmp_path).as_dict()
    rendered = json.dumps(payload)
    assert payload["provider"]["configured"] is True
    assert payload["provider"]["keys"]["OPENAI_API_KEY"] == "present"
    assert secret not in rendered
    assert all(name in rendered for name in PROVIDER_KEYS)


def test_agents_list_reports_actual_required_roles() -> None:
    names = {agent.name for agent in local_agent_descriptors()}
    assert names == {
        "PlannerAgent",
        "RepositoryAgent",
        "CoderAgent",
        "TesterAgent",
        "ReviewerAgent",
        "GitAgent",
        "MemoryAgent",
    }
    result = runner.invoke(app, ["agents", "list"])
    assert result.exit_code == 0
    assert names <= set(result.stdout.split())


def test_no_provider_key_fallback_runs_deterministic_task(tmp_path: Path, monkeypatch: Any) -> None:
    for key in PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve the README quickstart section", _readme_repo(tmp_path), dry_run=True
    )
    assert result.final_status == "dry_run"
    assert result.error is None


def test_task_service_runs_without_cli(tmp_path: Path) -> None:
    service = TaskService(TaskOrchestrator(tester=PassingTester()))
    result = service.run_task(
        _readme_repo(tmp_path),
        "Improve the README quickstart section",
        dry_run=True,
    )
    assert result.final_status == "dry_run"


def test_task_result_serializes_to_json(tmp_path: Path) -> None:
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve the README quickstart section", _readme_repo(tmp_path), dry_run=True
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["task_id"].startswith("task_")
    assert payload["plan"]["task_type"] == "readme_quickstart"
    assert isinstance(payload["checks"][0]["command"], list)


def test_malformed_local_state_is_handled_safely(tmp_path: Path) -> None:
    state = tmp_path / ".kodiak"
    state.mkdir()
    (state / "approvals.json").write_text("{broken", encoding="utf-8")
    (state / "task_history.jsonl").write_text('bad\n{"task_id": "valid"}\n[]\n', encoding="utf-8")
    store = LocalStateStore(tmp_path)
    assert store.load_approvals() == []
    assert store.history() == [{"task_id": "valid"}]


def test_history_redacts_secret_fields_and_token_patterns(tmp_path: Path) -> None:
    secret = "sk-" + "A" * 48
    LocalStateStore(tmp_path).append_history(
        {"task_id": "one", "api_key": secret, "instruction": f"do not store {secret}"}
    )
    raw = (tmp_path / ".kodiak" / "task_history.jsonl").read_text(encoding="utf-8")
    assert secret not in raw
    assert raw.count("[REDACTED]") == 2


def test_approval_policy_covers_every_v1_risky_category() -> None:
    paths = (
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/test.yml",
        "kodiak/auth/jwt.py",
        "alembic/versions/001.py",
        "one.py",
    )
    reasons = ApprovalManager.risk_reasons(
        action="git_commit", paths=paths, command=("git", "push", "origin")
    )
    rendered = " ".join(reasons)
    assert "Git commit" in rendered
    assert "more than five" in rendered
    assert "dependency" in rendered
    assert "lock file" in rendered
    assert "CI/CD" in rendered
    assert "auth/security" in rendered
    assert "migration" in rendered
    assert "risky shell" in rendered


def test_default_task_creates_commit_approval_but_does_not_commit(tmp_path: Path) -> None:
    if _git(tmp_path, "--version").returncode != 0:
        pytest.skip("Git is not installed")
    assert _git(tmp_path, "init").returncode == 0
    root = _readme_repo(tmp_path)
    before_head = _git(root, "rev-parse", "HEAD").stdout
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve the README quickstart section", root
    )
    after_head = _git(root, "rev-parse", "HEAD").stdout
    assert result.final_status == "awaiting_approval"
    assert result.approval_id
    assert before_head == after_head


def test_safe_editor_refuses_concurrent_overwrite(tmp_path: Path) -> None:
    root = _readme_repo(tmp_path)
    editor = SafeEditor()
    plan = TaskOrchestrator().planner.plan("Improve the README quickstart section")
    changes = editor.propose(root, plan)
    (root / "README.md").write_text("concurrent\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="concurrently changed"):
        editor.apply(root, changes)
