"""Coverage for Kodiak's deterministic local v1 workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from kodiak.api.main import app as api_app
from kodiak.cli.app import app
from kodiak.orchestration.local_agents import (
    DeterministicPlannerAgent,
    RepositoryAnalyzer,
    ReviewerAgent,
    TesterAgent,
)
from kodiak.orchestration.local_storage import LocalStateStore
from kodiak.orchestration.task_orchestrator import TaskOrchestrator
from kodiak.orchestration.v1_models import CheckResult, FileChange

runner = CliRunner()


class PassingTester:
    def run_checks(self, repository_path: Path) -> list[CheckResult]:
        return [CheckResult("pytest", ["pytest"], True, 0, "passed", "", 0.01)]


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "kodiak").mkdir()
    (tmp_path / "kodiak" / "__init__.py").write_text('__version__ = "1"\n')
    (tmp_path / "kodiak" / "api" / "routers").mkdir(parents=True)
    (tmp_path / "kodiak" / "api" / "__init__.py").write_text("")
    (tmp_path / "kodiak" / "api" / "routers" / "__init__.py").write_text("")
    (tmp_path / "kodiak" / "api" / "routers" / "health.py").write_text(
        "from fastapi import APIRouter\n\n"
        "router = APIRouter()\n\n"
        '@router.get("/health")\n'
        "async def health():\n"
        '    return {"status": "healthy"}\n'
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return tmp_path


def test_cli_help_and_groups_load() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("analyze", "task", "agents", "approval", "memory", "git", "doctor"):
        assert command in result.stdout


def test_required_local_api_routes_are_registered() -> None:
    routes = {(route.path, method) for route in api_app.routes for method in route.methods or []}
    assert ("/health", "GET") in routes
    assert ("/tasks/run", "POST") in routes
    assert ("/tasks/{task_id}", "GET") in routes
    assert ("/agents", "GET") in routes
    assert ("/approvals", "GET") in routes
    assert ("/approvals/{approval_id}/approve", "POST") in routes
    assert ("/approvals/{approval_id}/reject", "POST") in routes
    assert ("/memory/history", "GET") in routes


def test_api_health_route_works() -> None:
    with TestClient(api_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_analyze_command_structure_still_works(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyze", "analyze", str(_repo(tmp_path)), "--deep"])
    assert result.exit_code == 0
    assert "Repository Analysis" in result.stdout


def test_task_dry_run_creates_plan_and_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Add a simple health check test", repo, dry_run=True
    )
    assert result.final_status == "dry_run"
    assert result.dry_run is True
    assert result.no_commit is False
    assert result.plan.task_type == "health_check_test"
    assert result.plan.files_to_inspect[0] == "tests/test_health_check.py"
    assert not (repo / "tests" / "test_health_check.py").exists()
    assert LocalStateStore(repo).history()[0]["task_id"] == result.task_id
    serialized = result.to_dict()
    assert serialized["task_id"] == result.task_id
    assert serialized["plan"]["task_type"] == "health_check_test"
    assert serialized["review"]["approval_status"] == "not_required"


def test_readme_task_modifies_expected_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve the README quickstart section", repo, no_commit=True
    )
    assert result.final_status == "completed"
    assert result.changes[0].path == "README.md"
    assert "kodiak-v1-quickstart" in (repo / "README.md").read_text(encoding="utf-8")


def test_readme_task_does_not_duplicate_quickstart(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    readme = repo / "README.md"
    readme.write_text("# Demo\n\n## Quickstart\n\nExisting instructions.\n", encoding="utf-8")
    orchestrator = TaskOrchestrator(tester=PassingTester())
    first = orchestrator.run("Improve README quickstart", repo, no_commit=True)
    second = orchestrator.run("Improve README quickstart", repo, no_commit=True)
    content = readme.read_text(encoding="utf-8")
    assert first.final_status == "completed"
    assert second.final_status == "no_changes"
    assert content.count("## Quickstart") == 1
    assert content.count("kodiak-v1-quickstart") == 1


def test_readme_task_without_readme_is_manual(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "README.md").unlink()
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve the documentation quickstart", repo, no_commit=True
    )
    assert result.final_status == "manual_required"
    assert result.plan.task_type == "readme_missing"
    assert not (repo / "README.md").exists()


def test_health_check_task_creates_test(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Add a simple health check test", repo, no_commit=True
    )
    assert result.review.completed
    assert (repo / "tests" / "test_health_check.py").is_file()


def test_tester_records_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outcomes = iter(
        [
            subprocess.CompletedProcess(["pytest"], 0, "ok", ""),
            subprocess.CompletedProcess(["ruff"], 1, "", "lint failed"),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(outcomes))
    monkeypatch.setattr("shutil.which", lambda name: f"/tools/{name}")
    checks = TesterAgent().run_checks(tmp_path)
    assert checks[0].success is True
    assert checks[1].success is False
    assert checks[1].stderr == "lint failed"


def test_tester_skips_missing_ruff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(["pytest"], 0, "ok", ""),
    )
    monkeypatch.setattr("kodiak.orchestration.local_agents._find_executable", lambda name: None)
    checks = TesterAgent().run_checks(tmp_path)
    assert checks[1].success is None
    assert "not installed" in str(checks[1].skipped_reason)


def test_tester_skips_missing_pytest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda name: None)
    checks = TesterAgent().run_checks(tmp_path)
    assert checks[0].success is None
    assert "not installed" in str(checks[0].skipped_reason)


def test_reviewer_summarizes_changes(tmp_path: Path) -> None:
    context = RepositoryAnalyzer().analyze(_repo(tmp_path))
    plan = DeterministicPlannerAgent().plan("Improve README", context)
    change = FileChange("README.md", "modified", "docs", "before", "after")
    report = ReviewerAgent().review(
        plan, [change], [CheckResult("pytest", ["pytest"], True, 0, "", "", 0.1)]
    )
    assert report.completed
    assert report.changed_files == ["README.md"]


def test_risky_plan_creates_approval_before_edit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    class RiskyPlanner(DeterministicPlannerAgent):
        def plan(self, instruction: str, context: object):  # type: ignore[no-untyped-def]
            plan = super().plan(instruction, context)  # type: ignore[arg-type]
            plan.files_to_modify = ["pyproject.toml"]
            plan.requires_approval = True
            plan.risk_level = "high"
            return plan

    result = TaskOrchestrator(planner=RiskyPlanner(), tester=PassingTester()).run(
        "Improve README", repo, no_commit=True
    )
    assert result.final_status == "pending_approval"
    assert result.approval_id
    assert (repo / "README.md").read_text(encoding="utf-8") == "# Demo\n"


def test_approval_approve_and_reject(tmp_path: Path) -> None:
    store = LocalStateStore(_repo(tmp_path))
    first = store.create_approval(
        task_id="one", action="apply_changes", reason="risk", risk_level="high"
    )
    second = store.create_approval(
        task_id="two", action="apply_changes", reason="risk", risk_level="high"
    )
    assert store.update_approval(first.approval_id, "approved")["status"] == "approved"
    assert store.update_approval(second.approval_id, "rejected")["status"] == "rejected"


def test_no_llm_keys_are_needed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Add a simple health check test", _repo(tmp_path), dry_run=True
    )
    assert result.plan.task_type == "health_check_test"


def test_unsupported_task_returns_manual_result(tmp_path: Path) -> None:
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Replace the parser with a compiler", _repo(tmp_path), no_commit=True
    )
    assert result.final_status == "manual_required"
    assert result.changes == []
    assert "manual" in result.review.recommendations[0].lower()


def test_history_contains_safe_phase_summary(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = TaskOrchestrator(tester=PassingTester()).run(
        "Improve README quickstart", repo, no_commit=True
    )
    history = LocalStateStore(repo).history(1)[0]
    assert history["task_id"] == result.task_id
    assert history["dry_run"] is False
    assert history["no_commit"] is True
    assert history["changed_files"] == ["README.md"]
    assert "stdout" not in str(history)


def test_invalid_repository_path_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        TaskOrchestrator().run("Improve README", tmp_path / "missing", dry_run=True)


def test_non_git_repository_warns_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "kodiak.orchestration.local_agents._run_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(["git"], 128, "", "not a repo"),
    )
    context = RepositoryAnalyzer().analyze(_repo(tmp_path))
    assert context.git_branch is None
    assert "not a Git repository" in context.warnings[0]


def test_reviewer_warns_when_git_diff_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "kodiak.orchestration.local_agents._run_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(["git"], 128, "", "not a repo"),
    )
    context = RepositoryAnalyzer().analyze(_repo(tmp_path))
    plan = DeterministicPlannerAgent().plan("Improve README", context)
    report = ReviewerAgent().review(plan, [], [])
    assert "Git review unavailable" in report.warnings[0]


def test_cli_dry_run_uses_path_option(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = runner.invoke(
        app,
        ["task", "run", "Add a simple health check test", "--path", str(repo), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "dry_run" in result.stdout
