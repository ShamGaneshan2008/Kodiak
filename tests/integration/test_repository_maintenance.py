from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kodiak.agents.repository_intelligence import (
    EffortClass,
    FindingConfidence,
    FindingEvidence,
    FindingSeverity,
    FindingStatus,
    RepositoryFinding,
    RepositoryHealthSnapshot,
)
from kodiak.db.models.task import TaskPriority
from kodiak.orchestration.git_workflow import GitWorkflowResult, GitWorkflowStatus
from kodiak.orchestration.repository_maintenance import (
    FindingTransition,
    JSONMaintenanceStateStore,
    MaintenanceDecision,
    MaintenanceRequest,
    RepositoryMaintenanceService,
)
from kodiak.orchestration.scheduler import TaskScheduler
from kodiak.orchestration.state import ExecutionState


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Kodiak Test")
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "initial")
    return root


def _finding(*, summary: str = "deterministic failure", category: str = "test_failure"):
    return RepositoryFinding(
        repository_id="repo",
        category=category,
        title="A regression",
        description="A validated regression",
        evidence=(
            FindingEvidence(
                kind="test_failure", source="pytest", summary=summary, file_path="app.py"
            ),
        ),
        affected_files=("app.py",),
        severity=FindingSeverity.MEDIUM,
        confidence=FindingConfidence.HIGH,
        status=FindingStatus.VALIDATED,
        effort=EffortClass.SMALL,
        priority=TaskPriority.HIGH,
        impact_score=80,
        source_detectors=("test_runner",),
    )


class FakeIntelligence:
    def __init__(self, scans: list[tuple[RepositoryFinding, ...]]) -> None:
        self.scans = scans
        self.calls: list[dict] = []

    async def scan(self, repository_id, root, **kwargs):
        self.calls.append(kwargs)
        findings = self.scans[min(len(self.calls) - 1, len(self.scans) - 1)]
        return RepositoryHealthSnapshot(
            repository_id=repository_id,
            scan_id=str(len(self.calls)),
            findings=findings,
            files_considered=1,
            files_processed=("app.py",),
            files_unchanged=(),
            dimensions={},
            duration_seconds=0,
        )

    def propose_task(self, finding):
        return SimpleNamespace(id="task-1", title=finding.title, description=finding.description)


class FakeWorkflow:
    def __init__(self, status=GitWorkflowStatus.COMMITTED) -> None:
        self.status = status
        self.calls = []

    async def run(self, request):
        self.calls.append(request)
        return GitWorkflowResult(
            status=self.status,
            task_id=request.task_id,
            commit_sha="repair-sha" if self.status is GitWorkflowStatus.COMMITTED else None,
            error=None if self.status is GitWorkflowStatus.COMMITTED else "verification failed",
        )


def _service(tmp_path, scans, *, workflow=None, attempts=2):
    intelligence = FakeIntelligence(scans)
    store = JSONMaintenanceStateStore(tmp_path / "maintenance.json")
    service = RepositoryMaintenanceService(
        intelligence, store, git_workflow=workflow, max_attempts_per_finding=attempts
    )
    return service, intelligence, store


@pytest.mark.asyncio
async def test_initial_baseline_is_persisted(tmp_path):
    root = _repo(tmp_path)
    service, intelligence, store = _service(tmp_path, [(_finding(),)])
    result = await service.run(MaintenanceRequest("repo", root, run_tests=False))
    record = next(iter(store.load()["repositories"].values()))
    assert record["baseline"]["sha"] == _git(root, "rev-parse", "HEAD")
    assert result.transitions == {FindingTransition.NEW_FINDING.value: 1}
    assert len(intelligence.calls) == 1


@pytest.mark.asyncio
async def test_unchanged_run_is_a_fast_noop(tmp_path):
    root = _repo(tmp_path)
    service, intelligence, _ = _service(tmp_path, [(_finding(),)])
    request = MaintenanceRequest("repo", root, run_tests=False)
    await service.run(request)
    result = await service.run(request)
    assert result.no_change
    assert len(intelligence.calls) == 1


@pytest.mark.asyncio
async def test_new_commit_detects_changed_files_incrementally(tmp_path):
    root = _repo(tmp_path)
    service, intelligence, _ = _service(tmp_path, [(), ()])
    request = MaintenanceRequest("repo", root, run_tests=False)
    await service.run(request)
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "change")
    result = await service.run(request)
    assert result.changed_files == ("app.py",)
    assert intelligence.calls[-1]["incremental"] is True


@pytest.mark.asyncio
async def test_finding_resolution_is_retained_in_history(tmp_path):
    root = _repo(tmp_path)
    service, _, store = _service(tmp_path, [(_finding(),), ()])
    request = MaintenanceRequest("repo", root, run_tests=False)
    await service.run(request)
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "commit", "-am", "fix")
    result = await service.run(request)
    item = next(iter(next(iter(store.load()["repositories"].values()))["backlog"].values()))
    assert result.transitions["resolved"] == 1
    assert item["status"] == "resolved"


@pytest.mark.asyncio
async def test_resolved_finding_reopens_on_recurrence(tmp_path):
    root = _repo(tmp_path)
    finding = _finding()
    service, _, store = _service(tmp_path, [(finding,), (), (finding,)])
    request = MaintenanceRequest("repo", root, run_tests=False)
    await service.run(request)
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "commit", "-am", "fixed")
    await service.run(request)
    (root / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(root, "commit", "-am", "reintroduced")
    result = await service.run(request)
    item = next(iter(next(iter(store.load()["repositories"].values()))["backlog"].values()))
    assert result.transitions["reopened"] == 1
    assert item["recurrence_count"] == 1


@pytest.mark.asyncio
async def test_human_dismissal_is_respected_until_evidence_changes(tmp_path):
    root = _repo(tmp_path)
    first, changed = _finding(), _finding(summary="new deterministic evidence")
    service, _, store = _service(tmp_path, [(first,), (first,), (changed,)])
    request = MaintenanceRequest("repo", root, run_tests=False)
    await service.run(request)
    fingerprint = service.finding_fingerprint(first)
    service.dismiss("repo", root, fingerprint, "false positive")
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "commit", "-am", "second")
    unchanged = await service.run(request)
    assert unchanged.transitions["unchanged"] == 1
    (root / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(root, "commit", "-am", "third")
    reconsidered = await service.run(request)
    item = next(iter(next(iter(store.load()["repositories"].values()))["backlog"].values()))
    assert reconsidered.transitions["reopened"] == 1
    assert item["status"] == "active"


def test_auto_repair_policy_is_conservative(tmp_path):
    service, _, _ = _service(tmp_path, [()])
    assert service._decision(_finding()) is MaintenanceDecision.AUTO_REPAIR
    assert service._decision(_finding(category="architecture")) is MaintenanceDecision.HUMAN_REVIEW


@pytest.mark.asyncio
async def test_auto_repair_uses_existing_git_workflow(tmp_path):
    root = _repo(tmp_path)
    workflow = FakeWorkflow()
    service, _, store = _service(tmp_path, [(_finding(),)], workflow=workflow)
    result = await service.run(MaintenanceRequest("repo", root, run_tests=False))
    item = next(iter(next(iter(store.load()["repositories"].values()))["backlog"].values()))
    assert result.repairs_successful == 1
    assert item["status"] == "fixed"
    assert len(workflow.calls) == 1


@pytest.mark.asyncio
async def test_failed_repair_preserves_evidence_and_is_bounded(tmp_path):
    root = _repo(tmp_path)
    workflow = FakeWorkflow(GitWorkflowStatus.FAILED)
    service, _, store = _service(tmp_path, [(_finding(),)], workflow=workflow, attempts=1)
    request = MaintenanceRequest("repo", root, run_tests=False)
    first = await service.run(request)
    record = next(iter(store.load()["repositories"].values()))
    record["checkpoint"] = "failed"
    store.save(store.load())
    second = await service.run(request)
    assert first.repairs_attempted == 1
    assert second.repairs_attempted == 0
    assert len(workflow.calls) == 1


@pytest.mark.asyncio
async def test_per_run_repair_budget_is_respected(tmp_path):
    root = _repo(tmp_path)
    another = _finding(category="static_bug")
    another.affected_files = ("other.py",)
    workflow = FakeWorkflow()
    service, _, _ = _service(tmp_path, [(_finding(), another)], workflow=workflow)
    await service.run(MaintenanceRequest("repo", root, run_tests=False, max_repairs=1))
    assert len(workflow.calls) == 1


@pytest.mark.asyncio
async def test_documentation_only_change_does_not_run_tests(tmp_path):
    root = _repo(tmp_path)
    service, intelligence, _ = _service(tmp_path, [(), ()])
    request = MaintenanceRequest("repo", root)
    await service.run(request)
    (root / "README.md").write_text("docs\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "docs")
    await service.run(request)
    assert intelligence.calls[-1]["run_tests"] is False


@pytest.mark.asyncio
async def test_ci_failure_is_forwarded_as_regression_evidence(tmp_path):
    root = _repo(tmp_path)
    service, intelligence, _ = _service(tmp_path, [()])
    failures = [{"name": "unit", "conclusion": "failure"}]
    await service.run(
        MaintenanceRequest("repo", root, run_tests=False, ci_status="fail", ci_failures=failures)
    )
    assert intelligence.calls[0]["ci_failures"] == failures


@pytest.mark.asyncio
async def test_scheduler_adapter_uses_canonical_scheduler(tmp_path):
    scheduler = TaskScheduler(ExecutionState())
    task = await RepositoryMaintenanceService.schedule(
        scheduler, MaintenanceRequest("repo", tmp_path)
    )
    assert task.agent_type == "repository"
    assert task.input_data["operation"] == "maintenance"


def test_state_contains_digests_not_evidence_or_secret_contents(tmp_path):
    _repo(tmp_path)
    service, _, store = _service(tmp_path, [()])
    finding = _finding(summary="token=super-secret-value")
    record = {"backlog": {}}
    service._update_backlog(record, {service.finding_fingerprint(finding): finding}, "sha")
    store.save({"version": 1, "repositories": {"repo": record}})
    persisted = store.path.read_text(encoding="utf-8")
    assert "super-secret-value" not in persisted


def test_priority_order_is_deterministic_and_recurrence_first(tmp_path):
    service, _, _ = _service(tmp_path, [()])
    first, second = _finding(), _finding(category="static_bug")
    current = {
        service.finding_fingerprint(first): first,
        service.finding_fingerprint(second): second,
    }
    record = {"backlog": {}, "attempts": {}}
    service._update_backlog(record, current, "sha")
    record["backlog"][service.finding_fingerprint(second)]["recurrence_count"] = 2
    selected = service._repair_candidates(record, current, "sha", 2)
    assert selected[0][0] == service.finding_fingerprint(second)
