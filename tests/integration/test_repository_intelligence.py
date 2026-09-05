from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.agents.repository_intelligence import (
    FindingConfidence,
    FindingEvidence,
    FindingSeverity,
    FindingStatus,
    RepositoryFinding,
    RepositoryIntelligenceService,
)
from kodiak.db.models.task import Task, TaskPriority
from kodiak.db.models.task import TaskStatus as DbTaskStatus
from kodiak.memory.models import MemoryType
from kodiak.memory.service import MemoryService
from kodiak.orchestration.approval_gate import ApprovalRequest, ApprovalStatus
from kodiak.orchestration.autonomous_loop import AutonomousLoopResult
from kodiak.orchestration.git_workflow import (
    AutonomousGitWorkflow,
    GitWorkflowRequest,
    GitWorkflowStatus,
)
from kodiak.orchestration.state import TaskState, TaskStatus
from kodiak.orchestration.verification import VerificationResult, VerificationStatus
from kodiak.tools.builtin import register_builtin_tools
from kodiak.tools.registry import ToolRegistry
from kodiak.tools.router import ToolRouter
from kodiak.utils.git_utils import run_git


class DenyingApproval:
    async def request_approval(
        self, operation: str, details: dict[str, str] | None = None
    ) -> ApprovalRequest:
        return ApprovalRequest(
            operation=operation,
            details=details or {},
            status=ApprovalStatus.DENIED,
        )


class AllowingApproval(DenyingApproval):
    async def request_approval(
        self, operation: str, details: dict[str, str] | None = None
    ) -> ApprovalRequest:
        return ApprovalRequest(
            operation=operation,
            details=details or {},
            status=ApprovalStatus.APPROVED,
        )


class FakeIssues:
    def __init__(self, existing: list[dict[str, Any]] | None = None) -> None:
        self.existing = existing or []
        self.created: list[dict[str, Any]] = []

    async def list_issues(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self.existing

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        issue = {"number": 9, "title": title, "body": body, "labels": labels or []}
        self.created.append(issue)
        return issue


class RepairLoop:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls = 0

    async def run(self, goal: str, **kwargs: Any) -> AutonomousLoopResult:
        self.calls += 1
        (self.repo / "core.py").write_text(
            "def risky(value):\n    return value\n", encoding="utf-8"
        )
        state = TaskState(title=goal, objective=goal, status=TaskStatus.COMPLETED)
        verification = VerificationResult(
            status=VerificationStatus.VERIFIED,
            message="targeted regression passed",
        )
        return AutonomousLoopResult(
            task_state=state,
            plan=None,
            execution_result=None,
            verification_result=verification,
            selected_agent="coder",
        )

    def cancel(self) -> None:
        return None


def _repo(path: Path, source: str | None = None, test: str | None = None) -> Path:
    path.mkdir()
    (path / "core.py").write_text(
        source or "def risky(value):\n    return value\n", encoding="utf-8"
    )
    if test is not None:
        tests = path / "tests"
        tests.mkdir()
        (tests / "test_core.py").write_text(test, encoding="utf-8")
    return path


def _git_init(repo: Path) -> None:
    run_git(["init", "-b", "main"], repo)
    run_git(["config", "user.email", "kodiak@example.test"], repo)
    run_git(["config", "user.name", "Kodiak Test"], repo)
    run_git(["add", "--", "."], repo)
    run_git(["commit", "-m", "initial"], repo)


def _validated_finding(repo_id: str = "owner/repo") -> RepositoryFinding:
    return RepositoryFinding(
        repository_id=repo_id,
        category="static_bug",
        title="Broad exception handler in core.py",
        description="An empty handler hides failures.",
        evidence=(
            FindingEvidence(
                "ast_pattern",
                "broad_exception_detector",
                "Empty broad exception handler.",
                "core.py",
                line_start=3,
                metadata={"empty_handler": True},
            ),
        ),
        affected_files=("core.py",),
        severity=FindingSeverity.MEDIUM,
        confidence=FindingConfidence.HIGH,
        status=FindingStatus.VALIDATED,
        impact_score=20,
        priority=TaskPriority.HIGH,
    )


@pytest.mark.asyncio
async def test_basic_scan_is_deterministic_and_findings_have_evidence(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo", "# TODO: add timeout\ndef risky(value):\n    return value\n")
    service = RepositoryIntelligenceService()
    first = await service.scan("repo", repo)
    second = await service.scan("repo", repo)

    assert [(item.category, item.title) for item in first.findings] == [
        (item.category, item.title) for item in second.findings
    ]
    assert all(item.evidence for item in first.findings)
    assert any(item.category == "technical_debt" for item in first.findings)
    todo = next(item for item in first.findings if item.category == "technical_debt")
    assert todo.status is FindingStatus.NEW
    assert todo.severity is FindingSeverity.LOW


def test_unsupported_finding_is_rejected() -> None:
    with pytest.raises(ValueError, match="concrete evidence"):
        RepositoryFinding(
            repository_id="repo",
            category="vague",
            title="Maybe broken",
            description="No support",
            evidence=(),
            affected_files=(),
        )


@pytest.mark.asyncio
async def test_broad_empty_exception_has_precise_validated_evidence(tmp_path: Path) -> None:
    source = "def risky():\n    try:\n        return 1\n    except Exception:\n        pass\n"
    snapshot = await RepositoryIntelligenceService().scan("repo", _repo(tmp_path / "repo", source))
    finding = next(item for item in snapshot.findings if item.category == "static_bug")
    assert finding.status is FindingStatus.VALIDATED
    assert finding.confidence is FindingConfidence.HIGH
    assert finding.evidence[0].line_start == 4
    assert finding.auto_fix_eligible


@pytest.mark.asyncio
async def test_missing_test_is_candidate_not_bug(tmp_path: Path) -> None:
    snapshot = await RepositoryIntelligenceService().scan("repo", _repo(tmp_path / "repo"))
    finding = next(item for item in snapshot.findings if item.category == "testing_gap")
    assert finding.status is FindingStatus.NEW
    assert finding.confidence is FindingConfidence.LOW


@pytest.mark.asyncio
async def test_reproduced_test_failure_is_high_confidence(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path / "repo",
        test="from core import risky\n\ndef test_risky():\n    assert risky(1) == 2\n",
    )
    registry = ToolRegistry()
    register_builtin_tools(registry, workspace_root=repo)
    service = RepositoryIntelligenceService(tool_router=ToolRouter(registry=registry))
    snapshot = await service.scan("repo", repo, run_tests=True)
    finding = next(item for item in snapshot.findings if item.category == "test_failure")
    assert finding.status is FindingStatus.VALIDATED
    assert finding.confidence is FindingConfidence.HIGH
    assert finding.reproducibility == "reproduced"
    assert finding.evidence[0].metadata["returncode"] != 0


@pytest.mark.asyncio
async def test_git_churn_is_hotspot_signal_not_defect(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    _git_init(repo)
    for value in range(2, 5):
        (repo / "core.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
        run_git(["add", "--", "core.py"], repo)
        run_git(["commit", "-m", f"change {value}"], repo)
    snapshot = await RepositoryIntelligenceService().scan("repo", repo)
    finding = next(item for item in snapshot.findings if item.category == "hotspot")
    assert finding.status is FindingStatus.NEW
    assert finding.severity is FindingSeverity.INFO
    assert finding.evidence[0].metadata["commit_count"] == 4


@pytest.mark.asyncio
async def test_dependency_cycle_is_deterministic_validated_finding(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo", "import helper\ndef risky():\n    return helper.VALUE\n")
    (repo / "helper.py").write_text("import core\nVALUE = 1\n", encoding="utf-8")
    snapshot = await RepositoryIntelligenceService().scan("repo", repo)
    finding = next(item for item in snapshot.findings if item.category == "dependency_cycle")
    assert finding.status is FindingStatus.VALIDATED
    assert finding.evidence[0].metadata["cycle"]


@pytest.mark.asyncio
async def test_multi_signal_correlation_strengthens_one_hotspot(tmp_path: Path) -> None:
    branches = "\n".join(f"    if value == {i}: value += 1" for i in range(10))
    source = f"def risky(value):\n{branches}\n    raise ValueError('broken')\n"
    test = "from core import risky\n\ndef test_risky():\n    risky(1)\n"
    repo = _repo(tmp_path / "repo", source, test)
    _git_init(repo)
    for index in range(3):
        with (repo / "core.py").open("a", encoding="utf-8") as stream:
            stream.write(f"# revision {index}\n")
        run_git(["add", "--", "core.py"], repo)
        run_git(["commit", "-m", f"revision {index}"], repo)
    registry = ToolRegistry()
    register_builtin_tools(registry, workspace_root=repo)
    service = RepositoryIntelligenceService(
        tool_router=ToolRouter(registry=registry), complexity_threshold=5
    )
    snapshot = await service.scan("repo", repo, run_tests=True)
    correlated = [item for item in snapshot.findings if item.category == "correlated_hotspot"]
    assert len(correlated) == 1
    assert correlated[0].status is FindingStatus.VALIDATED
    assert len(correlated[0].evidence) >= 3


def test_deduplication_merges_detector_evidence() -> None:
    service = RepositoryIntelligenceService()
    first = _validated_finding()
    second = _validated_finding()
    second.evidence = (
        FindingEvidence("lint", "ruff", "Same handler flagged.", "core.py", line_start=3),
    )
    canonical = service._deduplicate([first, second])
    assert len(canonical) == 1
    assert len(canonical[0].evidence) == 2
    assert second.status is FindingStatus.DUPLICATE


def test_confidence_severity_dismissal_and_task_gate_are_independent() -> None:
    finding = _validated_finding()
    finding.severity = FindingSeverity.HIGH
    finding.confidence = FindingConfidence.LOW
    finding.dismiss("not reproducible")
    assert finding.severity is FindingSeverity.HIGH
    assert finding.confidence is FindingConfidence.LOW
    assert finding.status is FindingStatus.DISMISSED
    with pytest.raises(ValueError, match="validated"):
        RepositoryIntelligenceService().propose_task(finding)


@pytest.mark.asyncio
async def test_impact_ranking_is_deterministic(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path / "repo",
        (
            "# TODO: docs\n"
            "def risky():\n"
            "    try:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        pass\n"
        ),
    )
    snapshot = await RepositoryIntelligenceService().scan("repo", repo)
    scores = [finding.impact_score for finding in snapshot.findings]
    assert scores == sorted(scores, reverse=True)
    assert all(finding.impact_explanation for finding in snapshot.findings)


def test_validated_finding_generates_existing_task_model() -> None:
    task = RepositoryIntelligenceService().propose_task(_validated_finding())
    assert isinstance(task, Task)
    assert task.status is DbTaskStatus.PENDING
    assert task.context["finding"]["finding_id"] == task.source_ref
    assert "Suggested verification" in (task.description or "")


@pytest.mark.asyncio
async def test_denied_approval_and_duplicate_remote_issue_block_creation() -> None:
    finding = _validated_finding()
    denied_client = FakeIssues()
    denied = RepositoryIntelligenceService(approval_gate=DenyingApproval())
    assert (
        await denied.create_github_issue(finding, client=denied_client, owner="owner", repo="repo")
        is None
    )
    assert denied_client.created == []

    duplicate_client = FakeIssues(
        [{"title": finding.title, "body": "Existing matching report", "state": "open"}]
    )
    assert (
        await denied.create_github_issue(
            finding, client=duplicate_client, owner="owner", repo="repo"
        )
        is None
    )
    assert duplicate_client.created == []


@pytest.mark.asyncio
async def test_approved_validated_finding_creates_one_structured_issue() -> None:
    client = FakeIssues()
    service = RepositoryIntelligenceService(approval_gate=AllowingApproval(), max_remote_issues=1)
    created = await service.create_github_issue(
        _validated_finding(), client=client, owner="owner", repo="repo"
    )
    assert created is not None
    assert len(client.created) == 1
    assert "Evidence" in client.created[0]["body"]
    assert "severity:medium" in client.created[0]["labels"]
    another = _validated_finding()
    another.title = "Another validated repository problem"
    with pytest.raises(RuntimeError, match="quota exhausted"):
        await service.create_github_issue(another, client=client, owner="owner", repo="repo")


@pytest.mark.asyncio
async def test_ci_and_memory_contribute_bounded_historical_signals(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo", test="def test_ok():\n    assert True\n")
    memory = MemoryService()
    await memory.add(
        "Recurring failure in repository verification",
        memory_type=MemoryType.SEMANTIC,
        tags=["failure"],
        metadata={"failure_category": "verification"},
    )
    ci_failures = [
        {"name": "tests", "conclusion": "failure"},
        {"name": "tests", "conclusion": "failure"},
    ]
    snapshot = await RepositoryIntelligenceService(memory=memory).scan(
        "repo", repo, ci_failures=ci_failures
    )
    categories = {item.category for item in snapshot.findings}
    assert "historical_risk" in categories
    assert "ci_failure" in categories
    ci = next(item for item in snapshot.findings if item.category == "ci_failure")
    assert ci.status is FindingStatus.NEW
    assert ci.evidence[0].metadata["occurrences"] == 2


@pytest.mark.asyncio
async def test_incremental_scan_processes_only_changed_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    service = RepositoryIntelligenceService()
    first = await service.scan("repo", repo)
    (repo / "core.py").write_text("# TODO: changed\ndef risky():\n    return 1\n", encoding="utf-8")
    second = await service.scan("repo", repo)
    assert set(first.files_processed) == {"core.py", "helper.py"}
    assert second.files_processed == ("core.py",)
    assert second.files_unchanged == ("helper.py",)


@pytest.mark.asyncio
async def test_secret_material_is_redacted_and_dotenv_ignored(tmp_path: Path) -> None:
    token = "ghp_" + "x" * 40
    repo = _repo(tmp_path / "repo", f"# TODO: rotate {token}\ndef risky():\n    return 1\n")
    (repo / ".env").write_text(f"TOKEN={token}\n", encoding="utf-8")
    snapshot = await RepositoryIntelligenceService().scan("repo", repo)
    payload = str([finding.to_dict() for finding in snapshot.findings])
    assert token not in payload
    assert "REDACTED_GITHUB_TOKEN" in payload
    assert ".env" not in payload


@pytest.mark.asyncio
async def test_repository_agent_exposes_discovery_pipeline(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path / "repo",
        "def risky():\n    try:\n        return 1\n    except Exception:\n        pass\n",
    )
    agent = RepositoryAnalyzerAgent(intelligence=RepositoryIntelligenceService())
    from kodiak.agents.base import AgentInput

    output = await agent.run(
        AgentInput(
            task_id="scan-task",
            project_id="project",
            instruction="discover repository risks",
            context={"repository_path": str(repo), "discover_issues": True},
        )
    )
    intelligence = output.result["repository_intelligence"]
    assert intelligence["findings"]
    assert intelligence["findings"][0]["evidence"]


@pytest.mark.asyncio
async def test_discovery_to_task_to_existing_git_workflow(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path / "repo",
        "def risky():\n    try:\n        return 1\n    except Exception:\n        pass\n",
    )
    _git_init(repo)
    service = RepositoryIntelligenceService()
    snapshot = await service.scan("owner/repo", repo)
    finding = next(item for item in snapshot.findings if item.category == "static_bug")
    task = service.propose_task(finding)
    loop = RepairLoop(repo)
    workflow = AutonomousGitWorkflow(loop)  # type: ignore[arg-type]
    result = await workflow.run(
        GitWorkflowRequest(
            task_id=str(task.id),
            title=task.title,
            goal=task.description or task.title,
            repository=repo,
            intended_paths=("core.py",),
        )
    )
    assert result.status is GitWorkflowStatus.COMMITTED
    assert result.changed_files == ("core.py",)
    assert loop.calls == 1
