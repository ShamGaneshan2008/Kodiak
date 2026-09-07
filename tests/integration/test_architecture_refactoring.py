from __future__ import annotations

from pathlib import Path

import pytest

from kodiak.agents.architecture_intelligence import (
    ArchitectureIntelligenceService,
    ArchitectureRisk,
    ArchitectureRule,
    MigrationPlan,
    MigrationStage,
    RefactoringProposal,
)
from kodiak.agents.repository_intelligence import FindingStatus, RepositoryIntelligenceService
from kodiak.orchestration.approval_gate import ApprovalRequest, ApprovalStatus
from kodiak.orchestration.git_workflow import GitWorkflowResult, GitWorkflowStatus
from kodiak.orchestration.refactoring_workflow import (
    ArchitectureRefactoringWorkflow,
    SimulationDecision,
)


def _write_repo(root: Path) -> None:
    package = root / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(
        "from app import api\nfrom app import store\nVALUE = 1\n", encoding="utf-8"
    )
    (package / "api.py").write_text("from app import core\n", encoding="utf-8")
    (package / "store.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "consumer.py").write_text("from app import core\n", encoding="utf-8")


def _proposal(*, files: int = 1, risk: ArchitectureRisk = ArchitectureRisk.HIGH):
    paths = tuple(f"app/module_{index}.py" for index in range(files))
    return RefactoringProposal(
        proposal_id="proposal",
        finding_id="finding",
        title="Refactor architecture",
        objective="Break the dependency cycle.",
        target_modules=("app.core",),
        target_paths=paths,
        verification_targets=("tests/test_core.py",),
        estimated_files=files,
        risk=risk,
        constraints=("Preserve behavior.",),
    )


class Approval:
    def __init__(self, status: ApprovalStatus) -> None:
        self.status = status
        self.calls = []

    async def request_approval(self, operation, details):
        self.calls.append((operation, details))
        return ApprovalRequest(operation=operation, details=details, status=self.status)


class Workflow:
    def __init__(self) -> None:
        self.calls = []

    async def run(self, request):
        self.calls.append(request)
        return GitWorkflowResult(
            status=GitWorkflowStatus.READY_FOR_REVIEW,
            task_id=request.task_id,
            branch="codex/refactor",
            pr_number=7,
            ci_status="pass",
        )


def test_architecture_graph_detects_cycle_and_coupling(tmp_path):
    _write_repo(tmp_path)
    snapshot = ArchitectureIntelligenceService(high_coupling_threshold=2).analyze(tmp_path)
    assert snapshot.modules == 5
    assert snapshot.edges >= 4
    assert snapshot.cycles
    assert {item.category for item in snapshot.findings} >= {
        "dependency_cycle",
        "coupling_hotspot",
    }


def test_architecture_metrics_are_deterministic(tmp_path):
    _write_repo(tmp_path)
    service = ArchitectureIntelligenceService(high_coupling_threshold=2)
    first = service.analyze(tmp_path)
    second = service.analyze(tmp_path)
    assert first.metrics == second.metrics
    assert [item.finding_id for item in first.findings] == [
        item.finding_id for item in second.findings
    ]


def test_change_impact_contains_transitive_dependents(tmp_path):
    _write_repo(tmp_path)
    service = ArchitectureIntelligenceService(high_coupling_threshold=2)
    snapshot = service.analyze(tmp_path)
    impacted = service.predict_change_impact(snapshot, "app.core")
    assert "app.core" in impacted
    assert "app.consumer" in impacted


def test_impact_preserves_paths_and_recommends_importing_tests(tmp_path):
    _write_repo(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text("from app import core\n", encoding="utf-8")
    service = ArchitectureIntelligenceService(high_coupling_threshold=2)
    snapshot = service.analyze(tmp_path)
    impact = service.impact(snapshot, "app.core")
    assert {"app.api", "app.consumer"}.issubset(impact.direct_dependents)
    assert any(path.modules == ("app.core", "app.consumer") for path in impact.paths)
    assert "tests/test_core.py" in impact.relevant_tests
    assert "pytest tests/unit" in impact.verification_scope


def test_explicit_boundary_rule_has_exact_evidence(tmp_path):
    _write_repo(tmp_path)
    rule = ArchitectureRule("api-no-core", "app.api", "app.core", "API must use a service.")
    snapshot = ArchitectureIntelligenceService(rules=(rule,)).analyze(tmp_path)
    finding = next(item for item in snapshot.findings if item.category == "architecture_boundary")
    assert "Explicit rule api-no-core" in finding.evidence[0]
    assert "app.api -> app.core via import" in finding.evidence[1]


def test_no_boundary_violation_without_explicit_rules(tmp_path):
    _write_repo(tmp_path)
    snapshot = ArchitectureIntelligenceService().analyze(tmp_path)
    assert not any(item.category == "architecture_boundary" for item in snapshot.findings)


def test_findings_are_adapted_to_canonical_repository_model(tmp_path):
    _write_repo(tmp_path)
    snapshot = ArchitectureIntelligenceService(high_coupling_threshold=2).analyze(tmp_path)
    assert snapshot.repository_findings
    assert all(item.status is FindingStatus.VALIDATED for item in snapshot.repository_findings)
    assert all(item.evidence for item in snapshot.repository_findings)


def test_repository_intelligence_is_canonical_architecture_entrypoint(tmp_path):
    _write_repo(tmp_path)
    snapshot = RepositoryIntelligenceService().analyze_architecture(tmp_path)
    assert snapshot.cycles
    assert snapshot.repository_findings


def test_incremental_analysis_reuses_unchanged_modules(tmp_path):
    _write_repo(tmp_path)
    service = ArchitectureIntelligenceService()
    first = service.analyze(tmp_path)
    assert first.files_processed
    second = service.analyze(tmp_path)
    assert not second.files_processed
    assert len(second.files_unchanged) == second.modules
    (tmp_path / "app" / "store.py").write_text(
        "from app import consumer\nVALUE = 2\n", encoding="utf-8"
    )
    third = service.analyze(tmp_path)
    assert third.files_processed == ("app/store.py",)
    assert any("app.store" in cycle for cycle in third.cycles)


def test_plan_validation_detects_missing_consumers(tmp_path):
    _write_repo(tmp_path)
    service = ArchitectureIntelligenceService(high_coupling_threshold=2)
    snapshot = service.analyze(tmp_path)
    impact = service.impact(snapshot, "app.core")
    stage = MigrationStage(
        "one",
        "Change core only",
        ("app.core",),
        (),
        "Core changes.",
        ("pytest",),
        "Revert core.",
    )
    validation = service.validate_plan(MigrationPlan("invalid", (stage,)), impact)
    assert not validation.valid
    assert "app.consumer" in validation.errors[0]


def test_proposed_staged_plan_is_dependency_ordered(tmp_path):
    _write_repo(tmp_path)
    service = ArchitectureIntelligenceService(high_coupling_threshold=2)
    snapshot = service.analyze(tmp_path)
    proposal = snapshot.proposals[0]
    assert proposal.plan is not None
    impact = service.impact(snapshot, proposal.target_modules)
    assert service.validate_plan(proposal.plan, impact).valid
    assert proposal.plan.stages[1].dependencies == (proposal.plan.stages[0].stage_id,)


def test_before_after_reports_cycle_removed(tmp_path):
    _write_repo(tmp_path)
    service = ArchitectureIntelligenceService()
    before = service.analyze(tmp_path)
    (tmp_path / "app" / "api.py").write_text("VALUE = 3\n", encoding="utf-8")
    after = service.analyze(tmp_path)
    comparison = service.compare(before, after)
    assert comparison["cycles_after"] < comparison["cycles_before"]
    assert comparison["resolved_cycles"]


def test_proposals_have_concrete_scope_and_constraints(tmp_path):
    _write_repo(tmp_path)
    proposal = (
        ArchitectureIntelligenceService(high_coupling_threshold=2).analyze(tmp_path).proposals[0]
    )
    assert proposal.target_paths
    assert proposal.estimated_files == len(proposal.target_paths)
    assert proposal.constraints


def test_simulation_blocks_excessive_blast_radius():
    workflow = ArchitectureRefactoringWorkflow(Workflow(), max_files=2)
    simulation = workflow.simulate(_proposal(files=3))
    assert simulation.decision is SimulationDecision.BLOCKED
    assert "exceeds limit" in simulation.risk_reasons[0]


def test_simulation_blocks_proposal_without_paths():
    workflow = ArchitectureRefactoringWorkflow(Workflow())
    proposal = _proposal(files=0)
    assert workflow.simulate(proposal).decision is SimulationDecision.BLOCKED


def test_low_risk_simulation_is_safe_to_review():
    workflow = ArchitectureRefactoringWorkflow(Workflow())
    assert workflow.simulate(_proposal(risk=ArchitectureRisk.LOW)).decision is (
        SimulationDecision.SAFE_TO_REVIEW
    )


@pytest.mark.asyncio
async def test_denied_approval_prevents_migration(tmp_path):
    approval = Approval(ApprovalStatus.DENIED)
    git_workflow = Workflow()
    workflow = ArchitectureRefactoringWorkflow(git_workflow, approval_gate=approval)
    result = await workflow.run(_proposal(), repository=tmp_path)
    assert not result.approved
    assert not git_workflow.calls
    assert approval.calls[0][0] == "architecture_refactor"


@pytest.mark.asyncio
async def test_approved_proposal_delegates_to_git_workflow(tmp_path):
    approval = Approval(ApprovalStatus.APPROVED)
    git_workflow = Workflow()
    workflow = ArchitectureRefactoringWorkflow(git_workflow, approval_gate=approval)
    result = await workflow.run(
        _proposal(),
        repository=tmp_path,
        publish_remote=True,
        github_owner="owner",
        github_repo="repo",
    )
    request = git_workflow.calls[0]
    assert result.approved
    assert result.workflow.status is GitWorkflowStatus.READY_FOR_REVIEW
    assert request.intended_paths == ("app/module_0.py",)
    assert request.publish_remote
    assert "Preserve behavior" in request.goal


@pytest.mark.asyncio
async def test_blocked_simulation_never_requests_approval(tmp_path):
    approval = Approval(ApprovalStatus.APPROVED)
    git_workflow = Workflow()
    workflow = ArchitectureRefactoringWorkflow(git_workflow, approval_gate=approval, max_files=1)
    result = await workflow.run(_proposal(files=2), repository=tmp_path)
    assert result.simulation.decision is SimulationDecision.BLOCKED
    assert not approval.calls
    assert not git_workflow.calls
