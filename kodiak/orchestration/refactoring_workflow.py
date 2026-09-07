"""Approval-gated architecture refactoring through Kodiak's autonomous Git workflow."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

from kodiak.agents.architecture_intelligence import ArchitectureRisk, RefactoringProposal
from kodiak.orchestration.approval_gate import ApprovalGate, ApprovalStatus
from kodiak.orchestration.git_workflow import (
    AutonomousGitWorkflow,
    GitWorkflowRequest,
    GitWorkflowResult,
)


class SimulationDecision(enum.StrEnum):
    SAFE_TO_REVIEW = "safe_to_review"
    REQUIRES_APPROVAL = "requires_approval"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class RefactoringSimulation:
    decision: SimulationDecision
    affected_files: tuple[str, ...]
    verification_targets: tuple[str, ...]
    risk_reasons: tuple[str, ...]


@dataclass(slots=True)
class RefactoringWorkflowResult:
    proposal_id: str
    simulation: RefactoringSimulation
    approved: bool
    workflow: GitWorkflowResult | None = None
    error: str | None = None


class ArchitectureRefactoringWorkflow:
    """Simulate, approve, then delegate migration and verification to existing owners."""

    def __init__(
        self,
        git_workflow: AutonomousGitWorkflow,
        *,
        approval_gate: ApprovalGate | None = None,
        max_files: int = 20,
    ) -> None:
        self._git_workflow = git_workflow
        self._approval = approval_gate or ApprovalGate()
        self._max_files = max(1, max_files)

    def simulate(self, proposal: RefactoringProposal) -> RefactoringSimulation:
        reasons: list[str] = []
        decision = SimulationDecision.REQUIRES_APPROVAL
        if proposal.estimated_files > self._max_files:
            reasons.append(
                f"Estimated file count {proposal.estimated_files} exceeds limit {self._max_files}."
            )
            decision = SimulationDecision.BLOCKED
        if not proposal.target_paths:
            reasons.append("Proposal has no concrete repository paths.")
            decision = SimulationDecision.BLOCKED
        if proposal.risk is ArchitectureRisk.BLOCKED:
            reasons.append("Architecture analysis marked the proposal blocked.")
            decision = SimulationDecision.BLOCKED
        if proposal.plan is not None:
            stage_ids = {stage.stage_id for stage in proposal.plan.stages}
            for stage in proposal.plan.stages:
                unknown = set(stage.dependencies) - stage_ids
                if unknown:
                    reasons.append(
                        f"Stage {stage.stage_id} has unknown dependencies: "
                        f"{', '.join(sorted(unknown))}."
                    )
                    decision = SimulationDecision.BLOCKED
                if not stage.verification:
                    reasons.append(f"Stage {stage.stage_id} has no verification criteria.")
                    decision = SimulationDecision.BLOCKED
        if proposal.risk is ArchitectureRisk.LOW and decision is not SimulationDecision.BLOCKED:
            decision = SimulationDecision.SAFE_TO_REVIEW
        return RefactoringSimulation(
            decision=decision,
            affected_files=proposal.target_paths,
            verification_targets=proposal.verification_targets,
            risk_reasons=tuple(reasons),
        )

    async def run(
        self,
        proposal: RefactoringProposal,
        *,
        repository: str | Path,
        default_branch: str = "main",
        publish_remote: bool = False,
        github_owner: str | None = None,
        github_repo: str | None = None,
    ) -> RefactoringWorkflowResult:
        simulation = self.simulate(proposal)
        result = RefactoringWorkflowResult(proposal.proposal_id, simulation, approved=False)
        if simulation.decision is SimulationDecision.BLOCKED:
            result.error = "; ".join(simulation.risk_reasons)
            return result
        approval = await self._approval.request_approval(
            "architecture_refactor",
            {
                "proposal_id": proposal.proposal_id,
                "risk": proposal.risk.value,
                "files": str(proposal.estimated_files),
            },
        )
        if approval.status is not ApprovalStatus.APPROVED:
            result.error = "Architecture refactoring approval was not granted."
            return result
        result.approved = True
        stage_lines = (
            tuple(
                f"- {stage.stage_id}: {stage.title}; depends on "
                f"{', '.join(stage.dependencies) or 'none'}; verify "
                f"{', '.join(stage.verification)}; rollback: {stage.rollback_boundary}"
                for stage in proposal.plan.stages
            )
            if proposal.plan is not None
            else ("- Use the canonical planner to create verified dependent stages.",)
        )
        goal = "\n".join(
            (
                proposal.objective,
                "Constraints:",
                *(f"- {constraint}" for constraint in proposal.constraints),
                "Verification targets:",
                *(f"- {target}" for target in proposal.verification_targets),
                "Migration stages:",
                *stage_lines,
            )
        )
        result.workflow = await self._git_workflow.run(
            GitWorkflowRequest(
                task_id=proposal.proposal_id,
                title=proposal.title,
                goal=goal,
                repository=Path(repository),
                intended_paths=proposal.target_paths,
                default_branch=default_branch,
                publish_remote=publish_remote,
                github_owner=github_owner,
                github_repo=github_repo,
            )
        )
        return result


__all__ = [
    "ArchitectureRefactoringWorkflow",
    "RefactoringSimulation",
    "RefactoringWorkflowResult",
    "SimulationDecision",
]
