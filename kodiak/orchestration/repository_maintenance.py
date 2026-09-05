"""Persistent, bounded repository maintenance coordinated through existing owners."""

from __future__ import annotations

import enum
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import structlog

from kodiak.agents.repository_intelligence import (
    FindingConfidence,
    FindingSeverity,
    FindingStatus,
    RepositoryFinding,
    RepositoryIntelligenceService,
)
from kodiak.memory.models import MemoryType
from kodiak.memory.service import MemoryService
from kodiak.orchestration.git_workflow import (
    AutonomousGitWorkflow,
    GitWorkflowRequest,
    GitWorkflowStatus,
)
from kodiak.orchestration.scheduler import ScheduledTask, TaskScheduler
from kodiak.utils.git_utils import GitOperationError, current_branch, repo_root, run_git

logger = structlog.get_logger(__name__)


class MaintenanceDecision(enum.StrEnum):
    OBSERVE = "observe"
    HUMAN_REVIEW = "human_review"
    AUTO_REPAIR = "auto_repair"
    BLOCKING = "blocking"


class FindingTransition(enum.StrEnum):
    UNCHANGED = "unchanged"
    NEW_FINDING = "new_finding"
    RESOLVED = "resolved"
    REGRESSED = "regressed"
    REOPENED = "reopened"


class MaintenanceCheckpoint(enum.StrEnum):
    BASELINE_LOADED = "baseline_loaded"
    CHANGES_DETECTED = "changes_detected"
    SCAN_COMPLETE = "scan_complete"
    BACKLOG_UPDATED = "backlog_updated"
    REPAIR_STARTED = "repair_started"
    REPAIR_VERIFIED = "repair_verified"
    MAINTENANCE_COMPLETE = "maintenance_complete"
    FAILED = "failed"


@dataclass(slots=True)
class MaintenanceRequest:
    repository_id: str
    repository: Path
    run_tests: bool = True
    ci_status: str | None = None
    ci_failures: list[dict[str, Any]] = field(default_factory=list)
    max_repairs: int = 1
    publish_remote: bool = False
    default_branch: str = "main"
    github_owner: str | None = None
    github_repo: str | None = None


@dataclass(slots=True)
class MaintenanceSummary:
    run_id: str
    repository_id: str
    previous_sha: str | None
    current_sha: str
    changed_files: tuple[str, ...] = ()
    transitions: dict[str, int] = field(default_factory=dict)
    auto_repair_candidates: int = 0
    human_review: int = 0
    repairs_attempted: int = 0
    repairs_successful: int = 0
    final_ci: str | None = None
    no_change: bool = False
    checkpoint: MaintenanceCheckpoint = MaintenanceCheckpoint.MAINTENANCE_COMPLETE


class MaintenanceStateStore(Protocol):
    def load(self) -> dict[str, Any]: ...

    def save(self, state: dict[str, Any]) -> None: ...


class JSONMaintenanceStateStore:
    """Small crash-safe state store; source text and secret values are never stored."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "repositories": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)


class RepositoryMaintenanceService:
    """Run one resumable maintenance cycle without creating a second scheduler."""

    def __init__(
        self,
        intelligence: RepositoryIntelligenceService,
        store: MaintenanceStateStore,
        *,
        git_workflow: AutonomousGitWorkflow | None = None,
        memory: MemoryService | None = None,
        max_files: int = 500,
        max_attempts_per_finding: int = 2,
    ) -> None:
        self._intelligence = intelligence
        self._store = store
        self._workflow = git_workflow
        self._memory = memory
        self._max_files = max(1, max_files)
        self._max_attempts = max(1, max_attempts_per_finding)

    async def run(self, request: MaintenanceRequest) -> MaintenanceSummary:
        root = repo_root(request.repository).resolve()
        branch = current_branch(root)
        sha = run_git(["rev-parse", "HEAD"], root)
        key = self._repository_key(request.repository_id, root, branch)
        state = self._store.load()
        repositories = state.setdefault("repositories", {})
        record = repositories.setdefault(key, self._new_record(request, root, branch))
        previous_sha = record.get("baseline", {}).get("sha")
        previous_checkpoint = record.get("checkpoint")
        run_id = uuid.uuid4().hex
        summary = MaintenanceSummary(run_id, request.repository_id, previous_sha, sha)

        if previous_sha == sha and previous_checkpoint == "maintenance_complete":
            summary.no_change = True
            logger.info("maintenance_no_change", maintenance_run_id=run_id, current_sha=sha)
            return summary
        self._checkpoint(state, record, run_id, MaintenanceCheckpoint.BASELINE_LOADED)

        changed = self._changed_files(root, previous_sha, sha)[: self._max_files]
        summary.changed_files = tuple(changed)
        self._checkpoint(state, record, run_id, MaintenanceCheckpoint.CHANGES_DETECTED)
        test_target = self._impacted_test_target(root, changed)
        snapshot = await self._intelligence.scan(
            request.repository_id,
            root,
            incremental=previous_sha is not None,
            run_tests=request.run_tests and self._requires_tests(changed),
            test_target=test_target,
            ci_failures=request.ci_failures,
        )
        current = {self.finding_fingerprint(item): item for item in snapshot.findings}
        self._checkpoint(state, record, run_id, MaintenanceCheckpoint.SCAN_COMPLETE)
        transitions = self._update_backlog(record, current, sha)
        summary.transitions = dict(transitions)
        backlog = record["backlog"]
        summary.auto_repair_candidates = sum(
            item["status"] == "active" and item["decision"] == MaintenanceDecision.AUTO_REPAIR
            for item in backlog.values()
        )
        summary.human_review = sum(
            item["status"] == "active"
            and item["decision"] in {MaintenanceDecision.HUMAN_REVIEW, MaintenanceDecision.BLOCKING}
            for item in backlog.values()
        )
        self._checkpoint(state, record, run_id, MaintenanceCheckpoint.BACKLOG_UPDATED)

        if self._workflow is not None:
            candidates = self._repair_candidates(record, current, sha, request.max_repairs)
            for fingerprint, finding in candidates:
                await self._repair(state, record, run_id, request, finding, fingerprint, summary)

        record["baseline"] = {
            "sha": run_git(["rev-parse", "HEAD"], root),
            "branch": branch,
            "scanned_at": time.time(),
            "finding_ids": sorted(current),
            "ci_status": request.ci_status,
            "files_processed": len(snapshot.files_processed),
        }
        record.setdefault("health_history", []).append(
            {
                "sha": record["baseline"]["sha"],
                "validated_findings": len(snapshot.validated_findings),
                "ci_status": request.ci_status,
                "reopened": transitions.get(FindingTransition.REOPENED, 0),
            }
        )
        record["health_history"] = record["health_history"][-50:]
        summary.final_ci = request.ci_status
        self._checkpoint(state, record, run_id, MaintenanceCheckpoint.MAINTENANCE_COMPLETE)
        await self._remember(summary)
        logger.info("maintenance_completed", **asdict(summary))
        return summary

    @staticmethod
    async def schedule(scheduler: TaskScheduler, request: MaintenanceRequest) -> ScheduledTask:
        task = ScheduledTask(
            name=f"Repository maintenance: {request.repository_id}",
            agent_type="repository",
            input_data={
                "operation": "maintenance",
                "repository_id": request.repository_id,
                "repository": str(request.repository),
            },
            max_retries=1,
        )
        await scheduler.add_task(task)
        return task

    @staticmethod
    def finding_fingerprint(finding: RepositoryFinding) -> str:
        identity = {
            "category": finding.category,
            "files": sorted(finding.affected_files),
            "symbols": sorted(finding.affected_symbols),
            "sources": sorted(finding.source_detectors),
            "evidence": sorted((e.kind, e.file_path, e.symbol) for e in finding.evidence),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]

    @staticmethod
    def evidence_digest(finding: RepositoryFinding) -> str:
        evidence = [item.to_dict() for item in finding.evidence]
        return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()

    def dismiss(self, repository_id: str, repository: Path, fingerprint: str, reason: str) -> None:
        root = repo_root(repository).resolve()
        key = self._repository_key(repository_id, root, current_branch(root))
        state = self._store.load()
        item = state["repositories"][key]["backlog"][fingerprint]
        item.update(status="dismissed", dismissal_reason=reason, dismissed_digest=item["digest"])
        self._store.save(state)

    def _update_backlog(
        self, record: dict[str, Any], current: dict[str, RepositoryFinding], sha: str
    ) -> dict[str, int]:
        backlog = record.setdefault("backlog", {})
        counts: dict[str, int] = {}
        for fingerprint, finding in current.items():
            digest = self.evidence_digest(finding)
            old = backlog.get(fingerprint)
            if old and old.get("status") == "dismissed" and old.get("dismissed_digest") == digest:
                transition = FindingTransition.UNCHANGED
                decision = MaintenanceDecision.OBSERVE
                status = "dismissed"
            elif old and old.get("status") in {"resolved", "fixed", "dismissed"}:
                transition = FindingTransition.REOPENED
                decision = self._decision(finding)
                status = "active"
            elif old:
                transition = (
                    FindingTransition.REGRESSED
                    if old.get("digest") != digest
                    else FindingTransition.UNCHANGED
                )
                decision = self._decision(finding)
                status = "active"
            else:
                transition = FindingTransition.NEW_FINDING
                decision = self._decision(finding)
                status = "active"
            recurrence = int((old or {}).get("recurrence_count", 0))
            if transition is FindingTransition.REOPENED:
                recurrence += 1
            backlog[fingerprint] = {
                **(old or {}),
                "finding_id": fingerprint,
                "category": finding.category,
                "title": finding.title,
                "affected_files": list(finding.affected_files),
                "severity": finding.severity.value,
                "confidence": finding.confidence.value,
                "priority": finding.priority.value,
                "digest": digest,
                "status": status,
                "transition": transition.value,
                "decision": decision.value,
                "recurrence_count": recurrence,
                "last_seen_sha": sha,
            }
            counts[transition.value] = counts.get(transition.value, 0) + 1
        for fingerprint, item in backlog.items():
            if fingerprint not in current and item.get("status") == "active":
                item.update(status="resolved", transition="resolved", resolved_sha=sha)
                key = FindingTransition.RESOLVED.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _decision(finding: RepositoryFinding) -> MaintenanceDecision:
        if finding.severity is FindingSeverity.CRITICAL:
            return MaintenanceDecision.BLOCKING
        if finding.auto_fix_eligible and not any(
            word in finding.category.lower() for word in ("security", "architecture", "api")
        ):
            return MaintenanceDecision.AUTO_REPAIR
        if (
            finding.status is FindingStatus.VALIDATED
            and finding.confidence is not FindingConfidence.LOW
        ):
            return MaintenanceDecision.HUMAN_REVIEW
        return MaintenanceDecision.OBSERVE

    def _repair_candidates(
        self,
        record: dict[str, Any],
        current: dict[str, RepositoryFinding],
        sha: str,
        budget: int,
    ) -> list[tuple[str, RepositoryFinding]]:
        ranked = sorted(
            current.items(),
            key=lambda pair: (
                -int(record["backlog"][pair[0]].get("recurrence_count", 0)),
                -pair[1].impact_score,
                pair[0],
            ),
        )
        selected = []
        for fingerprint, finding in ranked:
            item = record["backlog"][fingerprint]
            signature = self._attempt_signature(fingerprint, sha, item["digest"])
            attempts = record.setdefault("attempts", {}).get(signature, [])
            if item["decision"] != "auto_repair" or len(attempts) >= self._max_attempts:
                continue
            selected.append((fingerprint, finding))
            if len(selected) >= max(0, budget):
                break
        return selected

    async def _repair(
        self,
        state: dict[str, Any],
        record: dict[str, Any],
        run_id: str,
        request: MaintenanceRequest,
        finding: RepositoryFinding,
        fingerprint: str,
        summary: MaintenanceSummary,
    ) -> None:
        assert self._workflow is not None
        sha = run_git(["rev-parse", "HEAD"], request.repository)
        signature = self._attempt_signature(fingerprint, sha, self.evidence_digest(finding))
        attempts = record.setdefault("attempts", {}).setdefault(signature, [])
        self._checkpoint(state, record, run_id, MaintenanceCheckpoint.REPAIR_STARTED)
        summary.repairs_attempted += 1
        task = self._intelligence.propose_task(finding)
        try:
            result = await self._workflow.run(
                GitWorkflowRequest(
                    task_id=str(task.id),
                    title=task.title,
                    goal=task.description or task.title,
                    repository=request.repository,
                    intended_paths=finding.affected_files,
                    default_branch=request.default_branch,
                    github_owner=request.github_owner,
                    github_repo=request.github_repo,
                    publish_remote=request.publish_remote,
                )
            )
        except Exception as exc:
            attempts.append({"run_id": run_id, "success": False, "error": type(exc).__name__})
            record["backlog"][fingerprint]["last_repair"] = attempts[-1]
            self._checkpoint(state, record, run_id, MaintenanceCheckpoint.FAILED)
            raise
        success = result.status in {
            GitWorkflowStatus.COMMITTED,
            GitWorkflowStatus.READY_FOR_REVIEW,
        }
        attempts.append({"run_id": run_id, "success": success, "error": result.error})
        item = record["backlog"][fingerprint]
        item["last_repair"] = attempts[-1]
        if success:
            item.update(status="fixed", resolved_sha=result.commit_sha)
            summary.repairs_successful += 1
            self._checkpoint(state, record, run_id, MaintenanceCheckpoint.REPAIR_VERIFIED)
        else:
            self._store.save(state)

    def _checkpoint(
        self,
        state: dict[str, Any],
        record: dict[str, Any],
        run_id: str,
        checkpoint: MaintenanceCheckpoint,
    ) -> None:
        record.update(checkpoint=checkpoint.value, active_run_id=run_id, updated_at=time.time())
        self._store.save(state)

    async def _remember(self, summary: MaintenanceSummary) -> None:
        if self._memory is None:
            return
        await self._memory.add(
            "Repository maintenance outcome",
            MemoryType.EPISODIC,
            tags=["repository-maintenance", summary.repository_id],
            metadata={
                "repository_id": summary.repository_id,
                "transitions": summary.transitions,
                "repairs_attempted": summary.repairs_attempted,
                "repairs_successful": summary.repairs_successful,
                "final_ci": summary.final_ci,
            },
        )

    @staticmethod
    def _new_record(request: MaintenanceRequest, root: Path, branch: str) -> dict[str, Any]:
        return {
            "repository_id": request.repository_id,
            "root": str(root),
            "branch": branch,
            "baseline": {},
            "backlog": {},
            "attempts": {},
            "health_history": [],
        }

    @staticmethod
    def _repository_key(repository_id: str, root: Path, branch: str) -> str:
        raw = f"{repository_id}\0{str(root).casefold()}\0{branch}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _changed_files(root: Path, previous_sha: str | None, sha: str) -> list[str]:
        if previous_sha is None:
            output = run_git(["ls-files"], root)
        else:
            try:
                run_git(["merge-base", "--is-ancestor", previous_sha, sha], root)
                output = run_git(["diff", "--name-only", f"{previous_sha}..{sha}"], root)
            except GitOperationError:
                output = run_git(["ls-files"], root)
        return sorted(path for path in output.splitlines() if path)

    @staticmethod
    def _impacted_test_target(root: Path, changed: list[str]) -> str:
        candidates: list[str] = []
        for path in changed:
            source = Path(path)
            if source.suffix != ".py" or source.parts[0] == "tests":
                continue
            test = Path("tests") / source.parent / f"test_{source.name}"
            if (root / test).is_file():
                candidates.append(test.as_posix())
        return " ".join(candidates) if candidates else "tests"

    @staticmethod
    def _requires_tests(changed: list[str]) -> bool:
        return any(Path(path).suffix in {".py", ".pyi", ".toml"} for path in changed)

    @staticmethod
    def _attempt_signature(fingerprint: str, sha: str, digest: str) -> str:
        return hashlib.sha256(f"{fingerprint}\0{sha}\0{digest}".encode()).hexdigest()


__all__ = [
    "FindingTransition",
    "JSONMaintenanceStateStore",
    "MaintenanceCheckpoint",
    "MaintenanceDecision",
    "MaintenanceRequest",
    "MaintenanceSummary",
    "RepositoryMaintenanceService",
]
