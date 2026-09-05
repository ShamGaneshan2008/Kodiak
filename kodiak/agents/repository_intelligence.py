"""Evidence-based proactive issue discovery for the repository agent domain."""

from __future__ import annotations

import ast
import asyncio
import enum
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import structlog

from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.memory.service import MemoryService
from kodiak.orchestration.approval_gate import ApprovalGate, ApprovalStatus
from kodiak.rag.dependency_graph import DependencyGraph
from kodiak.rag.indexer import FileHashTracker
from kodiak.rag.repository_index import ModuleInfo, RepositoryIndex, RepositoryIndexer
from kodiak.security.secrets import SecretManager
from kodiak.tools.models import ToolExecutionContext
from kodiak.tools.router import ToolRouter
from kodiak.utils.git_utils import GitOperationError, run_git

logger = structlog.get_logger(__name__)


class FindingSeverity(enum.StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingConfidence(enum.StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingStatus(enum.StrEnum):
    NEW = "new"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DISMISSED = "dismissed"
    DUPLICATE = "duplicate"
    ACCEPTED = "accepted"
    FIXED = "fixed"


class EffortClass(enum.StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class FindingEvidence:
    kind: str
    source: str
    summary: str
    file_path: str | None = None
    symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "file_path": self.file_path,
            "symbol": self.symbol,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RepositoryFinding:
    repository_id: str
    category: str
    title: str
    description: str
    evidence: tuple[FindingEvidence, ...]
    affected_files: tuple[str, ...]
    affected_symbols: tuple[str, ...] = ()
    severity: FindingSeverity = FindingSeverity.LOW
    confidence: FindingConfidence = FindingConfidence.LOW
    status: FindingStatus = FindingStatus.NEW
    impact_score: int = 0
    impact_explanation: tuple[str, ...] = ()
    effort: EffortClass = EffortClass.SMALL
    priority: TaskPriority = TaskPriority.LOW
    reproducibility: str = "not_attempted"
    source_detectors: tuple[str, ...] = ()
    suggested_verification: str = "Review the cited repository evidence."
    suggested_next_action: str = "human_review"
    duplicate_of: str | None = None
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("Repository findings require concrete evidence.")

    def dismiss(self, reason: str) -> None:
        self.status = FindingStatus.DISMISSED
        self.reproducibility = f"dismissed: {reason}"

    @property
    def auto_fix_eligible(self) -> bool:
        return (
            self.status is FindingStatus.VALIDATED
            and self.confidence is FindingConfidence.HIGH
            and self.severity in {FindingSeverity.LOW, FindingSeverity.MEDIUM}
            and self.effort is EffortClass.SMALL
            and self.category in {"test_failure", "static_bug"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "repository_id": self.repository_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "evidence": [item.to_dict() for item in self.evidence],
            "affected_files": list(self.affected_files),
            "affected_symbols": list(self.affected_symbols),
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "status": self.status.value,
            "impact_score": self.impact_score,
            "impact_explanation": list(self.impact_explanation),
            "effort": self.effort.value,
            "priority": self.priority.value,
            "reproducibility": self.reproducibility,
            "source_detectors": list(self.source_detectors),
            "suggested_verification": self.suggested_verification,
            "suggested_next_action": self.suggested_next_action,
            "auto_fix_eligible": self.auto_fix_eligible,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(frozen=True, slots=True)
class RepositoryHealthSnapshot:
    repository_id: str
    scan_id: str
    findings: tuple[RepositoryFinding, ...]
    files_considered: int
    files_processed: tuple[str, ...]
    files_unchanged: tuple[str, ...]
    dimensions: dict[str, dict[str, int]]
    duration_seconds: float

    @property
    def validated_findings(self) -> tuple[RepositoryFinding, ...]:
        return tuple(item for item in self.findings if item.status is FindingStatus.VALIDATED)


class GitHubIssueClient(Protocol):
    async def list_issues(
        self, owner: str, repo: str, state: str = "open", per_page: int = 30
    ) -> list[dict[str, Any]]: ...

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any]: ...


class RepositoryIntelligenceService:
    """Extend repository analysis with bounded, evidence-backed findings."""

    def __init__(
        self,
        *,
        repository_indexer: RepositoryIndexer | None = None,
        tool_router: ToolRouter | None = None,
        memory: MemoryService | None = None,
        approval_gate: ApprovalGate | None = None,
        secret_manager: SecretManager | None = None,
        max_findings: int = 50,
        max_remote_issues: int = 3,
        hotspot_commit_threshold: int = 3,
        complexity_threshold: int = 8,
    ) -> None:
        self._indexer = repository_indexer or RepositoryIndexer()
        self._tools = tool_router
        self._memory = memory
        self._approval = approval_gate or ApprovalGate()
        self._secrets = secret_manager or SecretManager()
        self._max_findings = max(1, max_findings)
        self._max_remote_issues = max(1, max_remote_issues)
        self._hotspot_threshold = max(2, hotspot_commit_threshold)
        self._complexity_threshold = max(2, complexity_threshold)
        self._hashes = FileHashTracker()
        self._module_cache: dict[str, dict[str, ModuleInfo]] = {}
        self._static_cache: dict[str, dict[str, list[RepositoryFinding]]] = {}
        self._remote_issue_counts: Counter[str] = Counter()

    async def scan(
        self,
        repository_id: str,
        root_path: str | Path,
        *,
        incremental: bool = True,
        run_tests: bool = False,
        test_target: str = "tests",
        ci_failures: list[dict[str, Any]] | None = None,
    ) -> RepositoryHealthSnapshot:
        started = time.monotonic()
        scan_id = uuid.uuid4().hex
        root = await asyncio.to_thread(Path(root_path).resolve)
        logger.info("repo_scan_started", repository=repository_id, scan_id=scan_id)
        index, processed, unchanged = await asyncio.to_thread(
            self._build_index, repository_id, root, incremental
        )
        signals: list[RepositoryFinding] = []
        signals.extend(await self._static_signals(repository_id, root, index, processed))
        signals.extend(self._dependency_signals(repository_id, index))
        signals.extend(self._missing_test_signals(repository_id, index))
        signals.extend(self._git_history_signals(repository_id, root, index))
        if run_tests:
            test_signal = await self._test_signal(repository_id, root, test_target)
            if test_signal:
                signals.append(test_signal)
        signals.extend(await self._memory_signals(repository_id))
        signals.extend(self._ci_signals(repository_id, ci_failures or []))

        findings = self._deduplicate(signals)
        findings = self._correlate(findings)
        for finding in findings:
            self._validate_and_rank(finding, index)
        findings.sort(key=lambda item: (-item.impact_score, item.category, item.title))
        findings = findings[: self._max_findings]
        snapshot = RepositoryHealthSnapshot(
            repository_id=repository_id,
            scan_id=scan_id,
            findings=tuple(findings),
            files_considered=index.module_count,
            files_processed=tuple(sorted(processed)),
            files_unchanged=tuple(sorted(unchanged)),
            dimensions=self._health_dimensions(findings),
            duration_seconds=time.monotonic() - started,
        )
        logger.info(
            "repo_scan_completed",
            repository=repository_id,
            scan_id=scan_id,
            findings=len(findings),
            validated=len(snapshot.validated_findings),
            files_processed=len(processed),
        )
        return snapshot

    def propose_task(self, finding: RepositoryFinding) -> Task:
        if finding.status is not FindingStatus.VALIDATED:
            raise ValueError("Only validated findings can become engineering tasks.")
        evidence = "\n".join(f"- {item.summary}" for item in finding.evidence)
        description = (
            f"Problem\n{finding.description}\n\nEvidence\n{evidence}\n\n"
            f"Affected area\n{', '.join(finding.affected_files)}\n\n"
            f"Impact\n{'; '.join(finding.impact_explanation)}\n\n"
            f"Confidence\n{finding.confidence.value}\n\n"
            f"Success criteria\nResolve the finding and preserve existing behavior.\n\n"
            f"Suggested verification\n{finding.suggested_verification}\n\n"
            f"Estimated effort\n{finding.effort.value}"
        )
        return Task(
            id=str(uuid.uuid4()),
            repository_id=(
                finding.repository_id
                if _is_uuid(finding.repository_id)
                else str(uuid.uuid5(uuid.NAMESPACE_URL, finding.repository_id))
            ),
            title=finding.title,
            description=description,
            status=TaskStatus.PENDING,
            priority=finding.priority,
            source=TaskSource.SCHEDULED,
            source_ref=finding.finding_id,
            context={"finding": finding.to_dict(), "auto_fix_eligible": finding.auto_fix_eligible},
        )

    async def create_github_issue(
        self,
        finding: RepositoryFinding,
        *,
        client: GitHubIssueClient,
        owner: str,
        repo: str,
    ) -> dict[str, Any] | None:
        if finding.status is not FindingStatus.VALIDATED:
            raise ValueError("Only validated findings can be published.")
        if finding.confidence is not FindingConfidence.HIGH or finding.severity in {
            FindingSeverity.LOW,
            FindingSeverity.INFO,
        }:
            raise ValueError("Remote issues require high confidence and medium-or-higher severity.")
        repository_key = f"{owner}/{repo}"
        if self._remote_issue_counts[repository_key] >= self._max_remote_issues:
            raise RuntimeError("Remote issue creation quota exhausted for this service run.")
        existing = await client.list_issues(owner, repo, state="open", per_page=100)
        if any(_issue_matches(finding, issue) for issue in existing):
            logger.info("repo_finding_duplicate", finding_id=finding.finding_id)
            return None
        approval = await self._approval.request_approval(
            "create_issue",
            {"repository": repository_key, "finding_id": finding.finding_id},
        )
        if approval.status is not ApprovalStatus.APPROVED:
            return None
        task = self.propose_task(finding)
        body = task.description or ""
        issue = await client.create_issue(
            owner,
            repo,
            finding.title,
            body,
            labels=["kodiak", f"severity:{finding.severity.value}"],
        )
        self._remote_issue_counts[repository_key] += 1
        return issue

    def _build_index(
        self, repository_id: str, root: Path, incremental: bool
    ) -> tuple[RepositoryIndex, set[str], set[str]]:
        cache = self._module_cache.setdefault(repository_id, {})
        files = self._indexer.iter_python_files(root)
        processed: set[str] = set()
        unchanged: set[str] = set()
        current_paths = {path.relative_to(root).as_posix() for path in files}
        for stale in set(cache) - current_paths:
            cache.pop(stale, None)
        errors = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                changed = self._hashes.has_changed(repository_id, relative, content)
                if incremental and not changed and relative in cache:
                    unchanged.add(relative)
                    continue
                cache[relative] = self._indexer.index_file(path, root)
                processed.add(relative)
            except (OSError, SyntaxError, UnicodeError) as exc:
                from kodiak.rag.repository_index import IndexingError

                errors.append(IndexingError(path=path, message=str(exc)))
        modules = tuple(sorted(cache.values(), key=lambda item: item.relative_path.as_posix()))
        return (
            RepositoryIndex(root_path=root, modules=modules, errors=tuple(errors)),
            processed,
            unchanged,
        )

    async def _static_signals(
        self,
        repository_id: str,
        root: Path,
        index: RepositoryIndex,
        processed: set[str],
    ) -> list[RepositoryFinding]:
        findings: list[RepositoryFinding] = []
        cache = self._static_cache.setdefault(repository_id, {})
        current_paths = {module.relative_path.as_posix() for module in index.modules}
        for stale in set(cache) - current_paths:
            cache.pop(stale, None)
        for module in index.modules:
            path = module.relative_path.as_posix()
            if path not in processed:
                findings.extend(cache.get(path, []))
                continue
            start = len(findings)
            source = (root / module.relative_path).read_text(encoding="utf-8", errors="replace")
            redacted = await self._secrets.mask_secrets(source)
            tree = ast.parse(source, filename=path)
            for line_number, line in enumerate(redacted.splitlines(), start=1):
                marker = re.search(r"\b(TODO|FIXME)\b[: ]*(.*)", line, re.IGNORECASE)
                if marker:
                    summary = f"{marker.group(1).upper()} marker: {marker.group(2).strip()[:120]}"
                    findings.append(
                        self._finding(
                            repository_id,
                            "technical_debt",
                            f"Review {marker.group(1).upper()} in {path}",
                            "A source marker identifies deferred engineering work, "
                            "not a proven bug.",
                            FindingEvidence(
                                "code_pattern",
                                "todo_detector",
                                summary,
                                path,
                                line_start=line_number,
                            ),
                            severity=FindingSeverity.LOW,
                        )
                    )
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and (
                    node.type is None
                    or _exception_name(node.type) in {"Exception", "BaseException"}
                ):
                    is_empty = not node.body or all(
                        isinstance(item, ast.Pass) for item in node.body
                    )
                    findings.append(
                        self._finding(
                            repository_id,
                            "static_bug" if is_empty else "reliability",
                            f"Broad exception handler in {path}",
                            "A broad exception handler can hide unrelated failures.",
                            FindingEvidence(
                                "ast_pattern",
                                "broad_exception_detector",
                                "Bare or broad exception handler detected by Python AST.",
                                path,
                                line_start=node.lineno,
                                line_end=getattr(node, "end_lineno", node.lineno),
                                metadata={"empty_handler": is_empty},
                            ),
                            severity=FindingSeverity.MEDIUM if is_empty else FindingSeverity.LOW,
                        )
                    )
            for function in _all_functions(tree):
                complexity = _cyclomatic_complexity(function)
                if complexity > self._complexity_threshold:
                    findings.append(
                        self._finding(
                            repository_id,
                            "complexity",
                            f"High-complexity function {function.name}",
                            "Control-flow complexity exceeds the configured "
                            "deterministic threshold.",
                            FindingEvidence(
                                "metric",
                                "ast_complexity",
                                f"Cyclomatic complexity {complexity} exceeds "
                                f"{self._complexity_threshold}.",
                                path,
                                symbol=function.name,
                                line_start=function.lineno,
                                line_end=getattr(function, "end_lineno", function.lineno),
                                metadata={
                                    "complexity": complexity,
                                    "threshold": self._complexity_threshold,
                                },
                            ),
                            severity=FindingSeverity.MEDIUM,
                        )
                    )
            cache[path] = findings[start:]
        return findings

    def _dependency_signals(
        self, repository_id: str, index: RepositoryIndex
    ) -> list[RepositoryFinding]:
        graph = DependencyGraph.from_index(index)
        findings = []
        for cycle in graph.detect_cycles():
            files = tuple(
                graph.nodes[name].relative_path.as_posix()
                for name in cycle[:-1]
                if name in graph.nodes
            )
            findings.append(
                self._finding(
                    repository_id,
                    "dependency_cycle",
                    f"Dependency cycle: {' -> '.join(cycle)}",
                    "The structural repository index contains a deterministic import cycle.",
                    FindingEvidence(
                        "dependency_graph",
                        "dependency_graph",
                        f"Cycle detected across {len(files)} modules.",
                        metadata={"cycle": cycle},
                    ),
                    files=files,
                    severity=FindingSeverity.MEDIUM,
                )
            )
        return findings

    def _missing_test_signals(
        self, repository_id: str, index: RepositoryIndex
    ) -> list[RepositoryFinding]:
        test_modules = {
            module.relative_path.stem.removeprefix("test_")
            for module in index.modules
            if _is_test_path(module.relative_path)
        }
        findings = []
        for module in index.modules:
            if _is_test_path(module.relative_path) or not (module.functions or module.classes):
                continue
            if module.relative_path.stem in test_modules:
                continue
            path = module.relative_path.as_posix()
            findings.append(
                self._finding(
                    repository_id,
                    "testing_gap",
                    f"No focused test module for {path}",
                    "An important production module has no name-matched test module; "
                    "this is a coverage signal, not a bug.",
                    FindingEvidence(
                        "repository_structure",
                        "test_relationship",
                        f"No test_{module.relative_path.stem}.py module was indexed.",
                        path,
                        metadata={"public_symbols": len(module.functions) + len(module.classes)},
                    ),
                    severity=FindingSeverity.LOW,
                )
            )
        return findings

    def _git_history_signals(
        self, repository_id: str, root: Path, index: RepositoryIndex
    ) -> list[RepositoryFinding]:
        try:
            output = run_git(["log", "--format=", "--name-only", "--", "*.py"], root)
        except GitOperationError:
            return []
        counts = Counter(line.replace("\\", "/") for line in output.splitlines() if line.strip())
        indexed = {module.relative_path.as_posix() for module in index.modules}
        return [
            self._finding(
                repository_id,
                "hotspot",
                f"Frequently changed module {path}",
                "Git churn identifies a review hotspot, not a defect by itself.",
                FindingEvidence(
                    "git_history",
                    "git_history",
                    f"Changed in {count} commits in available history.",
                    path,
                    metadata={"commit_count": count},
                ),
                severity=FindingSeverity.INFO,
            )
            for path, count in counts.items()
            if path in indexed and count >= self._hotspot_threshold
        ]

    async def _test_signal(
        self, repository_id: str, root: Path, test_target: str
    ) -> RepositoryFinding | None:
        if self._tools is None or not self._tools.has_tool("test_runner"):
            return None
        result = await self._tools.execute(
            "test_runner",
            {"test_target": test_target, "options": ["-q", "-o", "cache_dir=.pytest_cache"]},
            ToolExecutionContext(agent_name="repository", timeout_seconds=30),
        )
        if result.success:
            return None
        output = str(result.output.get("stdout", "")) + str(result.output.get("stderr", ""))
        redacted = await self._secrets.mask_secrets(output)
        paths = tuple(dict.fromkeys(re.findall(r"([\w./\\-]+\.py)(?::\d+)?", redacted)))[:10]
        evidence = FindingEvidence(
            "test_failure",
            "test_runner",
            (result.error or "Tests failed") + ": " + redacted[-500:],
            file_path=paths[0].replace("\\", "/") if paths else None,
            metadata={"returncode": result.output.get("returncode"), "test_target": test_target},
        )
        return self._finding(
            repository_id,
            "test_failure",
            f"Deterministic test failure in {test_target}",
            "The configured test command reproduced a failure.",
            evidence,
            files=tuple(path.replace("\\", "/") for path in paths),
            severity=FindingSeverity.HIGH,
        )

    async def _memory_signals(self, repository_id: str) -> list[RepositoryFinding]:
        if self._memory is None:
            return []
        results = await self._memory.search(
            f"recurring failure repair verification {repository_id}", limit=5
        )
        findings = []
        for result in results:
            memory = result.memory
            metadata = dict(memory.metadata)
            if not metadata.get("failure_category") and "failure" not in memory.tags:
                continue
            summary = await self._secrets.mask_secrets(memory.content[:300])
            findings.append(
                self._finding(
                    repository_id,
                    "historical_risk",
                    f"Historical failure: {memory.title or 'recorded execution'}",
                    "Kodiak memory records a prior failure; corroboration is required.",
                    FindingEvidence(
                        "memory",
                        "memory_service",
                        summary,
                        metadata={"memory_id": str(memory.id)},
                    ),
                    severity=FindingSeverity.LOW,
                )
            )
        return findings

    def _ci_signals(
        self, repository_id: str, failures: list[dict[str, Any]]
    ) -> list[RepositoryFinding]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for failure in failures[:100]:
            grouped.setdefault(str(failure.get("name", "unknown")), []).append(failure)
        return [
            self._finding(
                repository_id,
                "ci_failure",
                f"Recurring CI failure: {name}",
                "The existing CI integration reported the same failed check repeatedly.",
                FindingEvidence(
                    "ci_failure",
                    "github_checks",
                    f"Check failed {len(items)} times.",
                    metadata={"check": name, "occurrences": len(items)},
                ),
                severity=FindingSeverity.HIGH,
            )
            for name, items in grouped.items()
            if len(items) >= 2
        ]

    def _deduplicate(self, findings: list[RepositoryFinding]) -> list[RepositoryFinding]:
        canonical: dict[tuple[str, str, str], RepositoryFinding] = {}
        for finding in findings:
            key = (
                finding.category,
                finding.affected_files[0] if finding.affected_files else "",
                _normalize_title(finding.title),
            )
            existing = canonical.get(key)
            if existing is None:
                canonical[key] = finding
                continue
            existing.evidence = tuple(
                {
                    (_evidence_key(item)): item for item in (*existing.evidence, *finding.evidence)
                }.values()
            )
            existing.source_detectors = tuple(
                sorted(set(existing.source_detectors) | set(finding.source_detectors))
            )
            finding.status = FindingStatus.DUPLICATE
            finding.duplicate_of = existing.finding_id
        return list(canonical.values())

    def _correlate(self, findings: list[RepositoryFinding]) -> list[RepositoryFinding]:
        by_file: dict[str, list[RepositoryFinding]] = {}
        for finding in findings:
            for path in finding.affected_files:
                by_file.setdefault(path, []).append(finding)
        correlated: list[RepositoryFinding] = []
        consumed: set[str] = set()
        for path, items in by_file.items():
            categories = {item.category for item in items}
            if {"hotspot", "complexity", "test_failure"}.issubset(categories):
                evidence = tuple(item.evidence[0] for item in items if item.category in categories)
                correlated.append(
                    RepositoryFinding(
                        repository_id=items[0].repository_id,
                        category="correlated_hotspot",
                        title=f"High-impact reliability hotspot in {path}",
                        description=(
                            "Git churn, complexity, and a reproduced test failure "
                            "converge on this module."
                        ),
                        evidence=evidence,
                        affected_files=(path,),
                        severity=FindingSeverity.HIGH,
                        source_detectors=tuple(
                            sorted({source for item in items for source in item.source_detectors})
                        ),
                        suggested_verification="Run the failing test after a focused repair.",
                    )
                )
                consumed.update(item.finding_id for item in items if item.category in categories)
        return [item for item in findings if item.finding_id not in consumed] + correlated

    def _validate_and_rank(self, finding: RepositoryFinding, index: RepositoryIndex) -> None:
        kinds = {item.kind for item in finding.evidence}
        deterministic = bool(
            kinds & {"test_failure", "dependency_graph"}
            or any(item.metadata.get("empty_handler") for item in finding.evidence)
            or finding.category == "correlated_hotspot"
        )
        if deterministic:
            finding.status = FindingStatus.VALIDATED
            finding.confidence = FindingConfidence.HIGH
            finding.reproducibility = "reproduced" if "test_failure" in kinds else "static_evidence"
        elif len(kinds) >= 2:
            finding.status = FindingStatus.VALIDATED
            finding.confidence = FindingConfidence.MEDIUM
            finding.reproducibility = "corroborated"
        else:
            finding.status = FindingStatus.NEW
            finding.confidence = (
                FindingConfidence.MEDIUM
                if kinds & {"ast_pattern", "metric", "git_history"}
                else FindingConfidence.LOW
            )
        finding.effort = _estimate_effort(finding)
        severity_points = {
            FindingSeverity.CRITICAL: 5,
            FindingSeverity.HIGH: 4,
            FindingSeverity.MEDIUM: 3,
            FindingSeverity.LOW: 2,
            FindingSeverity.INFO: 1,
        }[finding.severity]
        confidence_points = {
            FindingConfidence.HIGH: 3,
            FindingConfidence.MEDIUM: 2,
            FindingConfidence.LOW: 1,
        }[finding.confidence]
        graph = DependencyGraph.from_index(index)
        centrality = 0
        for path in finding.affected_files:
            module = next(
                (item for item in index.modules if item.relative_path.as_posix() == path), None
            )
            if module:
                centrality = max(centrality, len(graph.get_dependents(module.module_name)))
        recurrence = max(
            (
                int(item.metadata.get("commit_count", item.metadata.get("occurrences", 0)))
                for item in finding.evidence
            ),
            default=0,
        )
        finding.impact_score = (
            severity_points * 4 + confidence_points * 2 + min(centrality, 3) + min(recurrence, 3)
        )
        finding.impact_explanation = (
            f"severity={finding.severity.value} ({severity_points * 4})",
            f"confidence={finding.confidence.value} ({confidence_points * 2})",
            f"dependency_dependents={centrality} (capped at 3)",
            f"recurrence={recurrence} (capped at 3)",
        )
        effort_penalty = {EffortClass.SMALL: 0, EffortClass.MEDIUM: 3, EffortClass.LARGE: 6}[
            finding.effort
        ]
        priority_score = finding.impact_score - effort_penalty
        finding.priority = (
            TaskPriority.CRITICAL
            if priority_score >= 24
            else TaskPriority.HIGH
            if priority_score >= 19
            else TaskPriority.MEDIUM
            if priority_score >= 13
            else TaskPriority.LOW
        )
        finding.suggested_next_action = (
            "auto_fix_candidate" if finding.auto_fix_eligible else "human_review"
        )

    @staticmethod
    def _finding(
        repository_id: str,
        category: str,
        title: str,
        description: str,
        evidence: FindingEvidence,
        *,
        files: tuple[str, ...] = (),
        severity: FindingSeverity,
    ) -> RepositoryFinding:
        affected = files or ((evidence.file_path,) if evidence.file_path else ())
        return RepositoryFinding(
            repository_id=repository_id,
            category=category,
            title=title,
            description=description,
            evidence=(evidence,),
            affected_files=affected,
            affected_symbols=((evidence.symbol,) if evidence.symbol else ()),
            severity=severity,
            source_detectors=(evidence.source,),
        )

    @staticmethod
    def _health_dimensions(findings: list[RepositoryFinding]) -> dict[str, dict[str, int]]:
        dimensions: dict[str, dict[str, int]] = {}
        mapping = {
            "test_failure": "test_health",
            "testing_gap": "test_health",
            "ci_failure": "ci_health",
            "reliability": "reliability",
            "static_bug": "reliability",
            "complexity": "maintainability",
            "technical_debt": "technical_debt",
            "hotspot": "hotspots",
            "correlated_hotspot": "hotspots",
            "dependency_cycle": "architecture",
            "historical_risk": "reliability",
        }
        for finding in findings:
            dimension = mapping.get(finding.category, "maintainability")
            bucket = dimensions.setdefault(dimension, {"candidates": 0, "validated": 0})
            bucket["validated" if finding.status is FindingStatus.VALIDATED else "candidates"] += 1
        return dimensions


def _all_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _cyclomatic_complexity(node: ast.AST) -> int:
    branch_types = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.IfExp,
    )
    return (
        1
        + sum(isinstance(child, branch_types) for child in ast.walk(node))
        + sum(
            max(len(child.values) - 1, 0)
            for child in ast.walk(node)
            if isinstance(child, ast.BoolOp)
        )
    )


def _exception_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_test_path(path: Path) -> bool:
    return path.name.startswith("test_") or "tests" in path.parts


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _evidence_key(evidence: FindingEvidence) -> tuple[Any, ...]:
    return (
        evidence.kind,
        evidence.source,
        evidence.file_path,
        evidence.symbol,
        evidence.line_start,
    )


def _estimate_effort(finding: RepositoryFinding) -> EffortClass:
    if (
        finding.category in {"dependency_cycle", "correlated_hotspot"}
        or len(finding.affected_files) > 3
    ):
        return EffortClass.LARGE
    if len(finding.affected_files) > 1 or finding.category in {"complexity", "historical_risk"}:
        return EffortClass.MEDIUM
    return EffortClass.SMALL


def _issue_matches(finding: RepositoryFinding, issue: dict[str, Any]) -> bool:
    title = _normalize_title(str(issue.get("title", "")))
    finding_title = _normalize_title(finding.title)
    body = str(issue.get("body", "")).lower()
    title_tokens = set(finding_title.split())
    overlap = len(title_tokens & set(title.split())) / max(len(title_tokens), 1)
    file_match = any(path.lower() in body for path in finding.affected_files)
    category_match = finding.category.replace("_", " ") in body
    return title == finding_title or overlap >= 0.8 or (file_match and category_match)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


__all__ = [
    "EffortClass",
    "FindingConfidence",
    "FindingEvidence",
    "FindingSeverity",
    "FindingStatus",
    "RepositoryFinding",
    "RepositoryHealthSnapshot",
    "RepositoryIntelligenceService",
]
