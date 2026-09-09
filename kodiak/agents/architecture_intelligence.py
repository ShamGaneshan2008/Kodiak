"""Evidence-first architecture analysis built on Kodiak's repository index and graph."""

from __future__ import annotations

import enum
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import structlog

from kodiak.agents.repository_intelligence import (
    FindingConfidence,
    FindingEvidence,
    FindingSeverity,
    FindingStatus,
    RepositoryFinding,
)
from kodiak.rag.dependency_graph import DependencyGraph
from kodiak.rag.repository_index import ModuleInfo, RepositoryIndex, RepositoryIndexer

logger = structlog.get_logger(__name__)


class ArchitectureRisk(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ArchitectureRule:
    """An explicit forbidden dependency direction; rules are never inferred."""

    name: str
    source_prefix: str
    forbidden_target_prefix: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ArchitectureMetric:
    module: str
    path: str
    dependencies: int
    dependents: int
    transitive_dependents: int
    line_count: int
    instability: float
    coupling_score: int
    cross_package_dependencies: int = 0
    public_symbols: int = 0
    risk: ArchitectureRisk = ArchitectureRisk.LOW
    risk_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImpactPath:
    target: str
    affected: str
    modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChangeImpact:
    targets: tuple[str, ...]
    direct_dependents: tuple[str, ...]
    transitive_dependents: tuple[str, ...]
    paths: tuple[ImpactPath, ...]
    relevant_tests: tuple[str, ...]
    likely_affected_apis: tuple[str, ...]
    risk: ArchitectureRisk
    risk_reasons: tuple[str, ...]
    verification_scope: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationStage:
    stage_id: str
    title: str
    target_modules: tuple[str, ...]
    dependencies: tuple[str, ...]
    expected_result: str
    verification: tuple[str, ...]
    rollback_boundary: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    proposal_id: str
    stages: tuple[MigrationStage, ...]


@dataclass(frozen=True, slots=True)
class PlanValidation:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureFinding:
    finding_id: str
    category: str
    title: str
    module: str
    path: str
    evidence: tuple[str, ...]
    severity: ArchitectureRisk
    affected_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RefactoringProposal:
    proposal_id: str
    finding_id: str
    title: str
    objective: str
    target_modules: tuple[str, ...]
    target_paths: tuple[str, ...]
    verification_targets: tuple[str, ...]
    estimated_files: int
    risk: ArchitectureRisk
    constraints: tuple[str, ...] = ()
    problem: str = ""
    evidence: tuple[str, ...] = ()
    why_it_matters: str = ""
    target_architecture: str = ""
    migration_strategy: str = ""
    rollback_strategy: str = ""
    plan: MigrationPlan | None = None


@dataclass(frozen=True, slots=True)
class ArchitectureSnapshot:
    repository: str
    modules: int
    edges: int
    cycles: tuple[tuple[str, ...], ...]
    metrics: tuple[ArchitectureMetric, ...]
    findings: tuple[ArchitectureFinding, ...]
    proposals: tuple[RefactoringProposal, ...]
    graph: DependencyGraph | None = None
    index: RepositoryIndex | None = None
    repository_findings: tuple[RepositoryFinding, ...] = ()
    files_processed: tuple[str, ...] = ()
    files_unchanged: tuple[str, ...] = ()


class ArchitectureIntelligenceService:
    """Find architecture pressure without executing code or changing the repository."""

    def __init__(
        self,
        *,
        indexer: RepositoryIndexer | None = None,
        rules: tuple[ArchitectureRule, ...] = (),
        high_coupling_threshold: int = 6,
        large_module_lines: int = 500,
        max_proposals: int = 10,
    ) -> None:
        self._indexer = indexer or RepositoryIndexer()
        self._rules = rules
        self._coupling_threshold = max(2, high_coupling_threshold)
        self._large_module_lines = max(100, large_module_lines)
        self._max_proposals = max(1, max_proposals)
        self._cache: dict[str, tuple[dict[str, tuple[int, int]], RepositoryIndex]] = {}

    def analyze(self, repository: str | Path, *, incremental: bool = True) -> ArchitectureSnapshot:
        root = Path(repository).resolve()
        index, processed, unchanged = self._incremental_index(root, incremental)
        return self.analyze_index(index, files_processed=processed, files_unchanged=unchanged)

    def analyze_index(
        self,
        index: RepositoryIndex,
        *,
        files_processed: tuple[str, ...] = (),
        files_unchanged: tuple[str, ...] = (),
    ) -> ArchitectureSnapshot:
        """Analyze a canonical index directly, without a second repository scan."""
        graph = DependencyGraph.from_index(index)
        modules = {module.module_name: module for module in index.modules}
        metrics = tuple(
            sorted(
                (self._metric(module, graph) for module in index.modules),
                key=lambda item: (-item.coupling_score, item.module),
            )
        )
        cycles = tuple(tuple(cycle) for cycle in graph.detect_cycles())
        findings = self._findings(modules, graph, metrics, cycles)
        proposals = tuple(
            self._proposal(item, modules, graph) for item in findings[: self._max_proposals]
        )
        canonical = tuple(self._repository_finding(str(index.root_path), item) for item in findings)
        logger.info(
            "architecture_analysis_completed",
            repository=str(index.root_path),
            modules=index.module_count,
            edges=len(graph.edges),
            cycle_count=len(cycles),
        )
        return ArchitectureSnapshot(
            repository=str(index.root_path),
            modules=index.module_count,
            edges=len(graph.edges),
            cycles=cycles,
            metrics=metrics,
            findings=tuple(findings),
            proposals=proposals,
            graph=graph,
            index=index,
            repository_findings=canonical,
            files_processed=files_processed,
            files_unchanged=files_unchanged,
        )

    @staticmethod
    def predict_change_impact(snapshot: ArchitectureSnapshot, module: str) -> tuple[str, ...]:
        if snapshot.graph is None or module not in snapshot.graph.nodes:
            return ()
        return tuple(sorted({module, *snapshot.graph.get_all_dependents(module)}))

    def impact(
        self, snapshot: ArchitectureSnapshot, targets: str | tuple[str, ...]
    ) -> ChangeImpact:
        """Explain statically proven reverse-dependency impact and test scope."""
        if snapshot.graph is None or snapshot.index is None:
            raise ValueError("Snapshot does not contain graph evidence.")
        requested = (targets,) if isinstance(targets, str) else targets
        resolved = tuple(
            dict.fromkeys(
                module
                for target in requested
                if (module := self._resolve_target(snapshot.index, snapshot.graph, target))
            )
        )
        direct = {item for target in resolved for item in snapshot.graph.get_dependents(target)}
        transitive = {
            item for target in resolved for item in snapshot.graph.get_all_dependents(target)
        }
        paths = tuple(
            path
            for target in resolved
            for affected in sorted(transitive)
            if (path := self._impact_path(snapshot.graph, target, affected)) is not None
        )
        tests = self._test_targets(snapshot.index, set(resolved) | transitive)
        apis = tuple(
            sorted(
                f"{module.module_name}.{symbol}"
                for module in snapshot.index.modules
                if module.module_name in resolved
                for symbol in (
                    *(item.name for item in module.classes),
                    *(item.name for item in module.functions),
                )
            )
        )
        score = max(
            (item.coupling_score for item in snapshot.metrics if item.module in resolved),
            default=0,
        )
        risk = (
            ArchitectureRisk.CRITICAL
            if len(transitive) >= 20
            else ArchitectureRisk.HIGH
            if len(transitive) >= 6 or score >= 20
            else ArchitectureRisk.MEDIUM
            if transitive
            else ArchitectureRisk.LOW
        )
        reasons = (
            f"direct_dependents={len(direct)}",
            f"transitive_dependents={len(transitive)}",
            f"maximum_coupling_score={score}",
        )
        verification = tuple(
            dict.fromkeys((*tests, "pytest tests/unit", "ruff check kodiak tests"))
        )
        return ChangeImpact(
            resolved,
            tuple(sorted(direct)),
            tuple(sorted(transitive)),
            paths,
            tests,
            apis,
            risk,
            reasons,
            verification,
        )

    def _metric(self, module: ModuleInfo, graph: DependencyGraph) -> ArchitectureMetric:
        dependency_names = graph.get_dependencies(module.module_name)
        dependencies = len(dependency_names)
        dependents = len(graph.get_dependents(module.module_name))
        transitive = len(graph.get_all_dependents(module.module_name))
        package = module.module_name.split(".", 1)[0]
        cross_package = sum(name.split(".", 1)[0] != package for name in dependency_names)
        total = dependencies + dependents
        instability = dependencies / total if total else 0.0
        score = (
            dependents * 3
            + dependencies * 2
            + transitive
            + cross_package * 2
            + module.line_count // 100
        )
        risk = (
            ArchitectureRisk.CRITICAL
            if score >= 30
            else ArchitectureRisk.HIGH
            if score >= 18
            else ArchitectureRisk.MEDIUM
            if score >= 8
            else ArchitectureRisk.LOW
        )
        return ArchitectureMetric(
            module=module.module_name,
            path=module.relative_path.as_posix(),
            dependencies=dependencies,
            dependents=dependents,
            transitive_dependents=transitive,
            line_count=module.line_count,
            instability=round(instability, 3),
            coupling_score=score,
            cross_package_dependencies=cross_package,
            public_symbols=len(module.functions) + len(module.classes),
            risk=risk,
            risk_reasons=(
                f"fan_in={dependents} (x3)",
                f"fan_out={dependencies} (x2)",
                f"transitive_dependents={transitive}",
                f"cross_package_dependencies={cross_package} (x2)",
                f"line_count_component={module.line_count // 100}",
            ),
        )

    def _findings(
        self,
        modules: dict[str, ModuleInfo],
        graph: DependencyGraph,
        metrics: tuple[ArchitectureMetric, ...],
        cycles: tuple[tuple[str, ...], ...],
    ) -> list[ArchitectureFinding]:
        findings: list[ArchitectureFinding] = []
        for cycle in cycles:
            body = tuple(dict.fromkeys(cycle))
            module = body[0]
            findings.append(
                self._finding(
                    "dependency_cycle",
                    f"Break dependency cycle involving {module}",
                    module,
                    modules[module].relative_path.as_posix(),
                    (f"Cycle: {' -> '.join(cycle)}",),
                    ArchitectureRisk.HIGH,
                    body,
                )
            )
        for rule in self._rules:
            for edge in graph.edges:
                if not (
                    edge.source.startswith(rule.source_prefix)
                    and edge.target.startswith(rule.forbidden_target_prefix)
                ):
                    continue
                findings.append(
                    self._finding(
                        "architecture_boundary",
                        f"Boundary violation: {edge.source} -> {edge.target}",
                        edge.source,
                        modules[edge.source].relative_path.as_posix(),
                        (
                            f"Explicit rule {rule.name}: {rule.rationale}",
                            f"{edge.source} -> {edge.target} via {edge.kind} at line {edge.line}",
                        ),
                        ArchitectureRisk.HIGH,
                        (edge.source, edge.target),
                    )
                )
        for metric in metrics:
            affected = tuple(sorted(graph.get_all_dependents(metric.module)))
            if metric.dependencies + metric.dependents >= self._coupling_threshold:
                severity = (
                    ArchitectureRisk.HIGH if metric.dependents >= 5 else ArchitectureRisk.MEDIUM
                )
                findings.append(
                    self._finding(
                        "coupling_hotspot",
                        f"Reduce coupling around {metric.module}",
                        metric.module,
                        metric.path,
                        (
                            f"Direct dependencies: {metric.dependencies}",
                            f"Direct dependents: {metric.dependents}",
                            f"Transitive dependents: {metric.transitive_dependents}",
                        ),
                        severity,
                        (metric.module, *affected),
                    )
                )
            if metric.line_count >= self._large_module_lines:
                findings.append(
                    self._finding(
                        "large_module",
                        f"Decompose large module {metric.module}",
                        metric.module,
                        metric.path,
                        (f"Module lines: {metric.line_count}",),
                        ArchitectureRisk.MEDIUM,
                        (metric.module, *affected),
                    )
                )
        severity_order = {
            ArchitectureRisk.CRITICAL: -1,
            ArchitectureRisk.HIGH: 0,
            ArchitectureRisk.MEDIUM: 1,
            ArchitectureRisk.LOW: 2,
        }
        return sorted(
            findings,
            key=lambda item: (severity_order[item.severity], item.category, item.module),
        )

    @staticmethod
    def _finding(
        category: str,
        title: str,
        module: str,
        path: str,
        evidence: tuple[str, ...],
        severity: ArchitectureRisk,
        affected: tuple[str, ...],
    ) -> ArchitectureFinding:
        identity = json.dumps(
            {"category": category, "module": module, "affected": sorted(affected)},
            sort_keys=True,
        )
        return ArchitectureFinding(
            finding_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
            category=category,
            title=title,
            module=module,
            path=path,
            evidence=evidence,
            severity=severity,
            affected_modules=tuple(sorted(set(affected))),
        )

    def _proposal(
        self,
        finding: ArchitectureFinding,
        modules: dict[str, ModuleInfo],
        graph: DependencyGraph,
    ) -> RefactoringProposal:
        affected_modules = tuple(
            sorted(
                set(finding.affected_modules)
                | {
                    dependent
                    for module in finding.affected_modules
                    for dependent in graph.get_all_dependents(module)
                }
            )
        )
        paths = tuple(
            sorted(
                modules[name].relative_path.as_posix()
                for name in affected_modules
                if name in modules
            )
        )
        tests = tuple(sorted({f"tests/test_{Path(path).stem}.py" for path in paths}))
        identity = f"{finding.finding_id}\0{'|'.join(paths)}"
        proposal_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        stages = self._stages(proposal_id, affected_modules, tests)
        return RefactoringProposal(
            proposal_id=proposal_id,
            finding_id=finding.finding_id,
            title=finding.title,
            objective=(
                f"Resolve {finding.category} in {finding.module} while preserving public behavior. "
                f"Evidence: {'; '.join(finding.evidence)}"
            ),
            target_modules=affected_modules,
            target_paths=paths,
            verification_targets=tests,
            estimated_files=len(paths),
            risk=finding.severity,
            constraints=(
                "Preserve public APIs unless separately approved.",
                "Keep migration steps independently verifiable.",
                "Do not modify unrelated files.",
            ),
            problem=finding.title,
            evidence=finding.evidence,
            why_it_matters="The measured dependency pressure increases coordinated-change risk.",
            target_architecture=(
                "Introduce the smallest compatibility-preserving boundary that removes the "
                "cited dependency pressure."
            ),
            migration_strategy="Introduce boundary, migrate consumers, then remove compatibility.",
            rollback_strategy="Stop after a failed stage and retain the prior verified boundary.",
            plan=MigrationPlan(proposal_id, stages),
        )

    def validate_plan(self, plan: MigrationPlan, impact: ChangeImpact) -> PlanValidation:
        errors: list[str] = []
        identifiers = {stage.stage_id for stage in plan.stages}
        covered = {module for stage in plan.stages for module in stage.target_modules}
        missing = set(impact.transitive_dependents) - covered
        if missing:
            errors.append(f"Plan omits impacted consumers: {', '.join(sorted(missing))}")
        for stage in plan.stages:
            unknown = set(stage.dependencies) - identifiers
            if unknown:
                errors.append(
                    f"Stage {stage.stage_id} has unknown dependencies: {', '.join(sorted(unknown))}"
                )
            if not stage.verification:
                errors.append(f"Stage {stage.stage_id} has no verification criteria.")
        if self._stage_cycle(plan.stages):
            errors.append("Migration stage dependencies contain a cycle.")
        return PlanValidation(not errors, tuple(errors))

    @staticmethod
    def compare(before: ArchitectureSnapshot, after: ArchitectureSnapshot) -> dict[str, object]:
        """Return measured architecture drift; functional verification remains separate."""
        return {
            "cycles_before": len(before.cycles),
            "cycles_after": len(after.cycles),
            "edges_before": before.edges,
            "edges_after": after.edges,
            "resolved_cycles": tuple(cycle for cycle in before.cycles if cycle not in after.cycles),
            "new_cycles": tuple(cycle for cycle in after.cycles if cycle not in before.cycles),
            "improved": len(after.cycles) < len(before.cycles) or after.edges < before.edges,
        }

    def _incremental_index(
        self, root: Path, incremental: bool
    ) -> tuple[RepositoryIndex, tuple[str, ...], tuple[str, ...]]:
        paths = self._indexer.iter_python_files(root)
        fingerprints = {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in paths}
        cached = self._cache.get(str(root)) if incremental else None
        old_fingerprints, old_index = cached if cached else ({}, None)
        old_modules = {str(item.path): item for item in old_index.modules} if old_index else {}
        modules: list[ModuleInfo] = []
        processed: list[str] = []
        unchanged: list[str] = []
        for path in paths:
            key = str(path)
            relative = path.relative_to(root).as_posix()
            if old_fingerprints.get(key) == fingerprints[key] and key in old_modules:
                modules.append(old_modules[key])
                unchanged.append(relative)
            else:
                modules.append(self._indexer.index_file(path, root))
                processed.append(relative)
        modules.sort(key=lambda item: item.relative_path.as_posix())
        index = RepositoryIndex(root, tuple(modules))
        self._cache[str(root)] = (fingerprints, index)
        return index, tuple(processed), tuple(unchanged)

    @staticmethod
    def _resolve_target(index: RepositoryIndex, graph: DependencyGraph, target: str) -> str | None:
        if target in graph.nodes:
            return target
        normalized = Path(target).as_posix()
        return next(
            (
                module.module_name
                for module in index.modules
                if module.relative_path.as_posix() == normalized
            ),
            None,
        )

    @staticmethod
    def _impact_path(graph: DependencyGraph, target: str, affected: str) -> ImpactPath | None:
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(target, (target,))])
        visited = {target}
        while queue:
            current, path = queue.popleft()
            if current == affected:
                return ImpactPath(target, affected, path)
            for dependent in sorted(graph.get_dependents(current)):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append((dependent, (*path, dependent)))
        return None

    @staticmethod
    def _test_targets(index: RepositoryIndex, impacted: set[str]) -> tuple[str, ...]:
        graph = DependencyGraph.from_index(index)
        stems = {name.split(".")[-1] for name in impacted}
        return tuple(
            sorted(
                module.relative_path.as_posix()
                for module in index.modules
                if (
                    "tests" in module.relative_path.parts
                    or module.relative_path.name.startswith("test_")
                )
                and (
                    module.relative_path.stem.removeprefix("test_") in stems
                    or bool(graph.get_dependencies(module.module_name) & impacted)
                )
            )
        )

    @staticmethod
    def _stages(
        proposal_id: str, targets: tuple[str, ...], verification: tuple[str, ...]
    ) -> tuple[MigrationStage, ...]:
        checks = verification or ("pytest tests/unit",)
        introduce = f"{proposal_id}-introduce"
        migrate = f"{proposal_id}-migrate"
        cleanup = f"{proposal_id}-cleanup"
        return (
            MigrationStage(
                introduce,
                "Introduce compatibility boundary",
                targets,
                (),
                "New boundary exists without caller breakage.",
                checks,
                "Revert this stage.",
            ),
            MigrationStage(
                migrate,
                "Migrate impacted consumers",
                targets,
                (introduce,),
                "Known consumers use the boundary.",
                checks,
                "Revert this stage and keep the boundary.",
            ),
            MigrationStage(
                cleanup,
                "Remove obsolete dependency and re-analyze",
                targets,
                (migrate,),
                "Finding is absent and functional checks pass.",
                checks,
                "Revert cleanup and retain compatibility.",
            ),
        )

    @staticmethod
    def _stage_cycle(stages: tuple[MigrationStage, ...]) -> bool:
        remaining = {stage.stage_id: set(stage.dependencies) for stage in stages}
        while remaining:
            ready = {stage for stage, dependencies in remaining.items() if not dependencies}
            if not ready:
                return True
            remaining = {
                stage: dependencies - ready
                for stage, dependencies in remaining.items()
                if stage not in ready
            }
        return False

    @staticmethod
    def _repository_finding(repository: str, finding: ArchitectureFinding) -> RepositoryFinding:
        severity = {
            ArchitectureRisk.CRITICAL: FindingSeverity.CRITICAL,
            ArchitectureRisk.HIGH: FindingSeverity.HIGH,
            ArchitectureRisk.MEDIUM: FindingSeverity.MEDIUM,
            ArchitectureRisk.LOW: FindingSeverity.LOW,
            ArchitectureRisk.BLOCKED: FindingSeverity.CRITICAL,
        }[finding.severity]
        return RepositoryFinding(
            repository_id=repository,
            category=finding.category,
            title=finding.title,
            description="Evidence-backed architecture debt from the canonical repository graph.",
            evidence=tuple(
                FindingEvidence(
                    "dependency_graph", "architecture_intelligence", evidence, finding.path
                )
                for evidence in finding.evidence
            ),
            affected_files=(finding.path,),
            severity=severity,
            confidence=FindingConfidence.HIGH,
            status=FindingStatus.VALIDATED,
            impact_score={
                FindingSeverity.CRITICAL: 28,
                FindingSeverity.HIGH: 22,
                FindingSeverity.MEDIUM: 15,
                FindingSeverity.LOW: 8,
                FindingSeverity.INFO: 2,
            }[severity],
            impact_explanation=(
                f"severity={severity.value}",
                f"affected_modules={len(finding.affected_modules)}",
            ),
            reproducibility="static_evidence",
            source_detectors=("architecture_intelligence",),
            suggested_verification="Run impacted tests and full repository verification.",
            finding_id=finding.finding_id,
        )


__all__ = [
    "ArchitectureFinding",
    "ArchitectureIntelligenceService",
    "ArchitectureMetric",
    "ArchitectureRisk",
    "ArchitectureRule",
    "ArchitectureSnapshot",
    "ChangeImpact",
    "ImpactPath",
    "MigrationPlan",
    "MigrationStage",
    "PlanValidation",
    "RefactoringProposal",
]
