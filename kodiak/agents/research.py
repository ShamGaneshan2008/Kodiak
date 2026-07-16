"""Research agent for Kodiak's autonomous engineering workflow.

The research agent is the second agent in the planner-driven workflow. It
consumes execution plans produced by :class:`kodiak.agents.planner.PlannerAgent`
and gathers repository knowledge for each planned task by delegating to
``ContextBuilder`` and ``SemanticSearch``.

This module never modifies repository files, generates code, executes code,
runs tests, commits changes, reviews diffs, calls LLMs, or duplicates RAG
logic. It only orchestrates existing repository-intelligence services and
returns structured research reports for later agents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - exercised only in minimal environments.
    import structlog
except ModuleNotFoundError:  # pragma: no cover
    import logging

    class _StructlogFallback:
        """Small fallback matching the structlog call shape used here."""

        @staticmethod
        def get_logger(name: str | None = None) -> logging.Logger:
            return logging.getLogger(name or __name__)

    structlog = _StructlogFallback()  # type: ignore[assignment]

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent
from kodiak.rag.context_builder import BuiltContext, ContextBuilder
from kodiak.rag.semantic_search import (
    SemanticSearch,
    SemanticSearchResponse,
    SemanticSearchResult,
)

logger = structlog.get_logger(__name__)


class ResearchFindingKind(str, Enum):
    """Kinds of evidence captured in a research report."""

    CONTEXT = "context"
    SYMBOL = "symbol"
    MODULE = "module"
    DEPENDENCY = "dependency"
    EXAMPLE = "example"
    PATTERN = "pattern"
    RISK = "risk"


@dataclass(frozen=True)
class PlannedTask:
    """Planner subtask normalized for research.

    Attributes:
        id: Stable planner subtask identifier.
        title: Short task title.
        description: Detailed task description.
        type: Planner task type, such as implementation, test, or review.
        complexity: Planner complexity estimate.
        depends_on: Planner task dependencies.
        likely_files: Files the planner expects the task to touch.
    """

    id: str
    title: str
    description: str
    type: str = "implementation"
    complexity: str = "medium"
    depends_on: tuple[str, ...] = ()
    likely_files: tuple[str, ...] = ()

    @property
    def search_text(self) -> str:
        """Return the natural-language query used for broad task research."""
        return " ".join(part for part in (self.title, self.description) if part).strip()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this planned task."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type,
            "complexity": self.complexity,
            "depends_on": list(self.depends_on),
            "likely_files": list(self.likely_files),
        }


@dataclass(frozen=True)
class ResearchEvidence:
    """A ranked repository evidence item discovered during research.

    Attributes:
        kind: Type of finding.
        title: Short human-readable finding title.
        summary: Deterministic summary of what was found.
        confidence: Confidence score from 0.0 to 1.0.
        relevance: Relevance score from 0.0 to 1.0.
        source: Search source or analysis source.
        file_path: Optional source file path.
        module_path: Optional module path.
        symbol_name: Optional symbol name.
        location: Optional file and line location.
        metadata: Additional structured metadata.
    """

    kind: ResearchFindingKind
    title: str
    summary: str
    confidence: float
    relevance: float
    source: str
    file_path: str | None = None
    module_path: str | None = None
    symbol_name: str | None = None
    location: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this evidence."""
        return {
            "kind": self.kind.value,
            "title": self.title,
            "summary": self.summary,
            "confidence": round(self.confidence, 6),
            "relevance": round(self.relevance, 6),
            "source": self.source,
            "file_path": self.file_path,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "location": self.location,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ResearchTaskReport:
    """Research output for one planner subtask.

    Attributes:
        task: Planner subtask being researched.
        context: ContextBuilder output for the task.
        findings: Ranked evidence gathered for the task.
        examples: Relevant implementation examples.
        related_modules: Modules found through search and dependency analysis.
        dependency_map: Dependency relationships relevant to the task.
        risks: Potential implementation risks inferred from plan and evidence.
        patterns: Existing repository patterns inferred from repeated evidence.
        confidence: Aggregate confidence for this task report.
    """

    task: PlannedTask
    context: BuiltContext
    findings: tuple[ResearchEvidence, ...]
    examples: tuple[ResearchEvidence, ...]
    related_modules: tuple[str, ...]
    dependency_map: dict[str, tuple[str, ...]]
    risks: tuple[ResearchEvidence, ...]
    patterns: tuple[ResearchEvidence, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this task report."""
        return {
            "task": self.task.to_dict(),
            "context": self.context.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "examples": [example.to_dict() for example in self.examples],
            "related_modules": list(self.related_modules),
            "dependency_map": {
                module: list(dependencies)
                for module, dependencies in self.dependency_map.items()
            },
            "risks": [risk.to_dict() for risk in self.risks],
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class ResearchReport:
    """Structured research report for an execution plan.

    Attributes:
        goal: Planner goal.
        acceptance_criteria: Planner acceptance criteria.
        task_reports: Per-subtask research reports.
        global_findings: Findings across the whole plan.
        related_modules: Unique modules related to the whole plan.
        risks: Plan-level risks.
        patterns: Plan-level implementation patterns.
        confidence: Aggregate report confidence.
        metadata: Additional structured report metadata.
    """

    goal: str
    acceptance_criteria: tuple[str, ...]
    task_reports: tuple[ResearchTaskReport, ...]
    global_findings: tuple[ResearchEvidence, ...]
    related_modules: tuple[str, ...]
    risks: tuple[ResearchEvidence, ...]
    patterns: tuple[ResearchEvidence, ...]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rag_context(self) -> str:
        """Return prompt-ready repository context for downstream agents."""
        return "\n\n".join(
            report.context.prompt_context
            for report in self.task_reports
            if report.context.prompt_context
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return {
            "goal": self.goal,
            "acceptance_criteria": list(self.acceptance_criteria),
            "task_reports": [report.to_dict() for report in self.task_reports],
            "global_findings": [finding.to_dict() for finding in self.global_findings],
            "related_modules": list(self.related_modules),
            "risks": [risk.to_dict() for risk in self.risks],
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "confidence": round(self.confidence, 6),
            "rag_context": self.rag_context,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ResearchConfig:
    """Configuration for deterministic research orchestration.

    Attributes:
        task_top_k: Search result limit for each planned task.
        symbol_top_k: Search result limit for symbol-focused searches.
        module_top_k: Search result limit for module/file searches.
        dependency_top_k: Search result limit for dependency-aware searches.
        example_top_k: Maximum implementation examples per task.
        max_context_tokens: ContextBuilder token budget override per task.
        min_evidence_confidence: Minimum confidence retained in reports.
        max_risks_per_task: Maximum risks retained per task.
    """

    task_top_k: int = 8
    symbol_top_k: int = 5
    module_top_k: int = 5
    dependency_top_k: int = 8
    example_top_k: int = 4
    max_context_tokens: int | None = None
    min_evidence_confidence: float = 0.0
    max_risks_per_task: int = 5

    def __post_init__(self) -> None:
        """Validate configuration values."""
        positive_fields = (
            "task_top_k",
            "symbol_top_k",
            "module_top_k",
            "dependency_top_k",
            "example_top_k",
            "max_risks_per_task",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero when provided")
        if not 0.0 <= self.min_evidence_confidence <= 1.0:
            raise ValueError("min_evidence_confidence must be between 0.0 and 1.0")


class ResearchAgent(BaseAgent):
    """Planner-aware repository research agent.

    Args:
        context_builder: Builder used to create prompt-ready repository context.
        semantic_search: Search service used for symbol, module, dependency, and
            example lookups. Defaults to ``context_builder.semantic_search``.
        config: Optional research configuration.
    """

    role = AgentRole.RESEARCH

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        semantic_search: SemanticSearch | None = None,
        config: ResearchConfig | None = None,
    ) -> None:
        """Initialize the research agent with injected RAG services."""
        super().__init__()
        self.context_builder = context_builder
        self.semantic_search = semantic_search or context_builder.semantic_search
        self.config = config or ResearchConfig()

    async def _run(self, input_: AgentInput) -> AgentOutput:
        """Run research for a planner output stored in ``AgentInput.context``."""
        plan = self._extract_plan(input_)
        if plan is None:
            return self._make_error(
                input_,
                "Planner output missing. Expected context['plan'] or "
                "context['planner_output']['plan'].",
            )

        report = await self.research(plan)
        return self._make_output(
            input_,
            result={"research": report.to_dict(), "rag_context": report.rag_context},
            metadata={
                "confidence": round(report.confidence, 6),
                "task_reports": len(report.task_reports),
                "related_modules": len(report.related_modules),
            },
        )

    async def research(self, plan: Mapping[str, Any] | Any) -> ResearchReport:
        """Research every task in a Planner Agent execution plan.

        Args:
            plan: Planner output as a dictionary or TaskPlan-like object.

        Returns:
            Structured plan-level research report.
        """
        goal, acceptance_criteria, tasks, metadata = self._normalize_plan(plan)
        task_reports = tuple(
            [
                await self.research_task(
                    task,
                    goal=goal,
                    acceptance_criteria=acceptance_criteria,
                )
                for task in tasks
            ]
        )
        global_findings = self._global_findings(goal, task_reports)
        risks = self._plan_risks(task_reports, metadata)
        patterns = self._plan_patterns(task_reports)
        related_modules = self._unique_modules(
            finding
            for report in task_reports
            for finding in (*report.findings, *report.examples)
        )
        confidence = self._aggregate_confidence(
            [
                report.confidence
                for report in task_reports
            ]
        )

        logger.info(
            "research.plan_complete",
            tasks=len(task_reports),
            findings=len(global_findings),
            risks=len(risks),
            confidence=round(confidence, 4),
        )
        return ResearchReport(
            goal=goal,
            acceptance_criteria=tuple(acceptance_criteria),
            task_reports=task_reports,
            global_findings=global_findings,
            related_modules=related_modules,
            risks=risks,
            patterns=patterns,
            confidence=confidence,
            metadata=metadata,
        )

    async def research_task(
        self,
        task: PlannedTask | Mapping[str, Any],
        *,
        goal: str = "",
        acceptance_criteria: Sequence[str] = (),
    ) -> ResearchTaskReport:
        """Gather repository research for one planned task.

        Args:
            task: Planner subtask as a normalized object or dictionary.
            goal: Parent plan goal.
            acceptance_criteria: Parent acceptance criteria.

        Returns:
            Structured research report for the task.
        """
        planned_task = task if isinstance(task, PlannedTask) else self._task_from_mapping(task)
        context = await self.collect_context(
            planned_task,
            goal=goal,
            acceptance_criteria=acceptance_criteria,
        )
        task_findings = self._context_findings(context)
        file_findings = await self._research_likely_files(planned_task)
        symbol_findings = await self._research_candidate_symbols(planned_task)
        module_findings = await self._research_candidate_modules(planned_task)
        examples = await self.collect_examples(planned_task)
        dependency_map, dependency_findings = await self.research_dependencies(planned_task)
        related_modules = self._unique_modules(
            (*task_findings, *file_findings, *symbol_findings, *module_findings, *examples)
        )
        patterns = self._detect_patterns(
            (*task_findings, *file_findings, *symbol_findings, *module_findings, *examples)
        )
        risks = self._detect_task_risks(
            planned_task,
            context=context,
            dependency_map=dependency_map,
            findings=(*task_findings, *file_findings, *symbol_findings, *module_findings),
        )
        findings = self._rank_findings(
            (
                *task_findings,
                *file_findings,
                *symbol_findings,
                *module_findings,
                *dependency_findings,
            )
        )
        confidence = self._task_confidence(context, findings, examples, risks)

        logger.info(
            "research.task_complete",
            task_id=planned_task.id,
            findings=len(findings),
            examples=len(examples),
            risks=len(risks),
            confidence=round(confidence, 4),
        )
        return ResearchTaskReport(
            task=planned_task,
            context=context,
            findings=findings,
            examples=examples,
            related_modules=related_modules,
            dependency_map=dependency_map,
            risks=risks,
            patterns=patterns,
            confidence=confidence,
        )

    async def research_symbol(
        self,
        symbol: str,
        *,
        top_k: int | None = None,
    ) -> tuple[ResearchEvidence, ...]:
        """Research a symbol through ``SemanticSearch``.

        Args:
            symbol: Symbol name or qualified suffix.
            top_k: Optional result limit.

        Returns:
            Ranked symbol evidence.
        """
        response = await self.semantic_search.search_symbol(
            symbol,
            top_k=top_k or self.config.symbol_top_k,
        )
        return self._evidence_from_response(
            response,
            kind=ResearchFindingKind.SYMBOL,
            title_prefix=f"Symbol match for {symbol}",
        )

    async def research_module(
        self,
        module: str,
        *,
        query: str | None = None,
        top_k: int | None = None,
    ) -> tuple[ResearchEvidence, ...]:
        """Research a module through ``SemanticSearch``.

        Args:
            module: Module path or file-like module query.
            query: Optional natural-language query scoped to the module.
            top_k: Optional result limit.

        Returns:
            Ranked module evidence.
        """
        response = await self.semantic_search.search_module(
            module,
            query=query,
            top_k=top_k or self.config.module_top_k,
        )
        return self._evidence_from_response(
            response,
            kind=ResearchFindingKind.MODULE,
            title_prefix=f"Module match for {module}",
        )

    async def research_dependencies(
        self,
        task_or_module: PlannedTask | str,
        *,
        top_k: int | None = None,
    ) -> tuple[dict[str, tuple[str, ...]], tuple[ResearchEvidence, ...]]:
        """Research dependency relationships for a module or planned task.

        Args:
            task_or_module: Module name or planned task whose likely files imply
                modules.
            top_k: Optional dependency result limit.

        Returns:
            Dependency map and ranked dependency evidence.
        """
        modules = (
            (task_or_module,)
            if isinstance(task_or_module, str)
            else self._modules_from_task(task_or_module)
        )
        dependency_map: dict[str, tuple[str, ...]] = {}
        evidence: list[ResearchEvidence] = []

        graph = self.semantic_search.dependency_graph
        for module in modules:
            if graph is not None:
                dependencies = tuple(sorted(graph.get_dependencies(module)))
                dependents = tuple(sorted(graph.get_dependents(module)))
                dependency_map[module] = dependencies
                if dependencies or dependents:
                    evidence.append(
                        ResearchEvidence(
                            kind=ResearchFindingKind.DEPENDENCY,
                            title=f"Dependency relationships for {module}",
                            summary=(
                                f"{module} depends on {len(dependencies)} internal modules "
                                f"and has {len(dependents)} direct dependents."
                            ),
                            confidence=0.8 if dependencies or dependents else 0.4,
                            relevance=0.75,
                            source="dependency_graph",
                            module_path=module,
                            metadata={
                                "dependencies": list(dependencies),
                                "dependents": list(dependents),
                            },
                        )
                    )

            response = await self.semantic_search.search_dependencies(
                module,
                top_k=top_k or self.config.dependency_top_k,
            )
            evidence.extend(
                self._evidence_from_response(
                    response,
                    kind=ResearchFindingKind.DEPENDENCY,
                    title_prefix=f"Dependency context for {module}",
                )
            )

        return dependency_map, self._rank_findings(evidence)

    async def collect_examples(
        self,
        task: PlannedTask | Mapping[str, Any] | str,
        *,
        top_k: int | None = None,
    ) -> tuple[ResearchEvidence, ...]:
        """Collect implementation examples relevant to a task.

        Args:
            task: Planned task, task mapping, or plain query string.
            top_k: Optional result limit.

        Returns:
            Ranked example evidence.
        """
        planned_task = self._coerce_task(task)
        query = self._example_query(planned_task)
        response = await self.semantic_search.search_code(
            query,
            top_k=top_k or self.config.example_top_k,
        )
        examples = self._evidence_from_response(
            response,
            kind=ResearchFindingKind.EXAMPLE,
            title_prefix="Implementation example",
        )
        return self._rank_findings(
            evidence
            for evidence in examples
            if evidence.confidence >= self.config.min_evidence_confidence
        )

    async def collect_context(
        self,
        task: PlannedTask | Mapping[str, Any] | str,
        *,
        goal: str = "",
        acceptance_criteria: Sequence[str] = (),
    ) -> BuiltContext:
        """Build prompt-ready repository context for a planned task.

        Args:
            task: Planned task, task mapping, or plain query string.
            goal: Parent plan goal.
            acceptance_criteria: Parent acceptance criteria.

        Returns:
            ContextBuilder output for the task.
        """
        planned_task = self._coerce_task(task)
        context_query = self._task_context_query(
            planned_task,
            goal=goal,
            acceptance_criteria=acceptance_criteria,
        )
        return await self.context_builder.build_task_context(
            context_query,
            top_k=self.config.task_top_k,
            max_tokens=self.config.max_context_tokens,
        )

    def generate_report(
        self,
        *,
        goal: str,
        acceptance_criteria: Sequence[str],
        task_reports: Sequence[ResearchTaskReport],
        metadata: Mapping[str, Any] | None = None,
    ) -> ResearchReport:
        """Generate a plan-level report from already collected task reports.

        Args:
            goal: Planner goal.
            acceptance_criteria: Planner acceptance criteria.
            task_reports: Per-task research reports.
            metadata: Optional plan metadata.

        Returns:
            Structured research report.
        """
        reports = tuple(task_reports)
        global_findings = self._global_findings(goal, reports)
        risks = self._plan_risks(reports, metadata or {})
        patterns = self._plan_patterns(reports)
        related_modules = self._unique_modules(
            finding
            for report in reports
            for finding in (*report.findings, *report.examples)
        )
        confidence = self._aggregate_confidence(report.confidence for report in reports)
        return ResearchReport(
            goal=goal,
            acceptance_criteria=tuple(acceptance_criteria),
            task_reports=reports,
            global_findings=global_findings,
            related_modules=related_modules,
            risks=risks,
            patterns=patterns,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )

    def _extract_plan(self, input_: AgentInput) -> Mapping[str, Any] | Any | None:
        direct = input_.context.get("plan")
        if direct is not None:
            return direct
        planner_output = input_.context.get("planner_output")
        if isinstance(planner_output, Mapping):
            return planner_output.get("plan") or planner_output.get("result", {}).get("plan")
        return None

    def _normalize_plan(
        self,
        plan: Mapping[str, Any] | Any,
    ) -> tuple[str, tuple[str, ...], tuple[PlannedTask, ...], dict[str, Any]]:
        if isinstance(plan, Mapping):
            goal = str(plan.get("goal", ""))
            acceptance_criteria = tuple(str(item) for item in plan.get("acceptance_criteria", ()))
            raw_subtasks = plan.get("subtasks", ())
            metadata = {
                "estimated_total_complexity": plan.get("estimated_total_complexity", "medium"),
                "requires_architecture_review": bool(
                    plan.get("requires_architecture_review", False)
                ),
            }
        else:
            goal = str(getattr(plan, "goal", ""))
            acceptance_criteria = tuple(
                str(item) for item in getattr(plan, "acceptance_criteria", ())
            )
            raw_subtasks = getattr(plan, "subtasks", ())
            metadata = {
                "estimated_total_complexity": getattr(
                    plan,
                    "estimated_total_complexity",
                    "medium",
                ),
                "requires_architecture_review": bool(
                    getattr(plan, "requires_architecture_review", False)
                ),
            }

        tasks = tuple(self._coerce_task(task) for task in raw_subtasks)
        return goal, acceptance_criteria, tasks, metadata

    def _coerce_task(self, task: PlannedTask | Mapping[str, Any] | str | Any) -> PlannedTask:
        if isinstance(task, PlannedTask):
            return task
        if isinstance(task, str):
            return PlannedTask(
                id="ad-hoc",
                title=task,
                description=task,
                type="research",
                complexity="medium",
            )
        if isinstance(task, Mapping):
            return self._task_from_mapping(task)
        return PlannedTask(
            id=str(getattr(task, "id", "unknown")),
            title=str(getattr(task, "title", "")),
            description=str(getattr(task, "description", "")),
            type=str(getattr(task, "type", "implementation")),
            complexity=str(getattr(task, "complexity", "medium")),
            depends_on=tuple(str(item) for item in getattr(task, "depends_on", ())),
            likely_files=tuple(str(item) for item in getattr(task, "likely_files", ())),
        )

    @staticmethod
    def _task_from_mapping(task: Mapping[str, Any]) -> PlannedTask:
        return PlannedTask(
            id=str(task.get("id", "unknown")),
            title=str(task.get("title", "")),
            description=str(task.get("description", "")),
            type=str(task.get("type", "implementation")),
            complexity=str(task.get("complexity", "medium")),
            depends_on=tuple(str(item) for item in task.get("depends_on", ())),
            likely_files=tuple(str(item) for item in task.get("likely_files", ())),
        )

    async def _research_likely_files(self, task: PlannedTask) -> tuple[ResearchEvidence, ...]:
        evidence: list[ResearchEvidence] = []
        for file_path in task.likely_files:
            response = await self.semantic_search.search_file(
                file_path,
                query=task.search_text or file_path,
                top_k=self.config.module_top_k,
            )
            evidence.extend(
                self._evidence_from_response(
                    response,
                    kind=ResearchFindingKind.CONTEXT,
                    title_prefix=f"Likely file context for {file_path}",
                )
            )
        return self._rank_findings(evidence)

    async def _research_candidate_symbols(self, task: PlannedTask) -> tuple[ResearchEvidence, ...]:
        evidence: list[ResearchEvidence] = []
        for symbol in self._candidate_symbols(task):
            evidence.extend(await self.research_symbol(symbol, top_k=self.config.symbol_top_k))
        return self._rank_findings(evidence)

    async def _research_candidate_modules(self, task: PlannedTask) -> tuple[ResearchEvidence, ...]:
        evidence: list[ResearchEvidence] = []
        for module in self._modules_from_task(task):
            evidence.extend(
                await self.research_module(
                    module,
                    query=task.search_text,
                    top_k=self.config.module_top_k,
                )
            )
        return self._rank_findings(evidence)

    def _context_findings(self, context: BuiltContext) -> tuple[ResearchEvidence, ...]:
        evidence = [
            ResearchEvidence(
                kind=ResearchFindingKind.CONTEXT,
                title=f"Context block: {block.symbol_name}",
                summary=(
                    f"{block.symbol_type.value} context from {block.location} "
                    f"with importance {block.importance:.2f}."
                ),
                confidence=self._clamp(block.importance),
                relevance=self._clamp(block.importance),
                source="context_builder",
                file_path=block.file_path.as_posix(),
                module_path=block.module_path,
                symbol_name=block.symbol_name,
                location=block.location,
                metadata={
                    "dependency_role": block.dependency_role,
                    "token_count": block.token_count,
                    "parent_class": block.parent_class,
                },
            )
            for block in context.blocks
        ]
        return self._rank_findings(evidence)

    def _evidence_from_response(
        self,
        response: SemanticSearchResponse,
        *,
        kind: ResearchFindingKind,
        title_prefix: str,
    ) -> tuple[ResearchEvidence, ...]:
        evidence = [
            self._result_to_evidence(result, kind=kind, title_prefix=title_prefix)
            for result in response.all_results
            if result.confidence >= self.config.min_evidence_confidence
        ]
        return self._rank_findings(evidence)

    @staticmethod
    def _result_to_evidence(
        result: SemanticSearchResult,
        *,
        kind: ResearchFindingKind,
        title_prefix: str,
    ) -> ResearchEvidence:
        return ResearchEvidence(
            kind=kind,
            title=f"{title_prefix}: {result.symbol_name}",
            summary=(
                f"{result.symbol_type.value} {result.symbol_name} in "
                f"{result.module_path} at {result.location}."
            ),
            confidence=ResearchAgent._clamp(result.confidence),
            relevance=ResearchAgent._clamp(result.confidence),
            source=f"semantic_search.{result.search_kind.value}.{result.retrieval_source}",
            file_path=result.file_path.as_posix(),
            module_path=result.module_path,
            symbol_name=result.symbol_name,
            location=result.location,
            metadata={
                "matched_terms": list(result.matched_terms),
                "parent_class": result.parent_class,
                "has_docstring": bool(result.docstring),
                **result.metadata,
            },
        )

    def _detect_task_risks(
        self,
        task: PlannedTask,
        *,
        context: BuiltContext,
        dependency_map: Mapping[str, tuple[str, ...]],
        findings: Sequence[ResearchEvidence],
    ) -> tuple[ResearchEvidence, ...]:
        risks: list[ResearchEvidence] = []
        if not context.blocks:
            risks.append(
                self._risk(
                    "No repository context found",
                    "The task has no retrieved context, so implementation may miss "
                    "local conventions.",
                    confidence=0.7,
                    relevance=0.9,
                    metadata={"task_id": task.id},
                )
            )
        if task.complexity == "high":
            risks.append(
                self._risk(
                    "High-complexity planner task",
                    "The planner marked this task as high complexity; changes may span "
                    "multiple behaviors.",
                    confidence=0.75,
                    relevance=0.8,
                    metadata={"task_id": task.id, "complexity": task.complexity},
                )
            )
        if len(task.likely_files) > 3:
            risks.append(
                self._risk(
                    "Broad file surface",
                    f"The planner identified {len(task.likely_files)} likely files.",
                    confidence=0.65,
                    relevance=0.7,
                    metadata={"likely_files": list(task.likely_files)},
                )
            )
        if any(dependencies for dependencies in dependency_map.values()):
            risks.append(
                self._risk(
                    "Dependency-sensitive change",
                    "Relevant modules have internal dependencies that may need "
                    "coordinated updates.",
                    confidence=0.7,
                    relevance=0.75,
                    metadata={
                        "dependency_map": {
                            module: list(dependencies)
                            for module, dependencies in dependency_map.items()
                        }
                    },
                )
            )
        if findings and self._aggregate_confidence(item.confidence for item in findings) < 0.35:
            risks.append(
                self._risk(
                    "Low-confidence evidence",
                    "Retrieved evidence is weakly matched to the planned task.",
                    confidence=0.6,
                    relevance=0.65,
                    metadata={"task_id": task.id},
                )
            )
        return self._rank_findings(risks)[: self.config.max_risks_per_task]

    def _detect_patterns(
        self,
        evidence: Iterable[ResearchEvidence],
    ) -> tuple[ResearchEvidence, ...]:
        items = tuple(evidence)
        module_counts = Counter(item.module_path for item in items if item.module_path)
        symbol_type_counts = Counter(
            str(item.metadata.get("symbol_type") or item.metadata.get("type"))
            for item in items
            if item.metadata.get("symbol_type") or item.metadata.get("type")
        )
        patterns: list[ResearchEvidence] = []

        for module, count in module_counts.most_common(3):
            if count < 2:
                continue
            patterns.append(
                ResearchEvidence(
                    kind=ResearchFindingKind.PATTERN,
                    title=f"Repeated evidence in {module}",
                    summary=f"{count} findings point to module {module}.",
                    confidence=self._clamp(0.45 + min(count, 5) * 0.1),
                    relevance=self._clamp(0.4 + min(count, 5) * 0.1),
                    source="research.pattern_detector",
                    module_path=module,
                    metadata={"finding_count": count},
                )
            )

        for symbol_type, count in symbol_type_counts.most_common(2):
            if symbol_type == "None" or count < 2:
                continue
            patterns.append(
                ResearchEvidence(
                    kind=ResearchFindingKind.PATTERN,
                    title=f"Repeated {symbol_type} evidence",
                    summary=f"{count} findings share symbol type {symbol_type}.",
                    confidence=self._clamp(0.4 + min(count, 5) * 0.08),
                    relevance=self._clamp(0.35 + min(count, 5) * 0.08),
                    source="research.pattern_detector",
                    metadata={"symbol_type": symbol_type, "finding_count": count},
                )
            )

        return self._rank_findings(patterns)

    @staticmethod
    def _risk(
        title: str,
        summary: str,
        *,
        confidence: float,
        relevance: float,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchEvidence:
        return ResearchEvidence(
            kind=ResearchFindingKind.RISK,
            title=title,
            summary=summary,
            confidence=confidence,
            relevance=relevance,
            source="research.risk_detector",
            metadata=metadata or {},
        )

    def _global_findings(
        self,
        goal: str,
        task_reports: Sequence[ResearchTaskReport],
    ) -> tuple[ResearchEvidence, ...]:
        modules = self._unique_modules(
            finding
            for report in task_reports
            for finding in (*report.findings, *report.examples)
        )
        total_context_blocks = sum(len(report.context.blocks) for report in task_reports)
        if not task_reports:
            return ()
        return (
            ResearchEvidence(
                kind=ResearchFindingKind.CONTEXT,
                title="Plan research coverage",
                summary=(
                    f"Research for '{goal}' gathered {total_context_blocks} context blocks "
                    f"across {len(modules)} modules."
                ),
                confidence=self._aggregate_confidence(report.confidence for report in task_reports),
                relevance=0.9,
                source="research.aggregate",
                metadata={
                    "task_reports": len(task_reports),
                    "context_blocks": total_context_blocks,
                    "modules": list(modules),
                },
            ),
        )

    def _plan_risks(
        self,
        task_reports: Sequence[ResearchTaskReport],
        metadata: Mapping[str, Any],
    ) -> tuple[ResearchEvidence, ...]:
        risks = [
            risk
            for report in task_reports
            for risk in report.risks
        ]
        if metadata.get("requires_architecture_review"):
            risks.append(
                self._risk(
                    "Architecture review required",
                    "The planner marked the plan as requiring architecture review.",
                    confidence=0.85,
                    relevance=0.9,
                    metadata=dict(metadata),
                )
            )
        return self._rank_findings(risks)

    def _plan_patterns(
        self,
        task_reports: Sequence[ResearchTaskReport],
    ) -> tuple[ResearchEvidence, ...]:
        patterns = [
            pattern
            for report in task_reports
            for pattern in report.patterns
        ]
        return self._rank_findings(patterns)

    def _task_confidence(
        self,
        context: BuiltContext,
        findings: Sequence[ResearchEvidence],
        examples: Sequence[ResearchEvidence],
        risks: Sequence[ResearchEvidence],
    ) -> float:
        evidence_scores = [item.confidence for item in (*findings, *examples)]
        evidence_confidence = self._aggregate_confidence(evidence_scores)
        context_bonus = min(len(context.blocks), 5) * 0.04
        example_bonus = min(len(examples), 3) * 0.03
        risk_penalty = min(len(risks), 4) * 0.04
        return self._clamp(evidence_confidence + context_bonus + example_bonus - risk_penalty)

    def _candidate_symbols(self, task: PlannedTask) -> tuple[str, ...]:
        candidates: list[str] = []
        for file_path in task.likely_files:
            stem = Path(file_path).stem
            if stem and stem != "__init__":
                candidates.append(stem)
        for token in self._search_tokens(task.search_text):
            if "_" in token or token[:1].isupper():
                candidates.append(token)
        return tuple(dict.fromkeys(candidates))[:4]

    def _modules_from_task(self, task: PlannedTask) -> tuple[str, ...]:
        modules: list[str] = []
        indexed = set(self.semantic_search.indexed_modules())
        for file_path in task.likely_files:
            module = self._module_from_file(file_path)
            if module in indexed or not indexed:
                modules.append(module)
        return tuple(dict.fromkeys(module for module in modules if module))

    @staticmethod
    def _module_from_file(file_path: str) -> str:
        path = Path(file_path)
        without_suffix = path.with_suffix("")
        parts = [part for part in without_suffix.parts if part not in {".", "__init__"}]
        if parts and parts[0] == "kodiak":
            parts = parts[1:]
        return ".".join(parts)

    @staticmethod
    def _search_tokens(text: str) -> tuple[str, ...]:
        tokens = [
            token.strip(".,:;()[]{}'\"`")
            for token in text.split()
        ]
        return tuple(token for token in tokens if len(token) > 2)

    @staticmethod
    def _task_context_query(
        task: PlannedTask,
        *,
        goal: str,
        acceptance_criteria: Sequence[str],
    ) -> str:
        parts = [task.search_text]
        if goal:
            parts.append(f"Goal: {goal}")
        if acceptance_criteria:
            parts.append("Acceptance criteria: " + "; ".join(acceptance_criteria))
        if task.likely_files:
            parts.append("Likely files: " + ", ".join(task.likely_files))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _example_query(task: PlannedTask) -> str:
        return (
            f"Existing implementation examples and patterns for {task.title}. "
            f"{task.description}"
        ).strip()

    @staticmethod
    def _unique_modules(evidence: Iterable[ResearchEvidence]) -> tuple[str, ...]:
        modules = sorted(
            {
                item.module_path
                for item in evidence
                if item.module_path
            }
        )
        return tuple(modules)

    @staticmethod
    def _rank_findings(
        evidence: Iterable[ResearchEvidence],
    ) -> tuple[ResearchEvidence, ...]:
        deduped: dict[tuple[str, str | None, str | None, str | None], ResearchEvidence] = {}
        for item in evidence:
            key = (item.title, item.module_path, item.symbol_name, item.location)
            current = deduped.get(key)
            if current is None or (item.confidence, item.relevance) > (
                current.confidence,
                current.relevance,
            ):
                deduped[key] = item
        return tuple(
            sorted(
                deduped.values(),
                key=lambda item: (item.relevance, item.confidence, item.title),
                reverse=True,
            )
        )

    @staticmethod
    def _aggregate_confidence(scores: Iterable[float]) -> float:
        values = [ResearchAgent._clamp(score) for score in scores]
        if not values:
            return 0.0
        weighted = sum(values) / len(values)
        coverage_bonus = min(len(values), 8) * 0.015
        return ResearchAgent._clamp(weighted + coverage_bonus)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


ResearcherAgent = ResearchAgent
