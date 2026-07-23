"""Review agent for Kodiak's autonomous engineering workflow.

The review agent validates generated code patches before they reach testing,
GitHub, or any downstream automation. It consumes execution plans, research
reports, generated code patches, and repository context prepared by existing
Kodiak services. It does not retrieve context, run tests, execute repository
code, modify files, commit changes, or push to GitHub.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

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
from kodiak.agents.coder import CodePatch, FilePatch, PatchAction
from kodiak.rag.context_builder import BuiltContext

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = """\
You are a senior code reviewer inside the Kodiak autonomous engineering system.
Review generated code patches against the supplied execution plan, research
report, and repository context.

Rules:
- Consume the supplied plan, research report, patch, and context only.
- Do not request repository retrieval or duplicate repository analysis.
- Do not suggest running tests as a substitute for review.
- Do not modify code, execute code, commit changes, or push changes.
- Focus on correctness, architecture, imports, style, security, performance,
  API compatibility, dependency usage, and maintainability.
- Output ONLY valid JSON; no prose and no markdown fences.

Output schema:
{
  "summary": "<one paragraph review summary>",
  "recommendation": "APPROVE | REQUEST_CHANGES | REJECT",
  "confidence": 0.0,
  "issues": [
    {
      "category": "<one of the review issue categories>",
      "severity": "critical | high | medium | low | info",
      "file": "relative/path.py",
      "line": 1,
      "title": "<short title>",
      "description": "<specific problem>",
      "suggestion": "<specific fix>",
      "confidence": 0.0
    }
  ],
  "approved_files": ["relative/path.py"],
  "must_fix": ["<blocking issue summary>"]
}
"""


class ReviewRecommendation(StrEnum):
    """Final review recommendation."""

    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    REJECT = "REJECT"


class ReviewSeverity(StrEnum):
    """Severity levels assigned to review issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewCategory(StrEnum):
    """Review issue categories."""

    CORRECTNESS = "correctness"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    STYLE = "style"
    IMPORTS = "imports"
    PERFORMANCE = "performance"
    API = "api"
    DEPENDENCIES = "dependencies"
    MAINTAINABILITY = "maintainability"


@dataclass(frozen=True)
class ReviewIssue:
    """A structured review issue.

    Attributes:
        category: Issue category.
        severity: Issue severity.
        file: Repository-relative file path, when known.
        line: One-based line number, when known.
        title: Short issue title.
        description: Specific explanation of the problem.
        suggestion: Concrete remediation guidance.
        confidence: Confidence score from 0.0 to 1.0.
        source: Detector that produced the issue.
    """

    category: ReviewCategory
    severity: ReviewSeverity
    file: str | None
    line: int | None
    title: str
    description: str
    suggestion: str
    confidence: float
    source: str = "reviewer"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the issue."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "file": self.file,
            "line": self.line,
            "title": self.title,
            "description": self.description,
            "suggestion": self.suggestion,
            "confidence": round(self.confidence, 6),
            "source": self.source,
        }


@dataclass(frozen=True)
class ReviewFile:
    """Patch file normalized for review.

    Attributes:
        path: Repository-relative file path.
        action: Patch action.
        old_content: File content before the patch.
        new_content: File content after the patch.
        unified_diff: Unified diff for this file.
    """

    path: str
    action: PatchAction
    old_content: str = ""
    new_content: str = ""
    unified_diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the file."""
        return {
            "path": self.path,
            "action": self.action.value,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "unified_diff": self.unified_diff,
        }


@dataclass(frozen=True)
class ReviewReport:
    """Structured review report for generated code changes.

    Attributes:
        recommendation: Final reviewer recommendation.
        score: Numeric quality score from 0 to 100.
        confidence: Aggregate review confidence from 0.0 to 1.0.
        summary: Human-readable review summary.
        issues: Ordered review issues.
        approved_files: Files with no blocking findings.
        must_fix: Blocking issue summaries.
        metadata: Additional structured review metadata.
        token_usage: LLM token usage metadata.
    """

    recommendation: ReviewRecommendation
    score: int
    confidence: float
    summary: str
    issues: tuple[ReviewIssue, ...] = ()
    approved_files: tuple[str, ...] = ()
    must_fix: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the review report."""
        return {
            "recommendation": self.recommendation.value,
            "score": self.score,
            "confidence": round(self.confidence, 6),
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "approved_files": list(self.approved_files),
            "must_fix": list(self.must_fix),
            "metadata": self.metadata,
            "token_usage": self.token_usage,
        }


class ReviewAgent(BaseAgent):
    """Review generated code patches without modifying or executing code."""

    role = AgentRole.REVIEWER

    _SECURITY_PATTERNS: tuple[tuple[re.Pattern[str], str, str, ReviewSeverity], ...] = (
        (
            re.compile(r"\b(eval|exec)\s*\("),
            "Dynamic code execution",
            "Avoid eval/exec or constrain execution through a reviewed safe parser.",
            ReviewSeverity.HIGH,
        ),
        (
            re.compile(r"shell\s*=\s*True"),
            "Shell command execution",
            "Pass argv lists without shell=True and validate all user-controlled input.",
            ReviewSeverity.HIGH,
        ),
        (
            re.compile(r"\b(pickle|dill)\.loads?\s*\("),
            "Unsafe deserialization",
            "Use a safe serialization format for untrusted data.",
            ReviewSeverity.HIGH,
        ),
        (
            re.compile(r"yaml\.load\s*\((?![^)]*SafeLoader)"),
            "Unsafe YAML loading",
            "Use yaml.safe_load or pass SafeLoader explicitly.",
            ReviewSeverity.HIGH,
        ),
        (
            re.compile(r"verify\s*=\s*False"),
            "TLS verification disabled",
            "Keep certificate verification enabled.",
            ReviewSeverity.HIGH,
        ),
        (
            re.compile(r"\b(md5|sha1)\s*\("),
            "Weak hash algorithm",
            "Use SHA-256 or a stronger algorithm for security-sensitive hashing.",
            ReviewSeverity.MEDIUM,
        ),
        (
            re.compile(r"(api_key|secret|password|token)\s*=\s*['\"][^'\"]+['\"]", re.I),
            "Hardcoded secret-like value",
            "Load secrets from Kodiak configuration or a secrets provider.",
            ReviewSeverity.CRITICAL,
        ),
    )

    def __init__(
        self,
        llm_client: Any,
        *,
        default_model_preference: str = "default",
        max_output_tokens: int = 5000,
    ) -> None:
        """Initialize the review agent.

        Args:
            llm_client: Injected LLM facade or compatible provider.
            default_model_preference: LLM routing preference.
            max_output_tokens: Maximum tokens requested for review generation.
        """
        super().__init__()
        self._llm = llm_client
        self._default_model_preference = default_model_preference
        self._max_output_tokens = max_output_tokens

    async def _run(self, input_: AgentInput) -> AgentOutput:
        """Run the review agent through BaseAgent orchestration."""
        patch = self._first_present(
            input_.context,
            "generated_patch",
            "code_patch",
            "patch",
        )
        if patch is None:
            patch = {
                "files": input_.context.get("code_changes", []),
                "unified_diff": input_.context.get("diff", ""),
            }
        if not patch:
            return self._make_error(input_, "generated code patch required in context")

        report = await self.review(
            instruction=input_.instruction,
            execution_plan=self._first_present(
                input_.context,
                "execution_plan",
                "plan",
                "task_plan",
            ),
            research_report=input_.context.get("research_report"),
            generated_patch=patch,
            repository_context=self._first_present(
                input_.context,
                "repository_context",
                "built_context",
                "context",
                "rag_context",
            ),
            model_preference=input_.context.get(
                "model_preference",
                self._default_model_preference,
            ),
        )
        return self._make_output(
            input_,
            result=report.to_dict(),
            token_usage=report.token_usage,
            metadata=report.metadata,
        )

    async def review(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        generated_patch: Any,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        model_preference: str | None = None,
    ) -> ReviewReport:
        """Review generated code changes against prepared workflow context.

        Args:
            instruction: Original implementation instruction.
            execution_plan: PlannerAgent output or normalized execution plan.
            generated_patch: CodePatch, serialized patch, or legacy change set.
            research_report: ResearchAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            model_preference: Optional LLM routing preference.

        Returns:
            Structured review report with recommendation, score, and issues.
        """
        files = self._normalize_patch_files(generated_patch)
        patch_diff = self._patch_unified_diff(generated_patch, files)
        deterministic_issues = self._detect_all_static_issues(files, execution_plan)
        llm_report = await self._llm_review(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=repository_context,
            files=files,
            patch_diff=patch_diff,
            static_issues=deterministic_issues,
            model_preference=model_preference,
        )
        return self.generate_review_report(
            files=files,
            static_issues=deterministic_issues,
            llm_report=llm_report,
        )

    async def review_patch(
        self,
        patch: Any,
        *,
        instruction: str = "",
        execution_plan: Any | None = None,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        model_preference: str | None = None,
    ) -> ReviewReport:
        """Review a generated code patch.

        Args:
            patch: CodePatch, serialized patch, or legacy patch dict.
            instruction: Original implementation instruction.
            execution_plan: PlannerAgent output or normalized execution plan.
            research_report: ResearchAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            model_preference: Optional LLM routing preference.

        Returns:
            Structured review report.
        """
        return await self.review(
            instruction=instruction,
            execution_plan=execution_plan,
            generated_patch=patch,
            research_report=research_report,
            repository_context=repository_context,
            model_preference=model_preference,
        )

    async def review_file(
        self,
        *,
        path: str,
        content: str,
        old_content: str = "",
        action: PatchAction | str = PatchAction.MODIFY,
        instruction: str = "",
        execution_plan: Any | None = None,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        model_preference: str | None = None,
    ) -> ReviewReport:
        """Review one created or modified file.

        Args:
            path: Repository-relative file path.
            content: New file content.
            old_content: Previous file content for modifications.
            action: Patch action for the file.
            instruction: Original implementation instruction.
            execution_plan: PlannerAgent output or normalized execution plan.
            research_report: ResearchAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            model_preference: Optional LLM routing preference.

        Returns:
            Structured review report for the single file.
        """
        file_patch = {
            "files": [
                {
                    "path": path,
                    "action": str(action.value if isinstance(action, PatchAction) else action),
                    "old_content": old_content,
                    "new_content": content,
                    "content": content,
                }
            ]
        }
        return await self.review_patch(
            file_patch,
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=repository_context,
            model_preference=model_preference,
        )

    async def review_project(
        self,
        *,
        patches: Sequence[Any],
        instruction: str = "",
        execution_plan: Any | None = None,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        model_preference: str | None = None,
    ) -> ReviewReport:
        """Review multiple generated patches as one project-level change set.

        Args:
            patches: Sequence of generated patches or patch dictionaries.
            instruction: Original implementation instruction.
            execution_plan: PlannerAgent output or normalized execution plan.
            research_report: ResearchAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            model_preference: Optional LLM routing preference.

        Returns:
            Combined structured review report.
        """
        files: list[dict[str, Any]] = []
        for patch in patches:
            files.extend(file.to_dict() for file in self._normalize_patch_files(patch))
        return await self.review_patch(
            {"files": files},
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=repository_context,
            model_preference=model_preference,
        )

    def detect_architecture_issues(
        self,
        patch: Any,
        *,
        execution_plan: Any | None = None,
    ) -> tuple[ReviewIssue, ...]:
        """Detect architecture and API boundary issues without running code.

        Args:
            patch: Generated patch or normalized file sequence.
            execution_plan: Planner output used to compare expected files.

        Returns:
            Architecture and API review issues.
        """
        files = self._normalize_patch_files(patch)
        issues: list[ReviewIssue] = []
        planned_files = self._planned_files(execution_plan)
        changed_files = {file.path for file in files}

        if planned_files:
            unexpected = sorted(changed_files - planned_files)
            for path in unexpected:
                issues.append(
                    self._issue(
                        ReviewCategory.ARCHITECTURE,
                        ReviewSeverity.MEDIUM,
                        path,
                        None,
                        "Unexpected file changed",
                        "The patch changes a file not listed by the execution plan.",
                        "Confirm the plan or split unrelated changes into a separate task.",
                        0.72,
                        "architecture",
                    )
                )

        for file in files:
            top_level = Path(file.path).parts[0] if Path(file.path).parts else ""
            if top_level and top_level not in {"kodiak", "tests", "docs", "scripts"}:
                issues.append(
                    self._issue(
                        ReviewCategory.ARCHITECTURE,
                        ReviewSeverity.HIGH,
                        file.path,
                        None,
                        "New top-level boundary",
                        "The change touches a top-level path outside Kodiak's layout.",
                        "Keep code inside the established Kodiak package structure.",
                        0.8,
                        "architecture",
                    )
                )
            if file.path.startswith("kodiak/") and "from tests" in file.new_content:
                issues.append(
                    self._issue(
                        ReviewCategory.ARCHITECTURE,
                        ReviewSeverity.HIGH,
                        file.path,
                        self._line_for(file.new_content, "from tests"),
                        "Production code imports tests",
                        "Production modules should not depend on test modules.",
                        "Move shared helpers into kodiak utilities or fixtures.",
                        0.9,
                        "architecture",
                    )
                )
            issues.extend(self._detect_breaking_api_changes(file))

        return tuple(issues)

    def detect_security_issues(self, patch: Any) -> tuple[ReviewIssue, ...]:
        """Detect security risks in generated file content.

        Args:
            patch: Generated patch or normalized file sequence.

        Returns:
            Security review issues.
        """
        issues: list[ReviewIssue] = []
        for file in self._normalize_patch_files(patch):
            if file.action is PatchAction.DELETE:
                continue
            for pattern, title, suggestion, severity in self._SECURITY_PATTERNS:
                match = pattern.search(file.new_content)
                if match:
                    issues.append(
                        self._issue(
                            ReviewCategory.SECURITY,
                            severity,
                            file.path,
                            self._line_for_offset(file.new_content, match.start()),
                            title,
                            f"Generated code matches security-sensitive pattern: {match.group(0)}",
                            suggestion,
                            0.78,
                            "security",
                        )
                    )
        return tuple(issues)

    def detect_style_issues(self, patch: Any) -> tuple[ReviewIssue, ...]:
        """Detect style, import, duplication, and maintainability issues.

        Args:
            patch: Generated patch or normalized file sequence.

        Returns:
            Style and maintainability review issues.
        """
        issues: list[ReviewIssue] = []
        for file in self._normalize_patch_files(patch):
            if file.action is PatchAction.DELETE:
                continue
            issues.extend(self._detect_python_syntax_issues(file))
            issues.extend(self._detect_unused_imports(file))
            issues.extend(self._detect_duplicate_defs(file))
            issues.extend(self._detect_line_style(file))
            issues.extend(self._detect_performance_smells(file))
        return tuple(issues)

    def generate_review_report(
        self,
        *,
        files: Sequence[ReviewFile],
        static_issues: Sequence[ReviewIssue],
        llm_report: ReviewReport | None = None,
    ) -> ReviewReport:
        """Combine static and LLM findings into a final review report.

        Args:
            files: Reviewed files.
            static_issues: Deterministic static findings.
            llm_report: Optional semantic review report from the LLM.

        Returns:
            Final structured review report.
        """
        all_issues = self._dedupe_issues(
            tuple(static_issues) + (llm_report.issues if llm_report else ())
        )
        score = self._score(all_issues)
        recommendation = self._recommendation(all_issues, score)
        confidence = self._confidence(all_issues, llm_report)
        changed_files = tuple(file.path for file in files)
        blocking_files = {
            issue.file
            for issue in all_issues
            if issue.file and issue.severity in self._blocking_severities()
        }
        approved_files = tuple(path for path in changed_files if path not in blocking_files)
        must_fix = tuple(
            issue.title for issue in all_issues if issue.severity in self._blocking_severities()
        )

        summary = self._summary(recommendation, score, all_issues, llm_report)
        return ReviewReport(
            recommendation=recommendation,
            score=score,
            confidence=confidence,
            summary=summary,
            issues=all_issues,
            approved_files=approved_files,
            must_fix=must_fix,
            metadata={
                "changed_files": list(changed_files),
                "static_issue_count": len(static_issues),
                "llm_issue_count": len(llm_report.issues) if llm_report else 0,
            },
            token_usage=llm_report.token_usage if llm_report else {},
        )

    async def _llm_review(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        research_report: Any | None,
        repository_context: Any | None,
        files: Sequence[ReviewFile],
        patch_diff: str,
        static_issues: Sequence[ReviewIssue],
        model_preference: str | None,
    ) -> ReviewReport:
        message = self._build_message(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=repository_context,
            files=files,
            patch_diff=patch_diff,
            static_issues=static_issues,
        )
        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
            model_preference=model_preference or self._default_model_preference,
            max_tokens=self._max_output_tokens,
            temperature=0.1,
        )
        return self._parse_llm_report(
            str(response.get("content", "")),
            token_usage=dict(response.get("usage", {})),
        )

    def _build_message(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        research_report: Any | None,
        repository_context: Any | None,
        files: Sequence[ReviewFile],
        patch_diff: str,
        static_issues: Sequence[ReviewIssue],
    ) -> str:
        parts = [
            f"## Original task\n{instruction}",
            f"## Execution plan\n{self._to_pretty_json(execution_plan)}",
        ]
        if research_report:
            parts.append(f"## Research report\n{self._to_pretty_json(research_report)}")
        context_text = self._context_to_text(repository_context)
        if context_text:
            parts.append(f"## Repository context\n{context_text}")
        if patch_diff:
            parts.append(f"## Generated patch\n```diff\n{patch_diff}\n```")
        parts.append(
            f"## Changed files\n{self._to_pretty_json([file.to_dict() for file in files])}"
        )
        if static_issues:
            parts.append(
                "## Static findings to consider\n"
                f"{self._to_pretty_json([issue.to_dict() for issue in static_issues])}"
            )
        return "\n\n".join(parts)

    def _detect_all_static_issues(
        self,
        files: Sequence[ReviewFile],
        execution_plan: Any,
    ) -> tuple[ReviewIssue, ...]:
        patch = tuple(files)
        return (
            *self.detect_architecture_issues(patch, execution_plan=execution_plan),
            *self.detect_security_issues(patch),
            *self.detect_style_issues(patch),
        )

    def _normalize_patch_files(self, patch: Any) -> tuple[ReviewFile, ...]:
        if isinstance(patch, CodePatch):
            return tuple(self._from_file_patch(file) for file in patch.files)
        if isinstance(patch, FilePatch):
            return (self._from_file_patch(patch),)
        if isinstance(patch, ReviewFile):
            return (patch,)
        if isinstance(patch, Sequence) and not isinstance(patch, str | bytes | Mapping):
            return tuple(
                item if isinstance(item, ReviewFile) else self._from_mapping(item) for item in patch
            )
        if isinstance(patch, Mapping):
            files = patch.get("files") or patch.get("code_changes")
            if isinstance(files, Sequence) and not isinstance(files, str | bytes):
                return tuple(self._from_mapping(file) for file in files)
            if "path" in patch:
                return (self._from_mapping(patch),)
        raise ValueError("generated patch must include reviewable file entries")

    def _from_file_patch(self, file: FilePatch) -> ReviewFile:
        return ReviewFile(
            path=file.path,
            action=file.action,
            old_content=file.old_content,
            new_content=file.new_content,
            unified_diff=file.unified_diff,
        )

    def _from_mapping(self, value: Any) -> ReviewFile:
        if not isinstance(value, Mapping):
            raise ValueError("patch file entries must be objects")
        action = self._coerce_action(value.get("action", PatchAction.MODIFY.value))
        new_content = str(value.get("new_content", value.get("content", "")))
        return ReviewFile(
            path=str(value.get("path", "")),
            action=action,
            old_content=str(value.get("old_content", "")),
            new_content="" if action is PatchAction.DELETE else new_content,
            unified_diff=str(value.get("unified_diff", value.get("diff", ""))),
        )

    def _coerce_action(self, value: Any) -> PatchAction:
        try:
            return PatchAction(str(value).lower())
        except ValueError:
            return PatchAction.MODIFY

    def _patch_unified_diff(
        self,
        patch: Any,
        files: Sequence[ReviewFile],
    ) -> str:
        if isinstance(patch, CodePatch):
            return patch.unified_diff
        if isinstance(patch, Mapping):
            diff = patch.get("unified_diff") or patch.get("diff")
            if isinstance(diff, str) and diff:
                return diff
        return "\n".join(file.unified_diff for file in files if file.unified_diff)

    def _detect_python_syntax_issues(self, file: ReviewFile) -> tuple[ReviewIssue, ...]:
        if not file.path.endswith(".py"):
            return ()
        try:
            ast.parse(file.new_content)
        except SyntaxError as exc:
            return (
                self._issue(
                    ReviewCategory.IMPORTS,
                    ReviewSeverity.CRITICAL,
                    file.path,
                    exc.lineno,
                    "Python syntax error",
                    exc.msg,
                    "Fix the syntax before the patch reaches testing.",
                    0.98,
                    "syntax",
                ),
            )
        return ()

    def _detect_unused_imports(self, file: ReviewFile) -> tuple[ReviewIssue, ...]:
        if not file.path.endswith(".py"):
            return ()
        try:
            tree = ast.parse(file.new_content)
        except SyntaxError:
            return ()

        imports: dict[str, tuple[str, int]] = {}
        used_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = (
                        alias.name,
                        node.lineno,
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imports[alias.asname or alias.name] = (alias.name, node.lineno)
            elif isinstance(node, ast.Name):
                used_names.add(node.id)

        issues: list[ReviewIssue] = []
        for local_name, (import_name, line) in imports.items():
            if local_name in used_names:
                continue
            if import_name in {"annotations", "TYPE_CHECKING"}:
                continue
            issues.append(
                self._issue(
                    ReviewCategory.IMPORTS,
                    ReviewSeverity.LOW,
                    file.path,
                    line,
                    "Possibly unused import",
                    f"Imported name '{import_name}' is not referenced in the file.",
                    "Remove the import or use it where intended.",
                    0.62,
                    "imports",
                )
            )
        return tuple(issues)

    def _detect_duplicate_defs(self, file: ReviewFile) -> tuple[ReviewIssue, ...]:
        if not file.path.endswith(".py"):
            return ()
        try:
            tree = ast.parse(file.new_content)
        except SyntaxError:
            return ()
        names: list[tuple[str, int]] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.append((node.name, node.lineno))
        counts = Counter(name for name, _ in names)
        return tuple(
            self._issue(
                ReviewCategory.MAINTAINABILITY,
                ReviewSeverity.MEDIUM,
                file.path,
                line,
                "Duplicate definition name",
                f"'{name}' is defined more than once in the generated file.",
                "Rename or consolidate the duplicated definition.",
                0.7,
                "duplication",
            )
            for name, line in names
            if counts[name] > 1
        )

    def _detect_line_style(self, file: ReviewFile) -> tuple[ReviewIssue, ...]:
        issues: list[ReviewIssue] = []
        for index, line in enumerate(file.new_content.splitlines(), start=1):
            if len(line) > 100 and file.path.endswith(".py"):
                issues.append(
                    self._issue(
                        ReviewCategory.STYLE,
                        ReviewSeverity.LOW,
                        file.path,
                        index,
                        "Long Python line",
                        "Generated Python line exceeds the configured 100 character limit.",
                        "Wrap the expression to match project formatting.",
                        0.75,
                        "style",
                    )
                )
            if re.search(r"\bTODO\b|\bFIXME\b", line):
                issues.append(
                    self._issue(
                        ReviewCategory.MAINTAINABILITY,
                        ReviewSeverity.MEDIUM,
                        file.path,
                        index,
                        "Placeholder left in generated code",
                        "Generated code contains TODO/FIXME text.",
                        "Complete the implementation before sending it downstream.",
                        0.82,
                        "style",
                    )
                )
            if file.path.startswith("kodiak/") and re.search(r"\bprint\s*\(", line):
                issues.append(
                    self._issue(
                        ReviewCategory.STYLE,
                        ReviewSeverity.LOW,
                        file.path,
                        index,
                        "Print call in production code",
                        "Production modules should use structured logging.",
                        "Use structlog or an existing Kodiak logger.",
                        0.74,
                        "style",
                    )
                )
        return tuple(issues)

    def _detect_performance_smells(self, file: ReviewFile) -> tuple[ReviewIssue, ...]:
        issues: list[ReviewIssue] = []
        nested_loop = re.compile(r"^\s*for .+:\n(?:\s+.+\n)*\s+for .+:", re.MULTILINE)
        if nested_loop.search(file.new_content):
            issues.append(
                self._issue(
                    ReviewCategory.PERFORMANCE,
                    ReviewSeverity.LOW,
                    file.path,
                    self._line_for(file.new_content, "for "),
                    "Potential nested loop hotspot",
                    "Generated code contains nested loops that may be expensive.",
                    "Confirm input sizes or use indexed lookups if this is hot path code.",
                    0.45,
                    "performance",
                )
            )
        return tuple(issues)

    def _detect_breaking_api_changes(self, file: ReviewFile) -> tuple[ReviewIssue, ...]:
        if not file.path.endswith(".py") or not file.old_content or not file.new_content:
            return ()
        try:
            old_tree = ast.parse(file.old_content)
            new_tree = ast.parse(file.new_content)
        except SyntaxError:
            return ()
        old_public = self._public_defs(old_tree)
        new_public = self._public_defs(new_tree)
        removed = sorted(set(old_public) - set(new_public))
        issues = [
            self._issue(
                ReviewCategory.API,
                ReviewSeverity.HIGH,
                file.path,
                old_public[name],
                "Public API removed",
                f"Public symbol '{name}' was removed by the patch.",
                "Preserve public APIs or document and coordinate the breaking change.",
                0.76,
                "api",
            )
            for name in removed
        ]
        return tuple(issues)

    def _public_defs(self, tree: ast.AST) -> dict[str, int]:
        public: dict[str, int] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if not node.name.startswith("_"):
                    public[node.name] = node.lineno
        return public

    def _parse_llm_report(self, raw: str, *, token_usage: dict[str, int]) -> ReviewReport:
        try:
            data = json.loads(self._strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            logger.warning("reviewer.parse_failed", error=str(exc))
            issue = self._issue(
                ReviewCategory.CORRECTNESS,
                ReviewSeverity.HIGH,
                None,
                None,
                "Review response parse failure",
                "The LLM review response was not valid JSON.",
                "Retry review generation or inspect the raw provider response.",
                0.9,
                "llm",
            )
            return ReviewReport(
                recommendation=ReviewRecommendation.REQUEST_CHANGES,
                score=50,
                confidence=0.5,
                summary=f"LLM review output could not be parsed: {exc}",
                issues=(issue,),
                must_fix=(issue.title,),
                token_usage=token_usage,
            )

        if not isinstance(data, Mapping):
            issue = self._issue(
                ReviewCategory.CORRECTNESS,
                ReviewSeverity.HIGH,
                None,
                None,
                "Review response shape failure",
                "The LLM review response was JSON but not a review object.",
                "Retry review generation or inspect the raw provider response.",
                0.9,
                "llm",
            )
            return ReviewReport(
                recommendation=ReviewRecommendation.REQUEST_CHANGES,
                score=50,
                confidence=0.5,
                summary="LLM review output had the wrong JSON shape.",
                issues=(issue,),
                must_fix=(issue.title,),
                token_usage=token_usage,
            )

        issues = tuple(self._parse_issue(issue) for issue in data.get("issues", []))
        recommendation = self._parse_recommendation(data.get("recommendation"))
        confidence = self._clamp_float(data.get("confidence", 0.65))
        return ReviewReport(
            recommendation=recommendation,
            score=self._score(issues),
            confidence=confidence,
            summary=str(data.get("summary", "")),
            issues=issues,
            approved_files=tuple(str(path) for path in data.get("approved_files", [])),
            must_fix=tuple(str(item) for item in data.get("must_fix", [])),
            token_usage=token_usage,
        )

    def _parse_issue(self, value: Any) -> ReviewIssue:
        if not isinstance(value, Mapping):
            return self._issue(
                ReviewCategory.CORRECTNESS,
                ReviewSeverity.MEDIUM,
                None,
                None,
                "Unstructured review issue",
                str(value),
                "Inspect and convert this finding into a concrete issue.",
                0.4,
                "llm",
            )
        return self._issue(
            self._parse_category(value.get("category")),
            self._parse_severity(value.get("severity")),
            str(value.get("file")) if value.get("file") else None,
            self._parse_line(value.get("line")),
            str(value.get("title", "Review issue")),
            str(value.get("description", "")),
            str(value.get("suggestion", "")),
            self._clamp_float(value.get("confidence", 0.65)),
            "llm",
        )

    def _dedupe_issues(self, issues: Sequence[ReviewIssue]) -> tuple[ReviewIssue, ...]:
        seen: set[tuple[str, str | None, int | None, str]] = set()
        deduped: list[ReviewIssue] = []
        for issue in sorted(issues, key=self._issue_sort_key):
            key = (issue.category.value, issue.file, issue.line, issue.title.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return tuple(deduped)

    def _issue_sort_key(self, issue: ReviewIssue) -> tuple[int, str, int]:
        severity_order = {
            ReviewSeverity.CRITICAL: 0,
            ReviewSeverity.HIGH: 1,
            ReviewSeverity.MEDIUM: 2,
            ReviewSeverity.LOW: 3,
            ReviewSeverity.INFO: 4,
        }
        return (severity_order[issue.severity], issue.file or "", issue.line or 0)

    def _score(self, issues: Sequence[ReviewIssue]) -> int:
        penalties = {
            ReviewSeverity.CRITICAL: 35,
            ReviewSeverity.HIGH: 20,
            ReviewSeverity.MEDIUM: 10,
            ReviewSeverity.LOW: 4,
            ReviewSeverity.INFO: 1,
        }
        score = 100 - sum(penalties[issue.severity] for issue in issues)
        return max(0, min(100, score))

    def _recommendation(
        self,
        issues: Sequence[ReviewIssue],
        score: int,
    ) -> ReviewRecommendation:
        if any(issue.severity is ReviewSeverity.CRITICAL for issue in issues):
            return ReviewRecommendation.REJECT
        if any(issue.severity is ReviewSeverity.HIGH for issue in issues) or score < 80:
            return ReviewRecommendation.REQUEST_CHANGES
        return ReviewRecommendation.APPROVE

    def _confidence(
        self,
        issues: Sequence[ReviewIssue],
        llm_report: ReviewReport | None,
    ) -> float:
        scores = [issue.confidence for issue in issues]
        if llm_report:
            scores.append(llm_report.confidence)
        if not scores:
            return 0.72
        return self._clamp_float(sum(scores) / len(scores))

    def _summary(
        self,
        recommendation: ReviewRecommendation,
        score: int,
        issues: Sequence[ReviewIssue],
        llm_report: ReviewReport | None,
    ) -> str:
        if llm_report and llm_report.summary:
            return llm_report.summary
        if not issues:
            return "No blocking review issues were detected in the generated patch."
        counts = Counter(issue.severity.value for issue in issues)
        return (
            f"Recommendation {recommendation.value} with score {score}. "
            f"Detected {len(issues)} issue(s): {dict(counts)}."
        )

    def _planned_files(self, execution_plan: Any) -> set[str]:
        if hasattr(execution_plan, "to_dict"):
            execution_plan = execution_plan.to_dict()
        if not isinstance(execution_plan, Mapping):
            return set()
        plan = execution_plan.get("plan", execution_plan)
        if not isinstance(plan, Mapping):
            return set()
        files: set[str] = set()
        for subtask in plan.get("subtasks", []):
            if isinstance(subtask, Mapping):
                files.update(str(path) for path in subtask.get("likely_files", []))
        return files

    def _context_to_text(self, context: Any | None) -> str:
        if context is None:
            return ""
        if isinstance(context, BuiltContext):
            return context.prompt_context
        if isinstance(context, str):
            return context
        if isinstance(context, Mapping):
            prompt_context = context.get("prompt_context") or context.get("rag_context")
            if isinstance(prompt_context, str):
                return prompt_context
            return self._to_pretty_json(context)
        if hasattr(context, "rag_context"):
            return str(context.rag_context)
        if hasattr(context, "to_dict"):
            return self._to_pretty_json(context.to_dict())
        return str(context)

    def _issue(
        self,
        category: ReviewCategory,
        severity: ReviewSeverity,
        file: str | None,
        line: int | None,
        title: str,
        description: str,
        suggestion: str,
        confidence: float,
        source: str,
    ) -> ReviewIssue:
        return ReviewIssue(
            category=category,
            severity=severity,
            file=file,
            line=line,
            title=title,
            description=description,
            suggestion=suggestion,
            confidence=self._clamp_float(confidence),
            source=source,
        )

    def _parse_category(self, value: Any) -> ReviewCategory:
        try:
            return ReviewCategory(str(value).lower())
        except ValueError:
            return ReviewCategory.CORRECTNESS

    def _parse_severity(self, value: Any) -> ReviewSeverity:
        try:
            return ReviewSeverity(str(value).lower())
        except ValueError:
            return ReviewSeverity.MEDIUM

    def _parse_recommendation(self, value: Any) -> ReviewRecommendation:
        normalized = str(value or "").upper()
        if normalized in {"NEEDS_CHANGES", "REQUEST_CHANGE"}:
            normalized = ReviewRecommendation.REQUEST_CHANGES.value
        try:
            return ReviewRecommendation(normalized)
        except ValueError:
            return ReviewRecommendation.REQUEST_CHANGES

    def _parse_line(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _blocking_severities(self) -> set[ReviewSeverity]:
        return {
            ReviewSeverity.CRITICAL,
            ReviewSeverity.HIGH,
            ReviewSeverity.MEDIUM,
        }

    def _line_for(self, text: str, needle: str) -> int | None:
        for index, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                return index
        return None

    def _line_for_offset(self, text: str, offset: int) -> int:
        return text.count("\n", 0, offset) + 1

    def _strip_json_fence(self, raw: str) -> str:
        clean = raw.strip()
        if not clean.startswith("```"):
            return clean
        first_newline = clean.find("\n")
        if first_newline == -1:
            return clean.strip("`")
        clean = clean[first_newline + 1 :]
        if clean.endswith("```"):
            clean = clean[:-3]
        return clean.strip()

    def _to_pretty_json(self, value: Any) -> str:
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(value)

    def _clamp_float(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return max(0.0, min(1.0, number))

    def _first_present(self, mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None


ReviewerAgent = ReviewAgent
