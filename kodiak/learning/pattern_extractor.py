from __future__ import annotations

import ast
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from kodiak.learning.pattern_store import Pattern, PatternStore, PatternType

logger = logging.getLogger(__name__)


class ExecutionOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class ExecutionRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    repo: str
    language: str
    code_before: str | None = None
    code_after: str
    outcome: ExecutionOutcome
    error_message: str | None = None
    error_type: str | None = None
    execution_time_ms: float | None = None
    test_results: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractedPattern(BaseModel):
    name: str
    description: str
    pattern_type: PatternType
    language: str
    tags: list[str]
    code_template: str | None = None
    example_before: str | None = None
    example_after: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source_records: list[str] = Field(default_factory=list)


class ASTNodeVisitor(ast.NodeVisitor):
    """Collect structural signals from Python AST."""

    def __init__(self) -> None:
        self.function_defs: list[str] = []
        self.class_defs: list[str] = []
        self.imports: list[str] = []
        self.decorators: list[str] = []
        self.async_functions: list[str] = []
        self.exception_handlers: list[str] = []
        self.comprehensions: int = 0
        self.context_managers: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_defs.append(node.name)
        self._collect_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.async_functions.append(node.name)
        self._collect_decorators(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_defs.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        exc_type = node.type.id if isinstance(node.type, ast.Name) else "Exception"
        self.exception_handlers.append(exc_type)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.context_managers += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.context_managers += 1
        self.generic_visit(node)

    def _collect_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                self.decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                self.decorators.append(dec.attr)


class PatternExtractor:
    _ANTI_PATTERN_SIGNALS: dict[str, str] = {
        r"\bexcept\s*:\s*pass\b": "bare except suppression",
        r"eval\s*\(": "dangerous eval usage",
        r"exec\s*\(": "dangerous exec usage",
        r"time\.sleep\s*\(\s*[5-9]\d+": "long blocking sleep",
        r"SELECT \*": "SELECT * anti-pattern",
        r"global\s+\w+": "mutable global state",
        r"import \*": "wildcard import",
    }

    _GOOD_PATTERN_SIGNALS: dict[str, str] = {
        r"async def .+:\s": "async function definition",
        r"with\s+\w+.*as\s+\w+": "context manager usage",
        r"@(property|staticmethod|classmethod)": "property/static/classmethod decorator",
        r"\[.+ for .+ in .+\]": "list comprehension",
        r"@(retry|backoff|lru_cache|cache)": "retry/caching decorator",
        r"TypeVar|Generic\[": "generic typing",
        r"Protocol\b": "protocol-based interface",
        r"dataclass|BaseModel": "structured data model",
    }

    def __init__(self, store: PatternStore) -> None:
        self._store = store

    async def process_execution(self, record: ExecutionRecord) -> list[Pattern]:
        saved: list[Pattern] = []

        if record.outcome == ExecutionOutcome.SUCCESS:
            extracted = await self._extract_from_success(record)
        elif record.outcome == ExecutionOutcome.FAILURE:
            extracted = await self._extract_from_failure(record)
        else:
            extracted = await self._extract_from_partial(record)

        for ep in extracted:
            pattern = self._to_pattern(ep, record.repo)
            saved_pattern = await self._store.create(pattern)
            saved.append(saved_pattern)

        logger.info(
            "Extracted %d patterns from execution %s (outcome=%s)",
            len(saved),
            record.id,
            record.outcome,
        )
        return saved

    async def process_batch(self, records: list[ExecutionRecord]) -> list[Pattern]:
        all_patterns: list[Pattern] = []
        grouped = self._group_by_language(records)

        for language, lang_records in grouped.items():
            cross_patterns = await self._extract_cross_execution_patterns(lang_records)
            for ep in cross_patterns:
                pattern = self._to_pattern(ep, source_repo=None)
                all_patterns.append(await self._store.create(pattern))

        for record in records:
            all_patterns.extend(await self.process_execution(record))

        return all_patterns

    async def _extract_from_success(self, record: ExecutionRecord) -> list[ExtractedPattern]:
        patterns: list[ExtractedPattern] = []

        code_signals = self._scan_code_signals(record.code_after, self._GOOD_PATTERN_SIGNALS)
        for signal, label in code_signals:
            patterns.append(
                ExtractedPattern(
                    name=f"{label} in {record.language}",
                    description=f"Successful usage of {label} detected in task {record.task_id}.",
                    pattern_type=PatternType.CODING,
                    language=record.language,
                    tags=[label.replace(" ", "_"), record.language, "successful"],
                    code_template=self._extract_snippet(record.code_after, signal),
                    example_after=record.code_after,
                    example_before=record.code_before,
                    context={"signal": signal, "task_id": record.task_id},
                    confidence=0.75,
                    source_records=[record.id],
                )
            )

        if record.language == "python":
            structural = self._extract_python_structural(record.code_after, record)
            patterns.extend(structural)

        if record.code_before and record.code_after:
            refactor = self._detect_refactoring(record)
            if refactor:
                patterns.append(refactor)

        return patterns

    async def _extract_from_failure(self, record: ExecutionRecord) -> list[ExtractedPattern]:
        patterns: list[ExtractedPattern] = []

        anti_signals = self._scan_code_signals(record.code_after, self._ANTI_PATTERN_SIGNALS)
        for signal, label in anti_signals:
            patterns.append(
                ExtractedPattern(
                    name=f"Anti-pattern: {label}",
                    description=(
                        f"Detected {label} associated with execution failure in task {record.task_id}. "
                        f"Error: {record.error_message or 'unknown'}"
                    ),
                    pattern_type=PatternType.ANTI_PATTERN,
                    language=record.language,
                    tags=[label.replace(" ", "_"), record.language, "anti_pattern", "failure"],
                    code_template=self._extract_snippet(record.code_after, signal),
                    example_before=record.code_after,
                    context={
                        "signal": signal,
                        "error_type": record.error_type,
                        "error_message": record.error_message,
                        "task_id": record.task_id,
                    },
                    confidence=0.8,
                    source_records=[record.id],
                )
            )

        if record.error_type:
            error_pattern = self._build_error_pattern(record)
            if error_pattern:
                patterns.append(error_pattern)

        return patterns

    async def _extract_from_partial(self, record: ExecutionRecord) -> list[ExtractedPattern]:
        success_parts = await self._extract_from_success(record)
        failure_parts = await self._extract_from_failure(record)
        for p in success_parts:
            p.confidence *= 0.6
        for p in failure_parts:
            p.confidence *= 0.6
        return success_parts + failure_parts

    def _extract_python_structural(
        self, code: str, record: ExecutionRecord
    ) -> list[ExtractedPattern]:
        patterns: list[ExtractedPattern] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return patterns

        visitor = ASTNodeVisitor()
        visitor.visit(tree)

        if visitor.async_functions and "asyncpg" in "\n".join(visitor.imports):
            patterns.append(
                ExtractedPattern(
                    name="Async database access pattern",
                    description="Async functions with asyncpg imports for non-blocking DB access.",
                    pattern_type=PatternType.PERFORMANCE,
                    language="python",
                    tags=["async", "asyncpg", "database", "non_blocking"],
                    code_template=self._first_function_snippet(code, visitor.async_functions),
                    context={"async_fns": visitor.async_functions, "task_id": record.task_id},
                    confidence=0.85,
                    source_records=[record.id],
                )
            )

        if visitor.context_managers >= 2:
            patterns.append(
                ExtractedPattern(
                    name="Multi context manager resource management",
                    description="Consistent use of context managers for safe resource handling.",
                    pattern_type=PatternType.CODING,
                    language="python",
                    tags=["context_manager", "resource_management", "safe"],
                    context={"count": visitor.context_managers, "task_id": record.task_id},
                    confidence=0.7,
                    source_records=[record.id],
                )
            )

        if "Protocol" in visitor.imports or any("Protocol" in c for c in visitor.class_defs):
            patterns.append(
                ExtractedPattern(
                    name="Protocol-based interface definition",
                    description="Structural subtyping via Protocol for loose coupling.",
                    pattern_type=PatternType.ARCHITECTURAL,
                    language="python",
                    tags=["protocol", "interface", "structural_typing", "solid"],
                    confidence=0.9,
                    source_records=[record.id],
                )
            )

        return patterns

    def _detect_refactoring(self, record: ExecutionRecord) -> ExtractedPattern | None:
        before = record.code_before or ""
        after = record.code_after

        before_lines = set(before.splitlines())
        after_lines = set(after.splitlines())
        removed = before_lines - after_lines
        added = after_lines - before_lines

        if not removed or not added:
            return None

        reduction = (len(before_lines) - len(after_lines)) / max(len(before_lines), 1)
        if reduction > 0.2:
            return ExtractedPattern(
                name=f"Code reduction refactoring ({record.language})",
                description=f"Reduced code by {reduction:.0%} while maintaining functionality.",
                pattern_type=PatternType.REFACTORING,
                language=record.language,
                tags=["refactoring", "code_reduction", record.language],
                example_before=before,
                example_after=after,
                context={
                    "lines_removed": len(removed),
                    "lines_added": len(added),
                    "reduction_pct": round(reduction, 4),
                    "task_id": record.task_id,
                },
                confidence=0.65,
                source_records=[record.id],
            )
        return None

    def _build_error_pattern(self, record: ExecutionRecord) -> ExtractedPattern | None:
        if not record.error_type:
            return None

        error_map: dict[str, tuple[str, list[str]]] = {
            "ImportError": ("Missing dependency or incorrect import path", ["import", "dependency"]),
            "ModuleNotFoundError": ("Module not found at runtime", ["import", "module_missing"]),
            "AttributeError": ("Attribute access on wrong type or None", ["type_error", "none_check"]),
            "KeyError": ("Unguarded dict key access", ["dict_safety", "key_guard"]),
            "TypeError": ("Type mismatch in function call or operation", ["type_safety"]),
            "asyncio.TimeoutError": ("Unhandled async timeout", ["async", "timeout", "resilience"]),
            "RecursionError": ("Unbounded recursion depth", ["recursion", "stack_overflow"]),
        }

        if record.error_type not in error_map:
            return None

        desc, tags = error_map[record.error_type]
        return ExtractedPattern(
            name=f"Anti-pattern: {record.error_type} vulnerability",
            description=f"{desc}. Observed in task {record.task_id}.",
            pattern_type=PatternType.ANTI_PATTERN,
            language=record.language,
            tags=tags + ["anti_pattern", record.error_type.lower()],
            example_before=record.code_after,
            context={
                "error_type": record.error_type,
                "error_message": record.error_message,
                "task_id": record.task_id,
            },
            confidence=0.88,
            source_records=[record.id],
        )

    async def _extract_cross_execution_patterns(
        self, records: list[ExecutionRecord]
    ) -> list[ExtractedPattern]:
        patterns: list[ExtractedPattern] = []
        if len(records) < 3:
            return patterns

        signal_counts: dict[str, list[str]] = defaultdict(list)
        for record in records:
            if record.outcome != ExecutionOutcome.SUCCESS:
                continue
            for signal, label in self._scan_code_signals(record.code_after, self._GOOD_PATTERN_SIGNALS):
                signal_counts[label].append(record.id)

        for label, record_ids in signal_counts.items():
            frequency = len(record_ids) / len(records)
            if frequency >= 0.5:
                patterns.append(
                    ExtractedPattern(
                        name=f"Recurring pattern: {label}",
                        description=(
                            f"{label} appears in {frequency:.0%} of successful executions "
                            f"across {len(records)} records."
                        ),
                        pattern_type=PatternType.CODING,
                        language=records[0].language,
                        tags=[label.replace(" ", "_"), "recurring", "high_frequency"],
                        context={"frequency": round(frequency, 4), "sample_size": len(records)},
                        confidence=min(0.95, 0.6 + frequency * 0.4),
                        source_records=record_ids[:10],
                    )
                )

        return patterns

    @staticmethod
    def _scan_code_signals(
        code: str, signal_map: dict[str, str]
    ) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for pattern, label in signal_map.items():
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                found.append((pattern, label))
        return found

    @staticmethod
    def _extract_snippet(code: str, signal_pattern: str, context_lines: int = 3) -> str | None:
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if re.search(signal_pattern, line, re.IGNORECASE):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return "\n".join(lines[start:end])
        return None

    @staticmethod
    def _first_function_snippet(code: str, fn_names: list[str]) -> str | None:
        if not fn_names:
            return None
        target = fn_names[0]
        lines = code.splitlines()
        capturing = False
        snippet: list[str] = []
        for line in lines:
            if re.match(rf"\s*(async\s+)?def\s+{re.escape(target)}\s*\(", line):
                capturing = True
            if capturing:
                snippet.append(line)
                if len(snippet) > 20:
                    break
        return "\n".join(snippet) if snippet else None

    @staticmethod
    def _group_by_language(
        records: list[ExecutionRecord],
    ) -> dict[str, list[ExecutionRecord]]:
        grouped: dict[str, list[ExecutionRecord]] = defaultdict(list)
        for r in records:
            grouped[r.language].append(r)
        return dict(grouped)

    @staticmethod
    def _to_pattern(ep: ExtractedPattern, source_repo: str | None) -> Pattern:
        return Pattern(
            name=ep.name,
            description=ep.description,
            pattern_type=ep.pattern_type,
            language=ep.language,
            tags=ep.tags,
            code_template=ep.code_template,
            example_before=ep.example_before,
            example_after=ep.example_after,
            context={**ep.context, "confidence": ep.confidence, "source_records": ep.source_records},
            source_repo=source_repo,
        )