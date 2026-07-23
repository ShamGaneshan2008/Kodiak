"""Coding agent for Kodiak's autonomous engineering workflow.

The coding agent is the third AI agent in the planner-driven workflow. It
consumes execution plans from ``PlannerAgent``, research reports from
``ResearchAgent``, and repository context built by ``ContextBuilder``. It does
not perform repository analysis, retrieval, test execution, review, commits, or
GitHub operations.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent
from kodiak.rag.context_builder import BuiltContext, ContextBuilder
from kodiak.utils.diff import unified_diff

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = """\
You are a senior software engineer inside the Kodiak autonomous engineering
system. Your job is to generate production-ready code changes from an execution
plan, research report, and repository context that were already prepared by
other Kodiak services.

Rules:
- Consume the supplied plan, research report, and context; do not invent
  repository analysis or ask for retrieval.
- Preserve the existing project architecture, imports, style, and dependency
  boundaries.
- Write complete file contents for every changed file; never use ellipses,
  placeholders, or TODOs.
- Change only files needed by the execution plan.
- Do not run tests, review code, commit changes, push changes, or execute
  repository code.
- Output ONLY valid JSON; no prose and no markdown fences.

Output schema:
{
  "implementation_plan": [
    {
      "step": "<short implementation step>",
      "files": ["relative/path.py"],
      "reason": "<why this step is needed>"
    }
  ],
  "files": [
    {
      "path": "relative/path.py",
      "action": "create | modify | delete",
      "content": "<complete new file content for create/modify; empty for delete>",
      "explanation": "<why this file changes>"
    }
  ],
  "explanation": "<brief summary of the change set>"
}
"""


class PatchAction(StrEnum):
    """Supported repository file patch actions."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


@dataclass(frozen=True)
class ImplementationStep:
    """One step in the coding agent's implementation plan.

    Attributes:
        step: Short description of the work to perform.
        files: Relative repository paths touched by the step.
        reason: Why the step is required.
    """

    step: str
    files: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the step."""
        return {"step": self.step, "files": list(self.files), "reason": self.reason}


@dataclass(frozen=True)
class GeneratedFile:
    """LLM-generated file specification before diff construction.

    Attributes:
        path: Repository-relative file path.
        action: Requested file action.
        content: Complete target content for create or modify actions.
        explanation: Human-readable rationale for the file change.
    """

    path: str
    action: PatchAction
    content: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the generated file."""
        return {
            "path": self.path,
            "action": self.action.value,
            "content": self.content,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class CodeGeneration:
    """Structured output parsed from the coding LLM response.

    Attributes:
        implementation_plan: Ordered implementation steps.
        files: Generated file contents.
        explanation: Summary of the proposed change set.
        raw_response: Raw LLM response content for diagnostics.
        token_usage: Provider token usage metadata.
        provider_metadata: Provider and model metadata.
    """

    implementation_plan: tuple[ImplementationStep, ...]
    files: tuple[GeneratedFile, ...]
    explanation: str
    raw_response: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable representation of generated code."""
        data: dict[str, Any] = {
            "implementation_plan": [step.to_dict() for step in self.implementation_plan],
            "files": [file.to_dict() for file in self.files],
            "explanation": self.explanation,
            "token_usage": self.token_usage,
            "provider_metadata": self.provider_metadata,
        }
        if include_raw:
            data["raw_response"] = self.raw_response
        return data


@dataclass(frozen=True)
class FilePatch:
    """Structured patch for one repository file.

    Attributes:
        path: Repository-relative file path.
        action: File action to apply.
        old_content: Existing file content, empty for creates.
        new_content: Target file content, empty for deletes.
        unified_diff: Unified diff for display and review.
        explanation: Why the file changes.
    """

    path: str
    action: PatchAction
    old_content: str
    new_content: str
    unified_diff: str
    explanation: str = ""

    @property
    def has_changes(self) -> bool:
        """Return whether this file patch changes content."""
        return self.old_content != self.new_content

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this file patch."""
        return {
            "path": self.path,
            "action": self.action.value,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "unified_diff": self.unified_diff,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class CodePatch:
    """Complete structured code patch generated by the coding agent.

    Attributes:
        files: File patches in application order.
        implementation_plan: Coding implementation plan.
        explanation: Summary of the patch.
        token_usage: Provider token usage metadata.
        provider_metadata: Provider and model metadata.
    """

    files: tuple[FilePatch, ...]
    implementation_plan: tuple[ImplementationStep, ...] = ()
    explanation: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def unified_diff(self) -> str:
        """Return all file diffs concatenated into one unified diff."""
        return "\n".join(file.unified_diff for file in self.files if file.unified_diff)

    @property
    def changed_files(self) -> tuple[str, ...]:
        """Return repository-relative paths changed by this patch."""
        return tuple(file.path for file in self.files if file.has_changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this patch."""
        return {
            "files": [file.to_dict() for file in self.files],
            "implementation_plan": [step.to_dict() for step in self.implementation_plan],
            "explanation": self.explanation,
            "unified_diff": self.unified_diff,
            "changed_files": list(self.changed_files),
            "token_usage": self.token_usage,
            "provider_metadata": self.provider_metadata,
        }


@dataclass(frozen=True)
class PatchValidationResult:
    """Validation result for a structured patch.

    Attributes:
        valid: Whether the patch is safe to apply.
        errors: Blocking validation errors.
        warnings: Non-blocking validation warnings.
    """

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the validation."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PatchApplicationResult:
    """Result from applying or dry-running a structured patch.

    Attributes:
        applied: Whether files were written.
        dry_run: Whether application was simulated.
        changed_files: Repository-relative paths that would change or changed.
        validation: Patch validation result.
    """

    applied: bool
    dry_run: bool
    changed_files: tuple[str, ...]
    validation: PatchValidationResult

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the application."""
        return {
            "applied": self.applied,
            "dry_run": self.dry_run,
            "changed_files": list(self.changed_files),
            "validation": self.validation.to_dict(),
        }


class CodingAgent(BaseAgent):
    """Generate and optionally apply code patches from prepared Kodiak context.

    The agent is intentionally patch-first: it asks the configured LLM provider
    for complete target file contents, derives unified diffs deterministically,
    validates the patch against the repository root, and only writes files when
    ``dry_run`` is disabled.
    """

    role = AgentRole.CODER

    def __init__(
        self,
        llm_client: Any,
        sandbox: Any | None = None,
        *,
        repository_root: str | Path | None = None,
        context_builder: ContextBuilder | None = None,
        default_model_preference: str = "default",
        max_output_tokens: int = 12000,
    ) -> None:
        """Initialize the coding agent.

        Args:
            llm_client: Injected LLM facade or compatible provider.
            sandbox: Deprecated compatibility parameter. The coding agent does
                not execute repository code or run tests.
            repository_root: Repository root used for path validation and writes.
            context_builder: Optional ContextBuilder used only when callers
                request context construction explicitly.
            default_model_preference: LLM routing preference.
            max_output_tokens: Maximum tokens requested for code generation.
        """
        super().__init__()
        self._llm = llm_client
        self._sandbox = sandbox
        self._repository_root = Path(repository_root).resolve() if repository_root else None
        self._context_builder = context_builder
        self._default_model_preference = default_model_preference
        self._max_output_tokens = max_output_tokens

    async def _run(self, input_: AgentInput) -> AgentOutput:
        """Run the coding agent through BaseAgent orchestration."""
        root = self._resolve_repository_root(input_.context.get("work_dir"))
        dry_run = bool(input_.context.get("dry_run", True))

        patch = await self.generate_patch(
            instruction=input_.instruction,
            execution_plan=self._first_present(
                input_.context,
                "execution_plan",
                "plan",
                "task_plan",
            ),
            research_report=input_.context.get("research_report"),
            repository_context=self._first_present(
                input_.context,
                "repository_context",
                "built_context",
                "context",
                "rag_context",
            ),
            repository_root=root,
            target_files=input_.context.get("target_files"),
            reviewer_feedback=input_.context.get("reviewer_feedback", ""),
            model_preference=input_.context.get(
                "model_preference",
                self._default_model_preference,
            ),
        )
        validation = self.validate_patch(patch, repository_root=root)

        application = PatchApplicationResult(
            applied=False,
            dry_run=dry_run,
            changed_files=patch.changed_files,
            validation=validation,
        )
        if validation.valid and not dry_run:
            application = self.apply_patch(patch, repository_root=root, dry_run=False)

        return self._make_output(
            input_,
            result={
                "patch": patch.to_dict(),
                "validation": validation.to_dict(),
                "application": application.to_dict(),
                "explanation": self.explain_changes(patch),
            },
            token_usage=patch.token_usage,
            metadata={"dry_run": dry_run, "repository_root": str(root) if root else None},
        )

    async def generate_code(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        context_query: str | None = None,
        target_files: Sequence[str] | None = None,
        reviewer_feedback: str = "",
        model_preference: str | None = None,
    ) -> CodeGeneration:
        """Generate structured file contents from prepared agent inputs.

        Args:
            instruction: User or workflow instruction to implement.
            execution_plan: PlannerAgent output or normalized execution plan.
            research_report: ResearchAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            context_query: Optional task text used with the injected
                ContextBuilder when repository_context is not supplied.
            target_files: Optional files the caller expects the agent to touch.
            reviewer_feedback: Optional reviewer feedback to address.
            model_preference: Optional LLM routing preference.

        Returns:
            Parsed and validated LLM code-generation output.

        Raises:
            ValueError: If the LLM response is malformed.
        """
        context_text = await self._repository_context_text(repository_context, context_query)
        user_message = self._build_user_message(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=context_text,
            target_files=target_files or (),
            reviewer_feedback=reviewer_feedback,
        )
        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            model_preference=model_preference or self._default_model_preference,
            max_tokens=self._max_output_tokens,
            temperature=0.1,
        )

        raw = str(response.get("content", ""))
        generation = self._parse_generation(raw)
        return CodeGeneration(
            implementation_plan=generation.implementation_plan,
            files=generation.files,
            explanation=generation.explanation,
            raw_response=raw,
            token_usage=dict(response.get("usage", {})),
            provider_metadata={
                "model": response.get("model"),
                "provider": response.get("provider"),
                "stop_reason": response.get("stop_reason"),
            },
        )

    async def stream_code(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        context_query: str | None = None,
        target_files: Sequence[str] | None = None,
        reviewer_feedback: str = "",
        model_preference: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream raw code-generation JSON chunks from the configured LLM.

        Args mirror :meth:`generate_code`. The caller is responsible for
        assembling and parsing the streamed JSON with :meth:`generate_patch`
        semantics if needed.

        Yields:
            Raw text chunks from the LLM provider.
        """
        context_text = await self._repository_context_text(repository_context, context_query)
        user_message = self._build_user_message(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=context_text,
            target_files=target_files or (),
            reviewer_feedback=reviewer_feedback,
        )
        async for chunk in self._llm.stream(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            model_preference=model_preference or self._default_model_preference,
            max_tokens=self._max_output_tokens,
            temperature=0.1,
        ):
            yield chunk

    async def generate_patch(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        repository_root: str | Path | None = None,
        research_report: Any | None = None,
        repository_context: Any | None = None,
        context_query: str | None = None,
        target_files: Sequence[str] | None = None,
        reviewer_feedback: str = "",
        model_preference: str | None = None,
    ) -> CodePatch:
        """Generate a structured patch and unified diffs without writing files.

        Args:
            instruction: User or workflow instruction to implement.
            execution_plan: PlannerAgent output or normalized execution plan.
            repository_root: Root used to read existing file contents.
            research_report: ResearchAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            context_query: Optional query used with an injected ContextBuilder.
            target_files: Optional expected files to touch.
            reviewer_feedback: Optional reviewer feedback to address.
            model_preference: Optional LLM routing preference.

        Returns:
            Structured patch containing per-file unified diffs.
        """
        root = self._resolve_repository_root(repository_root)
        generation = await self.generate_code(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            repository_context=repository_context,
            context_query=context_query,
            target_files=target_files,
            reviewer_feedback=reviewer_feedback,
            model_preference=model_preference,
        )

        patches = tuple(
            self._file_patch_from_generated(file, repository_root=root) for file in generation.files
        )
        return CodePatch(
            files=patches,
            implementation_plan=generation.implementation_plan,
            explanation=generation.explanation,
            token_usage=generation.token_usage,
            provider_metadata=generation.provider_metadata,
        )

    def create_file(
        self,
        path: str,
        content: str,
        *,
        explanation: str = "",
        repository_root: str | Path | None = None,
    ) -> FilePatch:
        """Create a structured patch for a new file without writing it.

        Args:
            path: Repository-relative file path.
            content: Complete file content to create.
            explanation: Rationale for the new file.
            repository_root: Root used to validate existence.

        Returns:
            FilePatch for the create action.
        """
        return self._file_patch_from_generated(
            GeneratedFile(
                path=path,
                action=PatchAction.CREATE,
                content=content,
                explanation=explanation,
            ),
            repository_root=self._resolve_repository_root(repository_root),
        )

    def modify_file(
        self,
        path: str,
        content: str,
        *,
        explanation: str = "",
        repository_root: str | Path | None = None,
    ) -> FilePatch:
        """Create a structured patch for an existing file without writing it.

        Args:
            path: Repository-relative file path.
            content: Complete replacement file content.
            explanation: Rationale for the modification.
            repository_root: Root used to read existing content.

        Returns:
            FilePatch for the modify action.
        """
        return self._file_patch_from_generated(
            GeneratedFile(
                path=path,
                action=PatchAction.MODIFY,
                content=content,
                explanation=explanation,
            ),
            repository_root=self._resolve_repository_root(repository_root),
        )

    def explain_changes(self, patch: CodePatch) -> str:
        """Return a concise explanation for a generated patch.

        Args:
            patch: Structured patch to summarize.

        Returns:
            Human-readable patch explanation.
        """
        if not patch.files:
            return patch.explanation or "No file changes were generated."

        lines = [patch.explanation.strip()] if patch.explanation.strip() else []
        for file in patch.files:
            detail = file.explanation.strip() or f"{file.action.value} {file.path}"
            lines.append(f"- {file.path}: {detail}")
        return "\n".join(lines)

    def validate_patch(
        self,
        patch: CodePatch,
        *,
        repository_root: str | Path | None = None,
    ) -> PatchValidationResult:
        """Validate that a structured patch is safe to apply.

        Args:
            patch: Structured patch to validate.
            repository_root: Repository root for path and file-state checks.

        Returns:
            Validation result with blocking errors and warnings.
        """
        root = self._resolve_repository_root(repository_root)
        errors: list[str] = []
        warnings: list[str] = []
        seen_paths: set[str] = set()

        if root is None:
            errors.append("repository_root is required to validate file patches")
            return PatchValidationResult(valid=False, errors=tuple(errors))

        for file in patch.files:
            if file.path in seen_paths:
                errors.append(f"{file.path}: duplicate patch entry")
            seen_paths.add(file.path)

            try:
                absolute_path = self._safe_path(root, file.path)
            except ValueError as exc:
                errors.append(f"{file.path}: {exc}")
                continue

            exists = absolute_path.exists()
            if file.action is PatchAction.CREATE and exists:
                errors.append(f"{file.path}: create action would overwrite an existing file")
            elif file.action is PatchAction.MODIFY and not exists:
                errors.append(f"{file.path}: modify action requires an existing file")
            elif file.action is PatchAction.DELETE and not exists:
                errors.append(f"{file.path}: delete action requires an existing file")

            if file.action is not PatchAction.DELETE and not file.new_content:
                warnings.append(f"{file.path}: generated content is empty")
            if not file.has_changes:
                warnings.append(f"{file.path}: patch does not change file content")

        return PatchValidationResult(
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def apply_patch(
        self,
        patch: CodePatch,
        *,
        repository_root: str | Path | None = None,
        dry_run: bool = False,
    ) -> PatchApplicationResult:
        """Apply a validated structured patch or simulate application.

        Args:
            patch: Structured patch to apply.
            repository_root: Repository root for writes.
            dry_run: When true, validate and report changes without writing.

        Returns:
            Patch application result.

        Raises:
            ValueError: If validation fails.
            OSError: If a filesystem write fails.
        """
        root = self._resolve_repository_root(repository_root)
        validation = self.validate_patch(patch, repository_root=root)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        if dry_run:
            return PatchApplicationResult(
                applied=False,
                dry_run=True,
                changed_files=patch.changed_files,
                validation=validation,
            )
        if root is None:
            raise ValueError("repository_root is required to apply a patch")

        for file in patch.files:
            absolute_path = self._safe_path(root, file.path)
            if file.action is PatchAction.DELETE:
                absolute_path.unlink()
                continue
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_path.write_text(file.new_content, encoding="utf-8")

        logger.info("coding_agent.patch_applied", changed_files=list(patch.changed_files))
        return PatchApplicationResult(
            applied=True,
            dry_run=False,
            changed_files=patch.changed_files,
            validation=validation,
        )

    def dry_run(
        self,
        patch: CodePatch,
        *,
        repository_root: str | Path | None = None,
    ) -> PatchApplicationResult:
        """Validate a patch without modifying repository files.

        Args:
            patch: Structured patch to simulate.
            repository_root: Repository root for validation.

        Returns:
            Dry-run application result.
        """
        return self.apply_patch(patch, repository_root=repository_root, dry_run=True)

    def _build_user_message(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        research_report: Any | None,
        repository_context: str,
        target_files: Sequence[str],
        reviewer_feedback: str,
    ) -> str:
        parts = [
            f"## Task\n{instruction}",
            f"## Execution plan from PlannerAgent\n{self._to_pretty_json(execution_plan)}",
        ]
        if research_report:
            parts.append(
                f"## Research report from ResearchAgent\n{self._to_pretty_json(research_report)}"
            )
        if repository_context:
            parts.append(f"## Context from ContextBuilder\n{repository_context}")
        if target_files:
            parts.append(
                "## Expected target files\n"
                f"{json.dumps(list(target_files), indent=2, sort_keys=True)}"
            )
        if reviewer_feedback:
            parts.append(f"## Reviewer feedback to address\n{reviewer_feedback}")
        return "\n\n".join(parts)

    async def _repository_context_text(
        self,
        repository_context: Any | None,
        context_query: str | None,
    ) -> str:
        if repository_context is not None:
            return self._context_to_text(repository_context)
        if context_query and self._context_builder is not None:
            built_context = await self._context_builder.build_task_context(context_query)
            return built_context.prompt_context
        return ""

    def _context_to_text(self, context: Any) -> str:
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

    def _file_patch_from_generated(
        self,
        file: GeneratedFile,
        *,
        repository_root: Path | None,
    ) -> FilePatch:
        old_content = ""
        if repository_root is not None:
            absolute_path = self._safe_path(repository_root, file.path)
            if absolute_path.exists() and file.action is not PatchAction.CREATE:
                old_content = absolute_path.read_text(encoding="utf-8")
        new_content = "" if file.action is PatchAction.DELETE else file.content
        diff = unified_diff(
            old_content,
            new_content,
            old_name=f"a/{file.path}",
            new_name=f"b/{file.path}",
        ).unified_diff
        return FilePatch(
            path=file.path,
            action=file.action,
            old_content=old_content,
            new_content=new_content,
            unified_diff=diff,
            explanation=file.explanation,
        )

    def _parse_generation(self, raw: str) -> CodeGeneration:
        try:
            data = json.loads(self._strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            logger.warning("coding_agent.parse_failed", error=str(exc))
            raise ValueError(f"Failed to parse code generation JSON: {exc}") from exc

        if not isinstance(data, Mapping):
            raise ValueError("Code generation response must be a JSON object")
        file_specs = data.get("files")
        if not isinstance(file_specs, list):
            raise ValueError("Code generation response must include a files list")

        steps = tuple(self._parse_step(item) for item in data.get("implementation_plan", []))
        files = tuple(self._parse_file(item) for item in file_specs)
        return CodeGeneration(
            implementation_plan=steps,
            files=files,
            explanation=str(data.get("explanation", "")),
            raw_response=raw,
        )

    def _parse_step(self, item: Any) -> ImplementationStep:
        if not isinstance(item, Mapping):
            return ImplementationStep(step=str(item))
        files = item.get("files", ())
        if not isinstance(files, Sequence) or isinstance(files, str):
            files = ()
        return ImplementationStep(
            step=str(item.get("step", "")),
            files=tuple(str(path) for path in files),
            reason=str(item.get("reason", "")),
        )

    def _parse_file(self, item: Any) -> GeneratedFile:
        if not isinstance(item, Mapping):
            raise ValueError("Each file entry must be a JSON object")
        raw_action = str(item.get("action", PatchAction.MODIFY.value)).lower()
        try:
            action = PatchAction(raw_action)
        except ValueError as exc:
            raise ValueError(f"Unsupported patch action: {raw_action}") from exc

        path = str(item.get("path", "")).strip()
        if not path:
            raise ValueError("File entry is missing path")
        return GeneratedFile(
            path=path,
            action=action,
            content=str(item.get("content", "")),
            explanation=str(item.get("explanation", "")),
        )

    def _resolve_repository_root(self, override: str | Path | None = None) -> Path | None:
        if override:
            return Path(override).resolve()
        return self._repository_root

    def _safe_path(self, root: Path, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("absolute paths are not allowed")
        if any(part == ".." for part in path.parts):
            raise ValueError("parent directory traversal is not allowed")
        absolute_path = (root / path).resolve()
        try:
            absolute_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes repository root") from exc
        return absolute_path

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

    def _first_present(self, mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None


CoderAgent = CodingAgent
