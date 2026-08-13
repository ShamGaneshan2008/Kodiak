"""Failure analysis and categorization from execution/verification evidence."""

from __future__ import annotations

from typing import Any

from kodiak.orchestration.execution.models import ExecutionOutcome
from kodiak.orchestration.reflection.models import (
    FailureCategory,
    ReflectionContext,
    ReflectionOutcome,
    ReflectionResult,
    RepairStrategy,
)
from kodiak.orchestration.verification.models import VerificationStatus


class FailureAnalyzer:
    """Derives structured failure diagnosis from available evidence."""

    def analyze(self, context: ReflectionContext) -> ReflectionResult:
        """Analyze failure evidence and produce a reflection result."""
        if context.execution_result.outcome is ExecutionOutcome.SUCCESS:
            verification = context.verification_result
            if verification is None or verification.status is VerificationStatus.VERIFIED:
                return ReflectionResult(
                    outcome=ReflectionOutcome.SUCCESS,
                    category=FailureCategory.UNKNOWN,
                    strategy=RepairStrategy.STOP,
                    root_cause="Task completed and verified successfully.",
                    suggested_correction="No corrective action required.",
                    should_retry=False,
                    attempt=context.attempt,
                    max_attempts=context.max_attempts,
                    summary="Success",
                    confidence=1.0,
                )

        evidence = self._collect_evidence(context)
        category = self._categorize(context, evidence)
        root_cause = self._root_cause(context, category, evidence)
        correction = self._suggested_correction(category, evidence)
        strategy, outcome = self._strategy_for(category, context, evidence)

        affected_files = tuple(evidence.get("affected_files", ()))
        affected_tool = evidence.get("affected_tool")

        return ReflectionResult(
            outcome=outcome,
            category=category,
            strategy=strategy,
            root_cause=root_cause,
            suggested_correction=correction,
            evidence=evidence,
            affected_files=affected_files,
            affected_tool=affected_tool,
            confidence=self._confidence(category, evidence),
            should_retry=strategy is RepairStrategy.RETRY,
            replan_required=strategy is RepairStrategy.REPLAN,
            attempt=context.attempt,
            max_attempts=context.max_attempts,
            summary=evidence.get("summary"),
        )

    def _collect_evidence(self, context: ReflectionContext) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        execution_error = context.execution_result.error or {}
        evidence["execution_error"] = execution_error
        evidence["execution_outcome"] = context.execution_result.outcome.value

        verification_dict = context.execution_result.verification
        if verification_dict:
            evidence["verification"] = verification_dict
        elif context.verification_result is not None:
            evidence["verification"] = context.verification_result.to_dict()

        verification = evidence.get("verification", {})
        failed_items = [
            item
            for item in verification.get("evidence", [])
            if item.get("status") == VerificationStatus.FAILED.value
        ]
        if failed_items:
            evidence["failed_verifiers"] = failed_items
            evidence["summary"] = failed_items[0].get("message") or verification.get("summary")
            files: list[str] = []
            for item in failed_items:
                files.extend(item.get("files_checked", []))
                files.extend(item.get("artifacts_checked", []))
            if files:
                evidence["affected_files"] = tuple(dict.fromkeys(files))
            evidence["affected_tool"] = failed_items[0].get("verifier")
            evidence["stdout_summary"] = failed_items[0].get("stdout_summary")
            evidence["stderr_summary"] = failed_items[0].get("stderr_summary")
        elif execution_error:
            evidence["summary"] = execution_error.get("message", str(execution_error))

        return evidence

    def _categorize(self, context: ReflectionContext, evidence: dict[str, Any]) -> FailureCategory:
        text = " ".join(
            filter(
                None,
                [
                    str(evidence.get("summary", "")),
                    str((evidence.get("execution_error") or {}).get("message", "")),
                    str(evidence.get("stdout_summary", "")),
                    str(evidence.get("stderr_summary", "")),
                ],
            )
        ).lower()

        failed_verifiers = evidence.get("failed_verifiers") or []
        if failed_verifiers:
            verifier = failed_verifiers[0].get("verifier", "")
            if verifier == "test":
                return FailureCategory.TEST_FAILURE
            if verifier == "file":
                return FailureCategory.MISSING_ARTIFACT
            if verifier == "output":
                return FailureCategory.INCORRECT_IMPLEMENTATION
            if verifier == "command":
                if "lint" in text or "ruff" in text:
                    return FailureCategory.LINT_FAILURE
                if "mypy" in text:
                    return FailureCategory.TYPE_ERROR
                return FailureCategory.EXECUTION_FAILURE

        if context.execution_result.outcome is ExecutionOutcome.TIMEOUT:
            return FailureCategory.TIMEOUT
        if "permission" in text or "denied" in text:
            return FailureCategory.PERMISSION_FAILURE
        if "timeout" in text or "timed out" in text:
            return FailureCategory.TIMEOUT
        if "syntax" in text or "syntaxerror" in text:
            return FailureCategory.SYNTAX_ERROR
        if "pytest" in text or "test" in text or "assert" in text:
            return FailureCategory.TEST_FAILURE
        if "type error" in text or "typeerror" in text:
            return FailureCategory.TYPE_ERROR
        if "modulenotfounderror" in text or "missing dependency" in text:
            return FailureCategory.MISSING_DEPENDENCY
        if "invalid" in text and "argument" in text:
            return FailureCategory.INVALID_TOOL_ARGUMENTS
        if context.execution_result.outcome is ExecutionOutcome.FAILURE:
            return FailureCategory.EXECUTION_FAILURE
        return FailureCategory.UNKNOWN

    def _root_cause(
        self,
        context: ReflectionContext,
        category: FailureCategory,
        evidence: dict[str, Any],
    ) -> str:
        summary = evidence.get("summary")
        if summary:
            if category is FailureCategory.TEST_FAILURE:
                return f"Tests did not pass: {summary}"
            if category is FailureCategory.MISSING_ARTIFACT:
                return f"Expected artifact or file missing: {summary}"
            if category is FailureCategory.INCORRECT_IMPLEMENTATION:
                return f"Agent output did not satisfy requirements: {summary}"
            return str(summary)

        error_message = (evidence.get("execution_error") or {}).get("message")
        if error_message:
            return str(error_message)

        return f"Execution ended with outcome {context.execution_result.outcome.value}."

    def _suggested_correction(
        self,
        category: FailureCategory,
        evidence: dict[str, Any],
    ) -> str:
        mapping = {
            FailureCategory.TEST_FAILURE: "Inspect failing tests, fix implementation, and rerun focused tests.",
            FailureCategory.MISSING_ARTIFACT: "Create or restore the missing artifact before retrying.",
            FailureCategory.INCORRECT_IMPLEMENTATION: "Adjust the implementation to satisfy required outputs.",
            FailureCategory.SYNTAX_ERROR: "Fix syntax errors in affected files before retrying.",
            FailureCategory.TYPE_ERROR: "Resolve type errors and rerun type checks.",
            FailureCategory.LINT_FAILURE: "Fix lint violations and rerun lint validation.",
            FailureCategory.PERMISSION_FAILURE: "Adjust permissions or use an authorized agent/tool configuration.",
            FailureCategory.TIMEOUT: "Reduce scope or increase timeout, then retry with a narrower target.",
            FailureCategory.MISSING_DEPENDENCY: "Install or declare the missing dependency, then retry.",
            FailureCategory.INVALID_TOOL_ARGUMENTS: "Correct tool inputs and retry the operation.",
            FailureCategory.EXECUTION_FAILURE: "Review execution error details and apply a targeted fix.",
        }
        base = mapping.get(category, "Review failure evidence and apply a targeted correction.")
        affected = evidence.get("affected_files")
        if affected:
            return f"{base} Affected files: {', '.join(affected)}."
        return base

    def _strategy_for(
        self,
        category: FailureCategory,
        context: ReflectionContext,
        evidence: dict[str, Any],
    ) -> tuple[RepairStrategy, ReflectionOutcome]:
        if context.attempt >= context.max_attempts:
            return RepairStrategy.STOP, ReflectionOutcome.MAX_RETRIES_REACHED

        non_retryable = {
            FailureCategory.PERMISSION_FAILURE,
            FailureCategory.INVALID_TOOL_ARGUMENTS,
            FailureCategory.MISSING_DEPENDENCY,
        }
        if category in non_retryable:
            return RepairStrategy.STOP, ReflectionOutcome.NON_RETRYABLE_FAILURE

        replan_categories = {FailureCategory.INCORRECT_IMPLEMENTATION}
        if category in replan_categories and context.attempt >= max(2, context.max_attempts - 1):
            return RepairStrategy.REPLAN, ReflectionOutcome.REPLAN_REQUIRED

        repeated_test_failure = (
            category is FailureCategory.TEST_FAILURE and context.attempt >= 2
        )
        if repeated_test_failure:
            return RepairStrategy.REPLAN, ReflectionOutcome.REPLAN_REQUIRED

        if category is FailureCategory.UNKNOWN and context.attempt >= 2:
            return RepairStrategy.REPLAN, ReflectionOutcome.REPLAN_REQUIRED

        return RepairStrategy.RETRY, ReflectionOutcome.RETRYABLE_FAILURE

    @staticmethod
    def _confidence(category: FailureCategory, evidence: dict[str, Any]) -> float:
        if evidence.get("failed_verifiers"):
            return 0.85
        if category is FailureCategory.UNKNOWN:
            return 0.3
        return 0.7


__all__ = ["FailureAnalyzer"]
