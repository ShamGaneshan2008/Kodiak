"""Test agent for Kodiak's autonomous engineering workflow.

The test agent validates reviewed code patches before commit or pull-request
stages. It consumes execution plans, research reports, generated patches,
review reports, and repository context from upstream agents. It does not
perform repository intelligence, review code, modify repository files, commit
changes, or push to GitHub.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

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
from kodiak.agents.coder import CodePatch, FilePatch
from kodiak.rag.context_builder import BuiltContext

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = """\
You are a senior test engineer inside the Kodiak autonomous engineering system.
Generate missing tests for the supplied execution plan, research report, review
report, generated patch, and repository context.

Rules:
- Use only supplied context; do not request or perform repository search.
- Do not modify production code.
- Output test files as complete file contents.
- Prefer existing repository test style and imports shown in context.
- Output ONLY valid JSON; no prose and no markdown fences.

Output schema:
{
  "tests": [
    {
      "path": "tests/unit/test_feature.py",
      "scope": "unit | integration | regression",
      "content": "<complete test file content>",
      "rationale": "<why this test is needed>"
    }
  ],
  "summary": "<brief test generation summary>"
}
"""


class TestRunnerName(StrEnum):
    """Supported Python test runners."""

    PYTEST = "pytest"
    UNITTEST = "unittest"


class TestScope(StrEnum):
    """Kinds of tests the agent can discover, generate, and execute."""

    UNIT = "unit"
    INTEGRATION = "integration"
    REGRESSION = "regression"
    PROJECT = "project"


class TestStatus(StrEnum):
    """Execution status for test commands and reports."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"
    ERROR = "error"


class AsyncCommandRunner(Protocol):
    """Protocol for injected sandbox or process execution services."""

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        environment: Mapping[str, str] | None = None,
    ) -> Any:
        """Run a command and return an execution-like result."""


@dataclass(frozen=True)
class DiscoveredTest:
    """Existing test file discovered in the repository.

    Attributes:
        path: Repository-relative test path.
        scope: Inferred test scope.
        runner: Preferred runner for the test.
    """

    path: str
    scope: TestScope
    runner: TestRunnerName

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the discovered test."""
        return {
            "path": self.path,
            "scope": self.scope.value,
            "runner": self.runner.value,
        }


@dataclass(frozen=True)
class GeneratedTest:
    """Generated test file content that callers may persist elsewhere.

    Attributes:
        path: Repository-relative suggested test path.
        scope: Test scope.
        content: Complete test file content.
        rationale: Why the test is needed.
    """

    path: str
    scope: TestScope
    content: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the generated test."""
        return {
            "path": self.path,
            "scope": self.scope.value,
            "content": self.content,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class TestCommand:
    """A test command selected for execution.

    Attributes:
        command: Command argv.
        scope: Test scope covered by the command.
        runner: Test runner.
        reason: Why this command was selected.
    """

    command: tuple[str, ...]
    scope: TestScope
    runner: TestRunnerName
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the command."""
        return {
            "command": list(self.command),
            "scope": self.scope.value,
            "runner": self.runner.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TestRunResult:
    """Result from one test command.

    Attributes:
        command: Executed or planned command.
        status: Command status.
        exit_code: Process exit code, when executed.
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_ms: Command duration in milliseconds.
    """

    command: TestCommand
    status: TestStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the run result."""
        return {
            "command": self.command.to_dict(),
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(frozen=True)
class CoverageReport:
    """Structured coverage summary.

    Attributes:
        enabled: Whether coverage collection was requested.
        total_percent: Parsed total coverage percentage.
        raw_output: Relevant coverage output.
    """

    enabled: bool
    total_percent: float | None = None
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of coverage data."""
        return {
            "enabled": self.enabled,
            "total_percent": self.total_percent,
            "raw_output": self.raw_output,
        }


@dataclass(frozen=True)
class TestFailure:
    """Parsed test failure or error.

    Attributes:
        test_name: Failed test node or best-effort name.
        file: File associated with the failure, when known.
        line: Line associated with the failure, when known.
        message: Failure message.
        traceback: Relevant stack trace excerpt.
        suggestion: Suggested next action.
    """

    test_name: str
    file: str | None
    line: int | None
    message: str
    traceback: str = ""
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the failure."""
        return {
            "test_name": self.test_name,
            "file": self.file,
            "line": self.line,
            "message": self.message,
            "traceback": self.traceback,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class TestReport:
    """Final structured testing report.

    Attributes:
        status: Overall status.
        runner: Test runner used.
        dry_run: Whether commands were planned but not executed.
        discovered_tests: Existing tests discovered in the repository.
        generated_tests: Missing tests generated by the agent.
        commands: Planned test commands.
        results: Executed or dry-run command results.
        failures: Parsed test failures.
        coverage: Coverage summary.
        summary: Human-readable summary.
        metadata: Additional structured metadata.
    """

    status: TestStatus
    runner: TestRunnerName
    dry_run: bool
    discovered_tests: tuple[DiscoveredTest, ...] = ()
    generated_tests: tuple[GeneratedTest, ...] = ()
    commands: tuple[TestCommand, ...] = ()
    results: tuple[TestRunResult, ...] = ()
    failures: tuple[TestFailure, ...] = ()
    coverage: CoverageReport = field(default_factory=lambda: CoverageReport(False))
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the test report."""
        return {
            "status": self.status.value,
            "runner": self.runner.value,
            "dry_run": self.dry_run,
            "discovered_tests": [test.to_dict() for test in self.discovered_tests],
            "generated_tests": [test.to_dict() for test in self.generated_tests],
            "commands": [command.to_dict() for command in self.commands],
            "results": [result.to_dict() for result in self.results],
            "failures": [failure.to_dict() for failure in self.failures],
            "coverage": self.coverage.to_dict(),
            "summary": self.summary,
            "metadata": self.metadata,
        }


class TestAgent(BaseAgent):
    """Validate generated code through discovered, generated, and executed tests."""

    role = AgentRole.TESTER

    def __init__(
        self,
        llm_client: Any | None = None,
        *,
        command_runner: AsyncCommandRunner | None = None,
        command_container: Any | None = None,
        repository_root: str | Path | None = None,
        default_runner: TestRunnerName | str = TestRunnerName.PYTEST,
        default_model_preference: str = "default",
        max_output_tokens: int = 6000,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Initialize the test agent.

        Args:
            llm_client: Optional injected LLM facade for test generation.
            command_runner: Injected execution service used to run test commands.
            command_container: Optional sandbox container for runners exposing
                ``execute_shell(container, command, timeout=...)``.
            repository_root: Repository root for discovery and command execution.
            default_runner: Default test runner, either pytest or unittest.
            default_model_preference: LLM routing preference for test generation.
            max_output_tokens: Maximum tokens requested for generated tests.
            timeout_seconds: Default timeout for each test command.
        """
        super().__init__()
        self._llm = llm_client
        self._command_runner = command_runner
        self._command_container = command_container
        self._repository_root = Path(repository_root).resolve() if repository_root else None
        self._default_runner = self._coerce_runner(default_runner)
        self._default_model_preference = default_model_preference
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds

    async def _run(self, input_: AgentInput) -> AgentOutput:
        """Run the test agent through BaseAgent orchestration."""
        report = await self.run_tests(
            instruction=input_.instruction,
            execution_plan=self._first_present(
                input_.context,
                "execution_plan",
                "plan",
                "task_plan",
            ),
            research_report=input_.context.get("research_report"),
            generated_patch=self._first_present(
                input_.context,
                "generated_patch",
                "code_patch",
                "patch",
            ),
            review_report=input_.context.get("review_report"),
            repository_context=self._first_present(
                input_.context,
                "repository_context",
                "built_context",
                "context",
                "rag_context",
            ),
            repository_root=input_.context.get("work_dir"),
            runner=input_.context.get("runner", self._default_runner),
            dry_run=bool(input_.context.get("dry_run", True)),
            affected_only=bool(input_.context.get("affected_only", True)),
            collect_coverage=bool(input_.context.get("collect_coverage", False)),
        )
        return self._make_output(
            input_,
            result=report.to_dict(),
            metadata={"dry_run": report.dry_run, "runner": report.runner.value},
        )

    async def run_tests(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        generated_patch: Any,
        research_report: Any | None = None,
        review_report: Any | None = None,
        repository_context: Any | None = None,
        repository_root: str | Path | None = None,
        runner: TestRunnerName | str | None = None,
        dry_run: bool = True,
        affected_only: bool = True,
        collect_coverage: bool = False,
    ) -> TestReport:
        """Discover, generate, execute, and report tests for a code patch.

        Args:
            instruction: Original implementation instruction.
            execution_plan: PlannerAgent output or normalized execution plan.
            generated_patch: CodePatch, serialized patch, or changed files.
            research_report: ResearchAgent report or serialized report.
            review_report: ReviewAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            repository_root: Repository root for discovery and execution.
            runner: Test runner to use.
            dry_run: Plan commands without executing them.
            affected_only: Prefer tests related to changed files.
            collect_coverage: Include coverage flags and parse coverage output.

        Returns:
            Structured testing report.
        """
        root = self._resolve_repository_root(repository_root)
        selected_runner = self._coerce_runner(runner or self._default_runner)
        patch_files = self._normalize_patch_files(generated_patch)
        discovered = self.discover_tests(
            repository_root=root,
            generated_patch=patch_files,
            runner=selected_runner,
            affected_only=affected_only,
        )
        generated = await self.generate_tests(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            generated_patch=patch_files,
            review_report=review_report,
            repository_context=repository_context,
            runner=selected_runner,
        )
        commands = self._select_commands(
            discovered,
            patch_files=patch_files,
            runner=selected_runner,
            affected_only=affected_only,
            collect_coverage=collect_coverage,
        )
        results = await self._execute_commands(
            commands,
            repository_root=root,
            dry_run=dry_run,
        )
        failures = self.analyze_failures(results)
        coverage = self.collect_coverage(results, enabled=collect_coverage)
        return self.generate_test_report(
            runner=selected_runner,
            dry_run=dry_run,
            discovered_tests=discovered,
            generated_tests=generated,
            commands=commands,
            results=results,
            failures=failures,
            coverage=coverage,
        )

    async def generate_tests(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        generated_patch: Any,
        research_report: Any | None = None,
        review_report: Any | None = None,
        repository_context: Any | None = None,
        runner: TestRunnerName | str | None = None,
        model_preference: str | None = None,
    ) -> tuple[GeneratedTest, ...]:
        """Generate missing tests without writing files.

        Args:
            instruction: Original implementation instruction.
            execution_plan: PlannerAgent output or normalized execution plan.
            generated_patch: CodePatch, serialized patch, or normalized files.
            research_report: ResearchAgent report or serialized report.
            review_report: ReviewAgent report or serialized report.
            repository_context: BuiltContext, serialized context, or prompt text.
            runner: Test runner the generated tests should target.
            model_preference: Optional LLM routing preference.

        Returns:
            Generated test files as structured data. Empty when no LLM is
            configured.
        """
        if self._llm is None:
            return ()
        patch_files = self._normalize_patch_files(generated_patch)
        message = self._build_generation_message(
            instruction=instruction,
            execution_plan=execution_plan,
            research_report=research_report,
            review_report=review_report,
            repository_context=repository_context,
            patch_files=patch_files,
            runner=self._coerce_runner(runner or self._default_runner),
        )
        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
            model_preference=model_preference or self._default_model_preference,
            max_tokens=self._max_output_tokens,
            temperature=0.1,
        )
        return self._parse_generated_tests(str(response.get("content", "")))

    def discover_tests(
        self,
        *,
        repository_root: str | Path | None = None,
        generated_patch: Any | None = None,
        runner: TestRunnerName | str | None = None,
        affected_only: bool = True,
    ) -> tuple[DiscoveredTest, ...]:
        """Discover existing tests without repository analysis or execution.

        Args:
            repository_root: Repository root to scan.
            generated_patch: Patch used to select affected tests.
            runner: Preferred runner.
            affected_only: Return only likely affected tests when possible.

        Returns:
            Discovered test files.
        """
        root = self._resolve_repository_root(repository_root)
        selected_runner = self._coerce_runner(runner or self._default_runner)
        if root is None or not root.exists():
            return ()

        candidates = [path for path in root.rglob("test_*.py") if self._is_test_path(path, root)]
        candidates.extend(
            path for path in root.rglob("*_test.py") if self._is_test_path(path, root)
        )
        tests = tuple(
            DiscoveredTest(
                path=self._relative(root, path),
                scope=self._infer_scope(path),
                runner=selected_runner,
            )
            for path in sorted(set(candidates))
        )
        if not affected_only:
            return tests

        affected = self._affected_test_paths(
            changed_files=self._changed_paths(generated_patch),
            discovered=tests,
        )
        return tuple(test for test in tests if test.path in affected) or tests

    async def run_unit_tests(
        self,
        *,
        repository_root: str | Path | None = None,
        runner: TestRunnerName | str | None = None,
        dry_run: bool = True,
    ) -> tuple[TestRunResult, ...]:
        """Run or plan unit test commands.

        Args:
            repository_root: Repository root for execution.
            runner: Test runner to use.
            dry_run: Plan command without executing.

        Returns:
            Test command results.
        """
        selected_runner = self._coerce_runner(runner or self._default_runner)
        command = self._scope_command(selected_runner, TestScope.UNIT)
        return await self._execute_commands(
            (command,),
            repository_root=self._resolve_repository_root(repository_root),
            dry_run=dry_run,
        )

    async def run_integration_tests(
        self,
        *,
        repository_root: str | Path | None = None,
        runner: TestRunnerName | str | None = None,
        dry_run: bool = True,
    ) -> tuple[TestRunResult, ...]:
        """Run or plan integration test commands.

        Args:
            repository_root: Repository root for execution.
            runner: Test runner to use.
            dry_run: Plan command without executing.

        Returns:
            Test command results.
        """
        selected_runner = self._coerce_runner(runner or self._default_runner)
        command = self._scope_command(selected_runner, TestScope.INTEGRATION)
        return await self._execute_commands(
            (command,),
            repository_root=self._resolve_repository_root(repository_root),
            dry_run=dry_run,
        )

    def collect_coverage(
        self,
        results: Sequence[TestRunResult],
        *,
        enabled: bool = True,
    ) -> CoverageReport:
        """Parse coverage output from executed test results.

        Args:
            results: Test execution results.
            enabled: Whether coverage was requested.

        Returns:
            Structured coverage report.
        """
        if not enabled:
            return CoverageReport(enabled=False)
        combined = "\n".join(f"{result.stdout}\n{result.stderr}" for result in results)
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%", combined)
        total = float(match.group(1)) if match else None
        return CoverageReport(enabled=True, total_percent=total, raw_output=combined[-4000:])

    def analyze_failures(
        self,
        results: Sequence[TestRunResult],
    ) -> tuple[TestFailure, ...]:
        """Analyze failed test output and stack traces.

        Args:
            results: Test execution results.

        Returns:
            Parsed failure reports.
        """
        failures: list[TestFailure] = []
        for result in results:
            if result.status not in {TestStatus.FAILED, TestStatus.ERROR}:
                continue
            output = f"{result.stdout}\n{result.stderr}"
            failures.extend(self._parse_pytest_failures(output))
            if not failures:
                failures.append(
                    TestFailure(
                        test_name="test command",
                        file=None,
                        line=None,
                        message=output.strip()[-1000:] or "Test command failed.",
                        traceback=self._traceback_excerpt(output),
                        suggestion="Inspect the failing command output and patch.",
                    )
                )
        return tuple(failures)

    def generate_test_report(
        self,
        *,
        runner: TestRunnerName | str,
        dry_run: bool,
        discovered_tests: Sequence[DiscoveredTest],
        generated_tests: Sequence[GeneratedTest],
        commands: Sequence[TestCommand],
        results: Sequence[TestRunResult],
        failures: Sequence[TestFailure],
        coverage: CoverageReport | None = None,
    ) -> TestReport:
        """Build the final structured test report.

        Args:
            runner: Test runner used.
            dry_run: Whether execution was skipped.
            discovered_tests: Existing tests found in the repository.
            generated_tests: Missing tests generated by the agent.
            commands: Planned test commands.
            results: Test command results.
            failures: Parsed failures.
            coverage: Optional coverage report.

        Returns:
            Structured test report.
        """
        selected_runner = self._coerce_runner(runner)
        status = self._overall_status(results, dry_run=dry_run)
        summary = self._summary(status, discovered_tests, generated_tests, results, failures)
        return TestReport(
            status=status,
            runner=selected_runner,
            dry_run=dry_run,
            discovered_tests=tuple(discovered_tests),
            generated_tests=tuple(generated_tests),
            commands=tuple(commands),
            results=tuple(results),
            failures=tuple(failures),
            coverage=coverage or CoverageReport(False),
            summary=summary,
            metadata={
                "discovered_count": len(discovered_tests),
                "generated_count": len(generated_tests),
                "command_count": len(commands),
                "failure_count": len(failures),
            },
        )

    def _build_generation_message(
        self,
        *,
        instruction: str,
        execution_plan: Any,
        research_report: Any | None,
        review_report: Any | None,
        repository_context: Any | None,
        patch_files: Sequence[Any],
        runner: TestRunnerName,
    ) -> str:
        parts = [
            f"## Original task\n{instruction}",
            f"## Test runner\n{runner.value}",
            f"## Execution plan\n{self._to_pretty_json(execution_plan)}",
        ]
        if research_report:
            parts.append(f"## Research report\n{self._to_pretty_json(research_report)}")
        if review_report:
            parts.append(f"## Review report\n{self._to_pretty_json(review_report)}")
        context_text = self._context_to_text(repository_context)
        if context_text:
            parts.append(f"## Repository context\n{context_text}")
        parts.append(
            "## Generated patch files\n"
            f"{self._to_pretty_json([self._file_to_dict(file) for file in patch_files])}"
        )
        return "\n\n".join(parts)

    def _select_commands(
        self,
        discovered: Sequence[DiscoveredTest],
        *,
        patch_files: Sequence[Any],
        runner: TestRunnerName,
        affected_only: bool,
        collect_coverage: bool,
    ) -> tuple[TestCommand, ...]:
        if discovered and affected_only:
            return tuple(
                self._file_command(test, collect_coverage=collect_coverage) for test in discovered
            )
        if discovered and not affected_only:
            scopes = sorted({test.scope for test in discovered}, key=lambda scope: scope.value)
            return tuple(
                self._scope_command(runner, scope, collect_coverage=collect_coverage)
                for scope in scopes
            )
        changed = self._changed_paths(patch_files)
        reason = "No existing tests were discovered; run the project test suite."
        if changed:
            reason = "No affected tests were discovered for changed files."
        return (
            TestCommand(
                command=self._project_command(runner, collect_coverage=collect_coverage),
                scope=TestScope.PROJECT,
                runner=runner,
                reason=reason,
            ),
        )

    async def _execute_commands(
        self,
        commands: Sequence[TestCommand],
        *,
        repository_root: Path | None,
        dry_run: bool,
    ) -> tuple[TestRunResult, ...]:
        if dry_run or self._command_runner is None or repository_root is None:
            return tuple(
                TestRunResult(command=command, status=TestStatus.DRY_RUN) for command in commands
            )

        results: list[TestRunResult] = []
        for command in commands:
            try:
                raw = await self._run_command(command, repository_root)
                exit_code = int(self._result_attr(raw, "exit_code", 0))
                status = TestStatus.PASSED if exit_code == 0 else TestStatus.FAILED
                results.append(
                    TestRunResult(
                        command=command,
                        status=status,
                        exit_code=exit_code,
                        stdout=str(self._result_attr(raw, "stdout", "")),
                        stderr=str(self._result_attr(raw, "stderr", "")),
                        duration_ms=float(self._result_attr(raw, "duration_ms", 0.0)),
                    )
                )
            except Exception as exc:
                logger.warning("test_agent.command_failed", error=str(exc))
                results.append(
                    TestRunResult(
                        command=command,
                        status=TestStatus.ERROR,
                        exit_code=-1,
                        stderr=str(exc),
                    )
                )
        return tuple(results)

    async def _run_command(self, command: TestCommand, repository_root: Path) -> Any:
        if self._command_runner is None:
            raise RuntimeError("command_runner is required to execute tests")
        if hasattr(self._command_runner, "run"):
            return await self._command_runner.run(
                command.command,
                cwd=repository_root,
                timeout_seconds=self._timeout_seconds,
                environment=None,
            )
        if hasattr(self._command_runner, "execute_shell") and self._command_container is not None:
            shell_command = (
                f"cd {shlex.quote(str(repository_root))} && {self._shell_join(command.command)}"
            )
            return await self._command_runner.execute_shell(
                self._command_container,
                shell_command,
                timeout=self._timeout_seconds,
            )
        raise RuntimeError("command_runner must expose run(...) or execute_shell(...)")

    def _file_command(
        self,
        test: DiscoveredTest,
        *,
        collect_coverage: bool,
    ) -> TestCommand:
        if test.runner is TestRunnerName.UNITTEST:
            command = ("python", "-m", "unittest", test.path.replace("/", ".").removesuffix(".py"))
        else:
            command = ("python", "-m", "pytest", test.path, "-q")
            if collect_coverage:
                command = (*command, "--cov=kodiak", "--cov-report=term")
        return TestCommand(
            command=command,
            scope=test.scope,
            runner=test.runner,
            reason="Selected as an affected existing test.",
        )

    def _scope_command(
        self,
        runner: TestRunnerName,
        scope: TestScope,
        *,
        collect_coverage: bool = False,
    ) -> TestCommand:
        target = {
            TestScope.UNIT: "tests/unit",
            TestScope.INTEGRATION: "tests/integration",
            TestScope.REGRESSION: "tests/regression",
        }.get(scope, "tests")
        if runner is TestRunnerName.UNITTEST:
            command = ("python", "-m", "unittest", "discover", "-s", target)
        else:
            command = ("python", "-m", "pytest", target, "-q")
            if collect_coverage:
                command = (*command, "--cov=kodiak", "--cov-report=term")
        return TestCommand(
            command=command,
            scope=scope,
            runner=runner,
            reason=f"Selected {scope.value} test suite.",
        )

    def _project_command(
        self,
        runner: TestRunnerName,
        *,
        collect_coverage: bool,
    ) -> tuple[str, ...]:
        if runner is TestRunnerName.UNITTEST:
            return ("python", "-m", "unittest", "discover")
        command = ("python", "-m", "pytest", "-q")
        if collect_coverage:
            command = (*command, "--cov=kodiak", "--cov-report=term")
        return command

    def _parse_generated_tests(self, raw: str) -> tuple[GeneratedTest, ...]:
        try:
            data = json.loads(self._strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            logger.warning("test_agent.generated_tests_parse_failed", error=str(exc))
            return ()
        if not isinstance(data, Mapping):
            return ()
        tests = data.get("tests", [])
        if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)):
            return ()
        generated: list[GeneratedTest] = []
        for item in tests:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            if not path or not content:
                continue
            generated.append(
                GeneratedTest(
                    path=path,
                    scope=self._coerce_scope(item.get("scope", TestScope.UNIT.value)),
                    content=content,
                    rationale=str(item.get("rationale", "")),
                )
            )
        return tuple(generated)

    def _normalize_patch_files(self, patch: Any) -> tuple[Any, ...]:
        if patch is None:
            return ()
        if isinstance(patch, CodePatch):
            return tuple(patch.files)
        if isinstance(patch, FilePatch):
            return (patch,)
        if isinstance(patch, Mapping):
            files = patch.get("files") or patch.get("code_changes")
            if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
                return tuple(files)
            if "path" in patch:
                return (patch,)
        if isinstance(patch, Sequence) and not isinstance(patch, (str, bytes)):
            return tuple(patch)
        return ()

    def _file_to_dict(self, file: Any) -> dict[str, Any]:
        if hasattr(file, "to_dict"):
            return file.to_dict()
        if isinstance(file, Mapping):
            return dict(file)
        return {"value": str(file)}

    def _changed_paths(self, patch: Any) -> set[str]:
        paths: set[str] = set()
        for file in self._normalize_patch_files(patch):
            if isinstance(file, FilePatch):
                paths.add(file.path)
            elif isinstance(file, Mapping) and file.get("path"):
                paths.add(str(file["path"]))
            elif hasattr(file, "path"):
                paths.add(str(file.path))
        return paths

    def _affected_test_paths(
        self,
        *,
        changed_files: set[str],
        discovered: Sequence[DiscoveredTest],
    ) -> set[str]:
        affected: set[str] = set()
        changed_stems = {Path(path).stem.removeprefix("__init__") for path in changed_files}
        for test in discovered:
            test_name = Path(test.path).stem
            if any(stem and stem in test_name for stem in changed_stems):
                affected.add(test.path)
        for path in changed_files:
            if path.startswith("tests/") and path.endswith(".py"):
                affected.add(path)
        return affected

    def _parse_pytest_failures(self, output: str) -> list[TestFailure]:
        failures: list[TestFailure] = []
        failed_lines = re.findall(r"FAILED\s+([^\s]+)\s+-\s+(.+)", output)
        for node_id, message in failed_lines:
            file, line = self._split_node_id(node_id)
            failures.append(
                TestFailure(
                    test_name=node_id,
                    file=file,
                    line=line,
                    message=message.strip(),
                    traceback=self._traceback_excerpt(output),
                    suggestion="Use the failure message and review report to update the patch.",
                )
            )
        error_matches = re.findall(r"ERROR\s+([^\s]+)\s+-\s+(.+)", output)
        for node_id, message in error_matches:
            file, line = self._split_node_id(node_id)
            failures.append(
                TestFailure(
                    test_name=node_id,
                    file=file,
                    line=line,
                    message=message.strip(),
                    traceback=self._traceback_excerpt(output),
                    suggestion="Fix setup/import errors before rerunning affected tests.",
                )
            )
        return failures

    def _split_node_id(self, node_id: str) -> tuple[str | None, int | None]:
        file_part = node_id.split("::", 1)[0]
        match = re.search(r":(\d+)$", file_part)
        line = int(match.group(1)) if match else None
        file_path = file_part.rsplit(":", 1)[0] if match else file_part
        return file_path or None, line

    def _traceback_excerpt(self, output: str) -> str:
        lines = output.splitlines()
        interesting = [
            line
            for line in lines
            if line.startswith("E   ") or "Traceback" in line or re.match(r"\s*File \"", line)
        ]
        return "\n".join(interesting[-40:])

    def _overall_status(
        self,
        results: Sequence[TestRunResult],
        *,
        dry_run: bool,
    ) -> TestStatus:
        if dry_run:
            return TestStatus.DRY_RUN
        if not results:
            return TestStatus.SKIPPED
        if any(result.status is TestStatus.ERROR for result in results):
            return TestStatus.ERROR
        if any(result.status is TestStatus.FAILED for result in results):
            return TestStatus.FAILED
        return TestStatus.PASSED

    def _summary(
        self,
        status: TestStatus,
        discovered: Sequence[DiscoveredTest],
        generated: Sequence[GeneratedTest],
        results: Sequence[TestRunResult],
        failures: Sequence[TestFailure],
    ) -> str:
        if status is TestStatus.DRY_RUN:
            return (
                f"Planned {len(results)} test command(s), discovered "
                f"{len(discovered)} existing test file(s), and generated "
                f"{len(generated)} missing test file suggestion(s)."
            )
        if status is TestStatus.PASSED:
            return f"Executed {len(results)} test command(s); all completed successfully."
        return f"Executed {len(results)} test command(s); found {len(failures)} failure(s)."

    def _is_test_path(self, path: Path, root: Path) -> bool:
        relative = self._relative(root, path)
        ignored = {".venv", "venv", "env", ".tox", "build", "dist", ".git"}
        if any(part in ignored for part in Path(relative).parts):
            return False
        return "tests" in Path(relative).parts or Path(relative).name.startswith("test_")

    def _infer_scope(self, path: Path) -> TestScope:
        parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        if "integration" in parts or "integration" in name:
            return TestScope.INTEGRATION
        if "regression" in parts or "regression" in name:
            return TestScope.REGRESSION
        return TestScope.UNIT

    def _relative(self, root: Path, path: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    def _resolve_repository_root(self, override: str | Path | None = None) -> Path | None:
        if override:
            return Path(override).resolve()
        return self._repository_root

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

    def _result_attr(self, result: Any, key: str, default: Any) -> Any:
        if isinstance(result, Mapping):
            return result.get(key, default)
        return getattr(result, key, default)

    def _coerce_runner(self, runner: TestRunnerName | str) -> TestRunnerName:
        if isinstance(runner, TestRunnerName):
            return runner
        try:
            return TestRunnerName(str(runner).lower())
        except ValueError:
            return TestRunnerName.PYTEST

    def _coerce_scope(self, scope: TestScope | str) -> TestScope:
        if isinstance(scope, TestScope):
            return scope
        try:
            return TestScope(str(scope).lower())
        except ValueError:
            return TestScope.UNIT

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

    def _shell_join(self, command: Sequence[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)


TesterAgent = TestAgent
