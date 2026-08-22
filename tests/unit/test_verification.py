"""Focused tests for the verification system."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from kodiak.db.models.task import Task, TaskPriority, TaskSource, TaskStatus
from kodiak.orchestration.execution.models import ExecutionOutcome, ExecutionResult
from kodiak.orchestration.verification.engine import VerificationEngine
from kodiak.orchestration.verification.models import (
    VerificationContext,
    VerificationStatus,
    aggregate_evidence,
)
from kodiak.orchestration.verification.verifiers import (
    CommandVerifier,
    FileVerifier,
    OutputVerifier,
    TestVerifier,
)
from kodiak.tools.builtin import CommandExecutionTool, TestRunnerTool
from kodiak.tools.registry import ToolRegistry
from kodiak.tools.router import ToolRouter


def _task(**context: object) -> Task:
    return Task(
        id=uuid4(),
        repository_id=str(uuid4()),
        title="verify me",
        description="verification test",
        source=TaskSource.API,
        status=TaskStatus.COMPLETED,
        priority=TaskPriority.MEDIUM,
        max_retries=0,
        context=dict(context),
    )


def _execution_result(result: dict | None = None) -> ExecutionResult:
    return ExecutionResult(
        task_id="task-1",
        outcome=ExecutionOutcome.SUCCESS,
        attempts=1,
        duration_seconds=0.1,
        result=result or {},
        final_status=TaskStatus.COMPLETED,
    )


@pytest.mark.asyncio
async def test_output_verifier_success() -> None:
    task = _task(verification={"required_output_fields": ["analysis", "status"]})
    context = VerificationContext.from_execution(
        task,
        _execution_result({"analysis": {}, "status": "ok"}),
    )
    evidence = await OutputVerifier().verify(context)
    assert evidence.status is VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_output_verifier_missing_field() -> None:
    task = _task(verification={"required_output_fields": ["analysis"]})
    context = VerificationContext.from_execution(task, _execution_result({"status": "ok"}))
    evidence = await OutputVerifier().verify(context)
    assert evidence.status is VerificationStatus.FAILED
    assert "analysis" in (evidence.message or "")


@pytest.mark.asyncio
async def test_file_verifier_missing_artifact(tmp_path: Path) -> None:
    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "required_artifacts": ["output.txt"],
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({"ok": True}))
    evidence = await FileVerifier().verify(context)
    assert evidence.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_file_verifier_success(tmp_path: Path) -> None:
    artifact = tmp_path / "output.txt"
    artifact.write_text("done", encoding="utf-8")
    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "required_artifacts": ["output.txt"],
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({"ok": True}))
    evidence = await FileVerifier().verify(context)
    assert evidence.status is VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_file_verifier_unexpected_file(tmp_path: Path) -> None:
    (tmp_path / "secrets.env").write_text("x", encoding="utf-8")
    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "expected_files": [],
            "unexpected_files": ["secrets.env"],
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({}))
    evidence = await FileVerifier().verify(context)
    assert evidence.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_test_verifier_success(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text(
        "def test_passes():\n    assert 1 == 1\n",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register_tool(TestRunnerTool(workspace_root=tmp_path))
    router = ToolRouter(registry=registry)

    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "run_tests": {"test_target": "tests/test_ok.py"},
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({"ok": True}))
    evidence = await TestVerifier(tool_router=router).verify(context)
    assert evidence.status is VerificationStatus.VERIFIED
    assert evidence.exit_code == 0


@pytest.mark.asyncio
async def test_test_verifier_failure(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_bad.py").write_text(
        "def test_fails():\n    assert 1 == 2\n",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register_tool(TestRunnerTool(workspace_root=tmp_path))
    router = ToolRouter(registry=registry)

    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "run_tests": {"test_target": "tests/test_bad.py"},
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({"ok": True}))
    evidence = await TestVerifier(tool_router=router).verify(context)
    assert evidence.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_command_verifier_via_tool_router(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_tool(CommandExecutionTool(workspace_root=tmp_path))
    router = ToolRouter(registry=registry)

    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "commands": [{"command": "python", "args": ["-c", "print('verified')"]}],
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({}))
    evidence = await CommandVerifier(tool_router=router).verify(context)
    assert evidence.status is VerificationStatus.VERIFIED


@pytest.mark.asyncio
async def test_command_verifier_failure(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_tool(CommandExecutionTool(workspace_root=tmp_path))
    router = ToolRouter(registry=registry)

    task = _task(
        verification={
            "commands": [{"command": "python", "args": ["-c", "raise SystemExit(3)"]}],
        }
    )
    context = VerificationContext.from_execution(task, _execution_result({}))
    evidence = await CommandVerifier(tool_router=router).verify(context)
    assert evidence.status is VerificationStatus.FAILED


def test_aggregate_verified() -> None:
    from kodiak.orchestration.verification.models import VerificationEvidence

    result = aggregate_evidence(
        [
            VerificationEvidence(verifier="a", status=VerificationStatus.VERIFIED),
            VerificationEvidence(verifier="b", status=VerificationStatus.VERIFIED),
        ]
    )
    assert result.status is VerificationStatus.VERIFIED


def test_aggregate_failed() -> None:
    from kodiak.orchestration.verification.models import VerificationEvidence

    result = aggregate_evidence(
        [
            VerificationEvidence(verifier="a", status=VerificationStatus.VERIFIED),
            VerificationEvidence(verifier="b", status=VerificationStatus.FAILED, message="bad"),
        ]
    )
    assert result.status is VerificationStatus.FAILED


def test_aggregate_inconclusive() -> None:
    from kodiak.orchestration.verification.models import VerificationEvidence

    result = aggregate_evidence(
        [
            VerificationEvidence(verifier="a", status=VerificationStatus.VERIFIED),
            VerificationEvidence(verifier="b", status=VerificationStatus.INCONCLUSIVE),
        ]
    )
    assert result.status is VerificationStatus.INCONCLUSIVE


@pytest.mark.asyncio
async def test_verification_engine_multiple_verifiers(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    task = _task(
        verification={
            "workspace_root": str(tmp_path),
            "required_output_fields": ["payload"],
            "required_artifacts": ["result.json"],
        }
    )
    engine = VerificationEngine()
    result = await engine.verify(task, _execution_result({"payload": {"ok": True}}))
    assert result.status is VerificationStatus.VERIFIED
    assert len(result.evidence) == 2


@pytest.mark.asyncio
async def test_verification_engine_inconclusive_without_criteria() -> None:
    task = _task(verification={"enabled": True})
    engine = VerificationEngine()
    result = await engine.verify(task, _execution_result({"ok": True}))
    assert result.status is VerificationStatus.INCONCLUSIVE
