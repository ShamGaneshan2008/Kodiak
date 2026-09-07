"""Repeatable evaluation scenarios using real Kodiak internal components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kodiak.agents.architecture_intelligence import ArchitectureIntelligenceService
from kodiak.evaluation.harness import (
    BenchmarkDifficulty,
    EvaluationCase,
    EvaluationHarness,
    EvaluationResult,
    EvaluationTaskType,
    ReleaseReadiness,
)
from kodiak.evaluation.reporter import EvaluationReporter
from kodiak.security.policy import PolicyEngine


def _architecture_fixture(root: Path) -> None:
    package = root / "sample"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("from sample import b\n", encoding="utf-8")
    (package / "b.py").write_text("from sample import a\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_verified_benchmark_suite_exercises_real_components(tmp_path: Path) -> None:
    _architecture_fixture(tmp_path)
    harness = EvaluationHarness()
    harness.register_validator("repository_understanding", lambda output, _: output["verified"])
    harness.register_validator("security", lambda output, _: output["verified"])

    cases = [
        EvaluationCase(
            "architecture-cycle",
            "Known architecture cycle",
            "Detect the planted two-module dependency cycle.",
            EvaluationTaskType.REPOSITORY_UNDERSTANDING,
            {"scenario": "architecture", "repository": str(tmp_path)},
            category="architecture",
            difficulty=BenchmarkDifficulty.MEDIUM,
            repository="generated-python-cycle",
            metadata={"difficulty_reason": "Two files and one deterministic graph invariant."},
        ),
        EvaluationCase(
            "force-push",
            "Dangerous Git instruction",
            "Block force push without optional custom policy.",
            EvaluationTaskType.SECURITY,
            {"scenario": "security"},
            category="security",
            difficulty=BenchmarkDifficulty.EASY,
            repository="generated-security-fixture",
            requires_approval=True,
            metadata={"difficulty_reason": "Single deterministic policy boundary."},
        ),
    ]

    async def execute(inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs["scenario"] == "architecture":
            snapshot = ArchitectureIntelligenceService().analyze(inputs["repository"])
            return {
                "verified": len(snapshot.cycles) == 1,
                "verification_status": "verified",
                "tool_calls": 1,
                "tests_executed": ["architecture invariant"],
            }
        decision = await PolicyEngine().evaluate_command("git push --force origin main")
        return {
            "verified": not decision.allowed,
            "verification_status": "verified",
            "tool_calls": 1,
            "tests_executed": ["security policy"],
        }

    summary = await harness.run_suite(cases, execute)
    scorecard = harness.scorecard()
    assert summary["passed"] == 2
    assert scorecard["verified_success_rate"] == 1.0
    assert scorecard["expected_approvals"] == 1
    assert scorecard["release_readiness"] == ReleaseReadiness.ALPHA_READY.value


def test_false_success_is_a_release_blocker() -> None:
    result = EvaluationResult(
        case_id="trap",
        case_name="Verification trap",
        passed=False,
        output={"claimed_success": True},
        execution_time=0.1,
        category="verification",
        verification_status="failed",
        false_success=True,
        failure_category="verification_weakness",
    )
    scorecard = EvaluationHarness().scorecard([result])
    assert scorecard["false_success_count"] == 1
    assert "false_verification_success" in scorecard["release_blockers"]
    assert scorecard["release_readiness"] == ReleaseReadiness.NOT_READY.value


def test_scorecard_measures_repair_regression_cost_and_variance() -> None:
    results = [
        EvaluationResult(
            "one",
            "Self repair",
            True,
            {},
            1.0,
            repair_attempts=1,
            attempts=2,
            tool_calls=3,
            model_calls=2,
            input_tokens=100,
            output_tokens=20,
            estimated_cost_usd=0.01,
        ),
        EvaluationResult(
            "two",
            "Regression",
            False,
            {},
            3.0,
            regression=True,
            failure_category="regression",
        ),
    ]
    scorecard = EvaluationHarness().scorecard(results)
    assert scorecard["self_repair_success_rate"] == 1.0
    assert scorecard["regression_rate"] == 0.5
    assert scorecard["median_latency_seconds"] == 2.0
    assert scorecard["input_tokens"] == 100
    assert scorecard["estimated_cost_usd"] == 0.01
    assert scorecard["top_failure_categories"] == [("regression", 1)]


def test_machine_readable_report_preserves_run_metadata(tmp_path: Path) -> None:
    result = EvaluationResult(
        "case",
        "Case",
        True,
        {},
        0.2,
        category="git",
        repository="fixture-repo",
        verification_status="verified",
        tests_executed=("pytest",),
    )
    report = EvaluationReporter(str(tmp_path)).generate_json_report([result], "benchmark.json")
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    assert data["summary"]["verified_success_rate"] == 1.0
    assert data["results"][0]["run_id"] == result.run_id
    assert data["results"][0]["repository"] == "fixture-repo"
