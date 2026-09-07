"""Regression coverage for the public-alpha feedback and failure-mining loop."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from kodiak.cli.app import app
from kodiak.evaluation.alpha_suite import alpha_scenarios
from kodiak.evaluation.bug_bash import (
    AlphaBugBash,
    AlphaScenario,
    FailureCategory,
    FailureSeverity,
    failure_fingerprint,
)


def _failure(message: str):
    def execute() -> None:
        raise RuntimeError(message)

    return execute


def test_duplicate_failures_cluster_with_stable_sanitized_fingerprint() -> None:
    runner = AlphaBugBash()
    scenario = AlphaScenario(
        id="cli-startup",
        description="Minimal install imports the CLI without optional RAG",
        category=FailureCategory.INSTALLATION,
        severity=FailureSeverity.HIGH,
        execute=_failure("Missing module at C:\\private\\repo\\x.py line 123"),
        regression_test="test_minimal_install_cli_does_not_import_optional_rag",
    )
    first = runner.run([scenario])[0]
    second = runner.run([scenario])[0]
    assert first.fingerprint == second.fingerprint
    assert second.occurrence_count == 2
    assert "private" not in second.fingerprint


def test_fingerprint_ignores_volatile_paths_and_numbers() -> None:
    first = failure_fingerprint(FailureCategory.CLI, "ValueError", "doctor", "C:\\a\\x.py:12")
    second = failure_fingerprint(FailureCategory.CLI, "ValueError", "doctor", "D:\\b\\y.py:99")
    assert first == second


def test_critical_security_failure_blocks_alpha_health() -> None:
    result = AlphaBugBash().run(
        [
            AlphaScenario(
                id="approval-bypass",
                description="Policy denial cannot be bypassed",
                category=FailureCategory.SECURITY,
                severity=FailureSeverity.CRITICAL,
                execute=_failure("approval bypass reproduced"),
            )
        ]
    )
    report = AlphaBugBash.health_report(result)
    assert report["alpha_ready"] is False
    assert report["critical"] == 1
    assert report["release_blockers"]


def test_runner_rejects_unisolated_repository_mutation() -> None:
    scenario = AlphaScenario(
        id="dirty-tree",
        description="Mutation scenarios need an isolated repository",
        category=FailureCategory.GIT,
        severity=FailureSeverity.HIGH,
        execute=lambda: None,
        mutates_repository=True,
    )
    with pytest.raises(ValueError, match="explicit isolated repository"):
        AlphaBugBash().run([scenario])


def test_doctor_json_is_machine_readable_and_never_contains_secret(monkeypatch) -> None:
    secret = "sk-" + "A" * 48
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["provider"]["configured"] is True
    assert secret not in result.stdout
    assert "OPENAI_API_KEY" in result.stdout


def test_minimal_install_cli_does_not_import_optional_rag() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_curated_alpha_suite_is_safe_and_passes() -> None:
    scenarios = alpha_scenarios()
    assert all(not scenario.mutates_repository for scenario in scenarios)
    results = AlphaBugBash().run(scenarios)
    report = AlphaBugBash.health_report(results)
    assert report["failed"] == 0
    assert report["alpha_ready"] is True
    assert {result.category for result in results} >= {
        FailureCategory.INSTALLATION,
        FailureCategory.CONFIGURATION,
        FailureCategory.SECURITY,
    }
