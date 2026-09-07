"""Runnable, deterministic public-alpha bug-bash suite.

Run with ``python -m kodiak.evaluation.alpha_suite``. The command emits one
machine-readable JSON health report and does not mutate the target repository.
"""

from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from kodiak.cli.app import app
from kodiak.evaluation.bug_bash import (
    AlphaBugBash,
    AlphaScenario,
    FailureCategory,
    FailureSeverity,
)
from kodiak.security.secrets import SecretManager


def _cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    if result.exit_code:
        raise RuntimeError(result.exception or result.stdout)


def _doctor_without_provider_key() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    if result.exit_code:
        raise RuntimeError(result.exception or result.stdout)
    json.loads(result.stdout)


def _secret_redaction() -> None:
    secret = "sk-" + "A" * 48
    output = asyncio.run(SecretManager().mask_secrets(f"token={secret}"))
    if secret in output or "REDACTED_OPENAI_API_KEY" not in output:
        raise RuntimeError("secret redaction regression")


def alpha_scenarios() -> list[AlphaScenario]:
    """Return the small, fast, no-side-effect release bug-bash set."""
    return [
        AlphaScenario(
            id="minimal-cli-startup",
            description="CLI starts without optional RAG dependencies",
            category=FailureCategory.INSTALLATION,
            severity=FailureSeverity.HIGH,
            execute=_cli_help,
            regression_test="test_minimal_install_cli_does_not_import_optional_rag",
        ),
        AlphaScenario(
            id="missing-provider-config",
            description="Diagnostics explain provider configuration without exposing values",
            category=FailureCategory.CONFIGURATION,
            severity=FailureSeverity.MEDIUM,
            execute=_doctor_without_provider_key,
            regression_test="test_doctor_json_is_machine_readable_and_never_contains_secret",
        ),
        AlphaScenario(
            id="secret-redaction",
            description="Known credential formats are redacted",
            category=FailureCategory.SECURITY,
            severity=FailureSeverity.CRITICAL,
            execute=_secret_redaction,
            regression_test="test_tool_output_is_bounded_control_free_and_secret_redacted",
        ),
    ]


def main() -> int:
    runner = AlphaBugBash()
    report = runner.health_report(runner.run(alpha_scenarios()))
    print(json.dumps(report, indent=2))
    return 0 if report["alpha_ready"] and report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
