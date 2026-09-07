"""Curated public-alpha failure evidence built on Kodiak's evaluation layer."""

from __future__ import annotations

import enum
import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class FailureCategory(enum.StrEnum):
    INSTALLATION = "installation"
    CONFIGURATION = "configuration"
    CLI = "cli"
    PROVIDER = "provider"
    PLANNER = "planner"
    AGENT_SELECTION = "agent_selection"
    AGENT_EXECUTION = "agent_execution"
    TOOL = "tool"
    VERIFICATION = "verification"
    REFLECTION = "reflection"
    MEMORY = "memory"
    MULTI_AGENT = "multi_agent"
    GIT = "git"
    GITHUB = "github"
    CI = "ci"
    CHECKPOINT = "checkpoint"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class FailureSeverity(enum.StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Reproducibility(enum.StrEnum):
    REPRODUCIBLE = "reproducible"
    INTERMITTENT = "intermittent"
    ENVIRONMENT_SPECIFIC = "environment_specific"
    NOT_REPRODUCED = "not_reproduced"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_VOLATILE = re.compile(
    r"(?:[A-Za-z]:)?[/\\][^\s:]+|\b0x[0-9a-f]+\b|\b[0-9a-f]{8,}\b|\b\d+\b",
    re.IGNORECASE,
)


def failure_fingerprint(
    category: FailureCategory,
    exception_type: str | None,
    component: str,
    message: str,
) -> str:
    """Return a stable, non-secret duplicate key for equivalent failures."""
    normalized = _VOLATILE.sub("?", message.lower())
    normalized = " ".join(normalized.split())[:300]
    material = "|".join((category.value, exception_type or "", component.lower(), normalized))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AlphaScenario:
    id: str
    description: str
    category: FailureCategory
    severity: FailureSeverity
    execute: Callable[[], Any]
    mutates_repository: bool = False
    regression_test: str | None = None
    issue_reference: str | None = None


@dataclass(slots=True)
class AlphaResult:
    scenario: str
    status: str
    category: FailureCategory
    severity: FailureSeverity
    duration_seconds: float
    reproduced: Reproducibility
    fingerprint: str | None = None
    regression_test: str | None = None
    issue_reference: str | None = None
    error_type: str | None = None
    error: str | None = None
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    occurrence_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "status": self.status,
            "failure_category": self.category.value,
            "severity": self.severity.value,
            "duration_seconds": round(self.duration_seconds, 6),
            "reproduced": self.reproduced.value,
            "fingerprint": self.fingerprint,
            "regression_test": self.regression_test,
            "issue_reference": self.issue_reference,
            "error_type": self.error_type,
            "error": self.error,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count,
        }


class AlphaBugBash:
    """Run explicit local scenarios and aggregate duplicate failure evidence."""

    def __init__(self) -> None:
        self._failures: dict[str, AlphaResult] = {}

    def run(self, scenarios: list[AlphaScenario]) -> list[AlphaResult]:
        results: list[AlphaResult] = []
        for scenario in scenarios:
            if scenario.mutates_repository:
                raise ValueError(
                    f"Scenario {scenario.id!r} requires an explicit isolated repository"
                )
            started = time.perf_counter()
            try:
                scenario.execute()
                result = AlphaResult(
                    scenario=scenario.id,
                    status="passed",
                    category=scenario.category,
                    severity=scenario.severity,
                    duration_seconds=time.perf_counter() - started,
                    reproduced=Reproducibility.NOT_REPRODUCED,
                    regression_test=scenario.regression_test,
                    issue_reference=scenario.issue_reference,
                )
            except Exception as exc:
                error_type = type(exc).__name__
                fingerprint = failure_fingerprint(
                    scenario.category, error_type, scenario.id, str(exc)
                )
                result = AlphaResult(
                    scenario=scenario.id,
                    status="failed",
                    category=scenario.category,
                    severity=scenario.severity,
                    duration_seconds=time.perf_counter() - started,
                    reproduced=Reproducibility.REPRODUCIBLE,
                    fingerprint=fingerprint,
                    regression_test=scenario.regression_test,
                    issue_reference=scenario.issue_reference,
                    error_type=error_type,
                    error=str(exc),
                )
                prior = self._failures.get(fingerprint)
                if prior:
                    prior.last_seen = result.last_seen
                    prior.occurrence_count += 1
                    result = prior
                else:
                    self._failures[fingerprint] = result
            results.append(result)
        return results

    @staticmethod
    def health_report(results: list[AlphaResult]) -> dict[str, Any]:
        failed = [result for result in results if result.status == "failed"]
        unique = {result.fingerprint for result in failed}
        severities = {
            severity.value: sum(result.severity is severity for result in failed)
            for severity in FailureSeverity
        }
        blockers = sorted(
            result.fingerprint
            for result in failed
            if result.severity is FailureSeverity.CRITICAL and result.fingerprint
        )
        return {
            "total_scenarios": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            **severities,
            "unique_failures": len(unique),
            "new_regressions": sum(
                bool(result.regression_test) and result.status == "failed" for result in results
            ),
            "environment_specific": sum(
                result.reproduced is Reproducibility.ENVIRONMENT_SPECIFIC for result in results
            ),
            "release_blockers": blockers,
            "alpha_ready": not blockers,
            "results": [result.to_dict() for result in results],
        }
