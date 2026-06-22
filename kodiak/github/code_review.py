from __future__ import annotations

from dataclasses import asdict, dataclass

from kodiak.utils.diff import risky_files
from kodiak.utils.git_utils import GitChangeSet


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    severity: str
    path: str
    message: str


def review_changes(changes: GitChangeSet) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for risk in risky_files(changes.files):
        findings.append(
            ReviewFinding(
                severity="medium",
                path=risk["path"],
                message=f"Touches {risk['reason']} code; verify behavior and rollback path.",
            )
        )
    if changes.total_additions > 1000:
        findings.append(
            ReviewFinding(
                severity="low",
                path=".",
                message="Large change set; consider splitting future work into smaller commits.",
            )
        )
    return findings


def summarize_review(changes: GitChangeSet) -> dict:
    findings = review_changes(changes)
    return {
        "findings": [asdict(finding) for finding in findings],
        "requires_human_attention": any(
            finding.severity in {"high", "medium"} for finding in findings
        ),
    }
