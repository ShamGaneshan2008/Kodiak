from __future__ import annotations

from dataclasses import dataclass

from kodiak.github.code_review import review_changes
from kodiak.utils.diff import human_summary
from kodiak.utils.git_utils import GitChangeSet


@dataclass(frozen=True, slots=True)
class PullRequestDraft:
    title: str
    body: str
    draft: bool = True


def draft_pull_request(
    changes: GitChangeSet, commit_subject: str, testing: list[str] | None = None
) -> PullRequestDraft:
    testing = testing or ["Not run"]
    findings = review_changes(changes)
    risk_lines = [f"- `{finding.path}`: {finding.message}" for finding in findings] or [
        "- Low: no sensitive areas detected by local heuristics."
    ]

    body = "\n".join(
        [
            "## Summary",
            f"- {human_summary(changes)}",
            f"- Primary areas: {', '.join(changes.areas) if changes.areas else 'none'}",
            "",
            "## Testing",
            *[f"- {item}" for item in testing],
            "",
            "## Risk",
            *risk_lines,
        ]
    )
    return PullRequestDraft(title=commit_subject, body=body)
