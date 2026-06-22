from __future__ import annotations

from collections import Counter

from kodiak.utils.git_utils import GitChangeSet, GitFileChange

RISKY_AREAS = {
    ".github": "automation",
    "alembic": "database migration",
    "auth": "authentication",
    "config": "configuration",
    "db": "database",
    "docker": "container/runtime",
    "security": "security",
}


def summarize_changes(changes: GitChangeSet) -> dict:
    area_counts = Counter(change.area for change in changes.files)
    return {
        "branch": changes.branch,
        "files_changed": len(changes.files),
        "additions": changes.total_additions,
        "deletions": changes.total_deletions,
        "areas": dict(area_counts.most_common()),
        "risky_files": risky_files(changes.files),
    }


def risky_files(files: tuple[GitFileChange, ...] | list[GitFileChange]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    for change in files:
        area = change.area
        if area in RISKY_AREAS:
            risks.append({"path": change.path, "reason": RISKY_AREAS[area]})
    return risks


def human_summary(changes: GitChangeSet) -> str:
    if changes.is_empty:
        return "No local changes."
    areas = ", ".join(changes.areas[:6])
    if len(changes.areas) > 6:
        areas += f", and {len(changes.areas) - 6} more"
    return (
        f"{len(changes.files)} files changed across {areas}; "
        f"+{changes.total_additions}/-{changes.total_deletions}."
    )


def changed_files(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git") and " b/" in line:
            files.append(line.split(" b/", 1)[1])
    return files
