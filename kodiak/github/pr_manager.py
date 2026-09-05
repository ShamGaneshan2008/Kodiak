"""Pull-request drafting and idempotent create/update helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kodiak.github.client import GitHubClient
from kodiak.utils.git_utils import GitChangeSet, human_summary

BRANCH_PREFIX = "kodiak/task"


@dataclass(frozen=True, slots=True)
class PullRequestDraft:
    title: str
    body: str
    draft: bool = False


def make_branch_name(task_id: str, slug: str) -> str:
    clean_slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")[:40]
    return f"{BRANCH_PREFIX}/{clean_slug or 'work'}-{task_id[:8]}"


def draft_pull_request(
    changes: GitChangeSet,
    title: str,
    *,
    testing: list[str] | None = None,
) -> PullRequestDraft:
    tests = testing or ["Verification not recorded"]
    body = "\n".join(
        [
            "## Summary",
            "",
            human_summary(changes),
            "",
            "## Verification",
            "",
            *[f"- {item}" for item in tests],
            "",
            "## Risks",
            "",
            "- Awaiting human review; no automatic merge was performed.",
        ]
    )
    return PullRequestDraft(title=title, body=body)


async def create_or_update_pull_request(
    client: GitHubClient,
    *,
    owner: str,
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
) -> tuple[dict[str, Any], bool]:
    """Reuse an open PR for ``head`` or create exactly one new PR."""
    existing = await client.list_pull_requests(
        owner,
        repo,
        state="all",
        head=f"{owner}:{head}",
        base=base,
    )
    open_pr = next((pr for pr in existing if pr.get("state") == "open"), None)
    if open_pr:
        updated = await client.update_pull_request(
            owner,
            repo,
            int(open_pr["number"]),
            title=title,
            body=body,
        )
        return updated, False
    if existing:
        raise RuntimeError("A closed or merged pull request already exists for this branch.")
    created = await client.create_pull_request(owner, repo, title, head, base, body)
    return created, True


__all__ = [
    "PullRequestDraft",
    "create_or_update_pull_request",
    "draft_pull_request",
    "make_branch_name",
]
