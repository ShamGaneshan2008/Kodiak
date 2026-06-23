"""
Manages the full lifecycle of a pull request authored by an agent: branch
creation, committing generated file changes, opening/updating the PR, and
reacting to PR webhook events (review requested, merged, closed).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from kodiak.github.client import GitHubClient, get_client_for_repo
from kodiak.db.models.pull_request import PullRequestRecord, PRStatus
from kodiak.db.session import get_session
from kodiak.events.bus import publish_event

logger = logging.getLogger(__name__)

BRANCH_PREFIX = "kodiak"


@dataclass
class FileChange:
    path: str
    content: str
    sha: Optional[str] = None  # required when updating an existing file


def make_branch_name(task_id: str, slug: str) -> str:
    clean_slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")[:40]
    return f"{BRANCH_PREFIX}/{clean_slug}-{task_id[:8]}"


async def create_branch_with_changes(
    client: GitHubClient,
    task_id: str,
    slug: str,
    base_branch: str,
    changes: list[FileChange],
    commit_message: str,
) -> str:
    """Creates a feature branch and commits the given file changes to it."""
    base_sha = await client.get_default_branch_sha()
    branch_name = make_branch_name(task_id, slug)

    await client.create_branch(branch_name, base_sha)
    logger.info("Created branch %s from %s", branch_name, base_sha)

    for change in changes:
        await client.create_or_update_file(
            path=change.path,
            content=change.content,
            message=commit_message,
            branch=branch_name,
            sha=change.sha,
        )

    return branch_name


async def open_pull_request(
    repo_owner: str,
    repo_name: str,
    task_id: str,
    title: str,
    body: str,
    base_branch: str,
    changes: list[FileChange],
    draft: bool = False,
) -> dict[str, Any]:
    """End-to-end: branch + commits + PR creation, with a DB record."""
    client = await get_client_for_repo(repo_owner, repo_name)
    try:
        branch_name = await create_branch_with_changes(
            client,
            task_id=task_id,
            slug=title,
            base_branch=base_branch,
            changes=changes,
            commit_message=f"kodiak: {title}",
        )

        pr_data = await client.create_pull_request(
            title=title, head=branch_name, base=base_branch, body=body, draft=draft
        )

        async with get_session() as session:
            session.add(
                PullRequestRecord(
                    task_id=task_id,
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                    pr_number=pr_data["number"],
                    branch_name=branch_name,
                    status=PRStatus.OPEN,
                )
            )
            await session.commit()

        await publish_event(
            topic="github.pr_opened",
            payload={"repo": f"{repo_owner}/{repo_name}", "pr_number": pr_data["number"]},
        )
        logger.info("Opened PR #%s on %s/%s", pr_data["number"], repo_owner, repo_name)
        return pr_data
    finally:
        await client.close()


async def push_additional_commits(
    repo_owner: str,
    repo_name: str,
    branch_name: str,
    changes: list[FileChange],
    commit_message: str,
) -> None:
    """Used for review-feedback follow-up commits on an existing PR branch."""
    client = await get_client_for_repo(repo_owner, repo_name)
    try:
        for change in changes:
            await client.create_or_update_file(
                path=change.path,
                content=change.content,
                message=commit_message,
                branch=branch_name,
                sha=change.sha,
            )
    finally:
        await client.close()


async def handle_pull_request_event(payload: dict[str, Any]) -> None:
    """Webhook entrypoint for `pull_request` events."""
    action = payload.get("action")
    pr = payload["pull_request"]
    repo = payload["repository"]
    pr_number = pr["number"]

    logger.info("PR event '%s' for %s#%s", action, repo["full_name"], pr_number)

    status_map = {
        "closed": PRStatus.MERGED if pr.get("merged") else PRStatus.CLOSED,
        "reopened": PRStatus.OPEN,
    }
    new_status = status_map.get(action)
    if new_status is None:
        return

    async with get_session() as session:
        record = await session.get(PullRequestRecord, {"pr_number": pr_number, "repo_name": repo["name"]})
        if record:
            record.status = new_status
            await session.commit()

    await publish_event(
        topic=f"github.pr_{new_status.value}",
        payload={"repo": repo["full_name"], "pr_number": pr_number},
    )