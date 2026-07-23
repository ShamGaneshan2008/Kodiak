"""
Bridges the reviewer agent and GitHub's PR review API: posting automated
reviews/line comments, and parsing human review feedback back into
actionable revision requests for the agent loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from kodiak.github.client import get_client_for_repo
from kodiak.orchestration.reflection_loop import enqueue_revision_request

logger = logging.getLogger(__name__)

ReviewEvent = Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"]


@dataclass
class LineComment:
    path: str
    line: int
    body: str
    side: Literal["LEFT", "RIGHT"] = "RIGHT"


@dataclass
class ReviewResult:
    verdict: ReviewEvent
    summary: str
    line_comments: list[LineComment]


async def post_agent_review(
    repo_owner: str, repo_name: str, pr_number: int, result: ReviewResult
) -> dict[str, Any]:
    """Submit the reviewer agent's findings as a GitHub PR review."""
    client = await get_client_for_repo(repo_owner, repo_name)
    try:
        comments_payload = [
            {"path": c.path, "line": c.line, "body": c.body, "side": c.side}
            for c in result.line_comments
        ]
        review = await client.create_review(
            pr_number=pr_number,
            body=result.summary,
            event=result.verdict,
            comments=comments_payload or None,
        )
        logger.info(
            "Posted %s review on %s/%s#%s (%d line comments)",
            result.verdict,
            repo_owner,
            repo_name,
            pr_number,
            len(result.line_comments),
        )
        return review
    finally:
        await client.close()


async def post_check_run_result(
    repo_owner: str,
    repo_name: str,
    head_sha: str,
    check_name: str,
    passed: bool,
    summary: str,
    details: str = "",
) -> dict[str, Any]:
    """Reports automated check results (lint/tests/build) as a GitHub check run."""
    client = await get_client_for_repo(repo_owner, repo_name)
    try:
        return await client.create_check_run(
            name=check_name,
            head_sha=head_sha,
            status="completed",
            conclusion="success" if passed else "failure",
            output={"title": summary, "summary": details or summary},
        )
    finally:
        await client.close()


def _extract_review_comments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    review = payload.get("review", {})
    comment = payload.get("comment")
    items = []
    if comment:
        items.append(
            {
                "path": comment.get("path"),
                "line": comment.get("line") or comment.get("original_line"),
                "body": comment.get("body", ""),
            }
        )
    elif review.get("body"):
        items.append({"path": None, "line": None, "body": review["body"]})
    return items


async def handle_review_comment_event(payload: dict[str, Any]) -> None:
    """
    Webhook entrypoint for `pull_request_review` and
    `pull_request_review_comment` events. Human feedback requesting changes
    gets routed back into the agent's reflection loop as a revision request.
    """
    action = payload.get("action")
    if action not in ("submitted", "created"):
        return

    review = payload.get("review", {})
    state = review.get("state", "")  # approved | changes_requested | commented

    pr = payload.get("pull_request", {})
    repo = payload.get("repository", {})
    if not pr or not repo:
        return

    comments = _extract_review_comments(payload)
    if not comments:
        return

    if state == "approved" and not comments[0]["body"]:
        return

    logger.info(
        "Routing human review feedback (state=%s) for %s#%s back to reflection loop",
        state,
        repo.get("full_name"),
        pr.get("number"),
    )

    await enqueue_revision_request(
        repo_owner=repo["owner"]["login"],
        repo_name=repo["name"],
        pr_number=pr["number"],
        branch_name=pr["head"]["ref"],
        feedback=comments,
        requires_changes=(state == "changes_requested"),
    )
