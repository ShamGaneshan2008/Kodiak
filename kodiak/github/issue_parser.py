"""
Translates raw GitHub `issues` / `issue_comment` webhook payloads into
structured TaskSpec objects that the planner/orchestration layer consumes.

Also handles slash-command style triggers in issue comments
(e.g. `/kodiak fix`, `/kodiak plan`).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from kodiak.orchestration.task_planner import enqueue_task
from kodiak.db.models.task import TaskSource

logger = logging.getLogger(__name__)

COMMAND_PATTERN = re.compile(r"^/kodiak\s+(?P<command>\w+)\s*(?P<args>.*)$", re.IGNORECASE | re.MULTILINE)

LABEL_TRIGGERS = {"kodiak", "kodiak:auto-fix", "kodiak:plan"}


@dataclass
class IssueTaskSpec:
    repo_owner: str
    repo_name: str
    issue_number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    command: Optional[str] = None
    command_args: str = ""
    author: str = ""
    source: TaskSource = TaskSource.GITHUB_ISSUE

    @property
    def should_auto_process(self) -> bool:
        return bool(self.command) or bool(LABEL_TRIGGERS.intersection(self.labels))


def parse_issue_payload(payload: dict[str, Any]) -> IssueTaskSpec:
    issue = payload["issue"]
    repo = payload["repository"]

    labels = [label["name"] for label in issue.get("labels", [])]
    body = issue.get("body") or ""

    command, args = _extract_command(body)

    return IssueTaskSpec(
        repo_owner=repo["owner"]["login"],
        repo_name=repo["name"],
        issue_number=issue["number"],
        title=issue["title"],
        body=body,
        labels=labels,
        command=command,
        command_args=args,
        author=issue["user"]["login"],
    )


def parse_issue_comment_payload(payload: dict[str, Any]) -> Optional[IssueTaskSpec]:
    """Issue comments can also trigger commands (e.g. `/kodiak fix` posted later)."""
    comment_body = payload.get("comment", {}).get("body", "")
    command, args = _extract_command(comment_body)
    if command is None:
        return None

    issue = payload["issue"]
    repo = payload["repository"]
    labels = [label["name"] for label in issue.get("labels", [])]

    return IssueTaskSpec(
        repo_owner=repo["owner"]["login"],
        repo_name=repo["name"],
        issue_number=issue["number"],
        title=issue["title"],
        body=issue.get("body") or "",
        labels=labels,
        command=command,
        command_args=args,
        author=payload["comment"]["user"]["login"],
    )


def _extract_command(text: str) -> tuple[Optional[str], str]:
    match = COMMAND_PATTERN.search(text)
    if not match:
        return None, ""
    return match.group("command").lower(), match.group("args").strip()


async def handle_issue_event(payload: dict[str, Any]) -> None:
    """Webhook entrypoint for `issues` and `issue_comment` events."""
    action = payload.get("action")
    event_type = "issue_comment" if "comment" in payload else "issues"

    if event_type == "issues" and action not in ("opened", "labeled", "edited"):
        return
    if event_type == "issue_comment" and action != "created":
        return

    spec: Optional[IssueTaskSpec]
    if event_type == "issues":
        spec = parse_issue_payload(payload)
    else:
        spec = parse_issue_comment_payload(payload)

    if spec is None or not spec.should_auto_process:
        logger.debug("Issue event did not meet auto-process criteria; skipping")
        return

    logger.info(
        "Enqueuing task for %s/%s#%s (command=%s)",
        spec.repo_owner, spec.repo_name, spec.issue_number, spec.command,
    )
    await enqueue_task(
        repo_owner=spec.repo_owner,
        repo_name=spec.repo_name,
        title=spec.title,
        description=spec.body,
        source=spec.source,
        source_ref=str(spec.issue_number),
        command=spec.command or "plan",
        command_args=spec.command_args,
    )