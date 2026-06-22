from __future__ import annotations

from collections import Counter

from kodiak.agents.base import AgentContext, AgentResult, BaseAgent
from kodiak.github.pr_manager import draft_pull_request
from kodiak.utils.diff import human_summary, summarize_changes
from kodiak.utils.git_utils import GitChangeSet, make_branch_name, read_changes


class GitAgent(BaseAgent):
    name = "git"

    async def run(self, context: AgentContext) -> AgentResult:
        repo = context.repository or "."
        changes = read_changes(repo)
        task_title = str(context.inputs.get("title") or context.task_id)
        commit_plan = build_commit_plan(changes, task_title)
        pr_draft = draft_pull_request(
            changes,
            commit_plan["subject"],
            testing=list(context.inputs.get("testing", [])) or ["Syntax compile check"],
        )
        return AgentResult(
            agent=self.name,
            status="completed",
            output={
                "branch": make_branch_name(task_title),
                "summary": human_summary(changes),
                "changes": summarize_changes(changes),
                "commit": commit_plan,
                "pull_request": {
                    "title": pr_draft.title,
                    "body": pr_draft.body,
                    "draft": pr_draft.draft,
                },
            },
        )


def build_commit_plan(changes: GitChangeSet, task_title: str) -> dict[str, str | list[str]]:
    if changes.is_empty:
        return {
            "subject": "chore: no local changes detected",
            "body": "No modified files were found in the working tree.",
            "footer": [],
        }

    commit_type = infer_commit_type(changes)
    scope = infer_scope(changes)
    verb = infer_subject_verb(changes)
    subject = f"{commit_type}({scope}): {verb}"
    body = "\n".join(
        [
            f"Task: {task_title}",
            "",
            human_summary(changes),
            "",
            "Changed areas:",
            *[f"- {area}" for area in changes.areas],
        ]
    )
    footers = []
    if any(
        change.area in {"auth", "security", "db", "docker", ".github"} for change in changes.files
    ):
        footers.append("Review-sensitive: true")
    return {"subject": subject, "body": body, "footer": footers}


def infer_commit_type(changes: GitChangeSet) -> str:
    areas = set(changes.areas)
    paths = changes.changed_paths
    if any(path.startswith(("tests/", "test_")) or "/tests/" in path for path in paths):
        return "test"
    if areas & {"docker", ".github"}:
        return "ci"
    if areas & {"docs"} or all(path.endswith((".md", ".rst")) for path in paths):
        return "docs"
    if areas & {"config", "db", "api", "github", "orchestration", "agents"}:
        return "feat"
    return "chore"


def infer_scope(changes: GitChangeSet) -> str:
    counts = Counter(changes.areas)
    scope, _ = counts.most_common(1)[0]
    return scope.replace("_", "-")[:24]


def infer_subject_verb(changes: GitChangeSet) -> str:
    areas = set(changes.areas)
    if "github" in areas or "git" in areas:
        return "improve commit and PR workflow"
    if "api" in areas:
        return "wire API endpoints"
    if "db" in areas:
        return "add persistence models"
    if "rag" in areas:
        return "add retrieval pipeline foundation"
    if "agents" in areas:
        return "add agent execution foundation"
    return "fill project implementation"
