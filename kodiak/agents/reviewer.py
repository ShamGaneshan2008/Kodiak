from __future__ import annotations

import json
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior code reviewer inside the Kodiak autonomous engineering system.
Review the provided diff/code changes against the task requirements and codebase context.

Evaluate across these dimensions:
- Correctness: does the code do what the task requires?
- Completeness: are all acceptance criteria addressed?
- Code quality: naming, structure, complexity, duplication.
- Security: injection, secrets, input validation, auth bypass risks.
- Performance: obvious inefficiencies, N+1 queries, blocking calls.
- Test coverage: are the changes adequately tested?
- Conventions: does it follow existing codebase patterns?

Output ONLY valid JSON:
{
  "verdict": "approved | needs_changes | rejected",
  "score": 0-100,
  "summary": "<one paragraph overall assessment>",
  "issues": [
    {
      "severity": "critical | major | minor | suggestion",
      "file": "path/to/file.py",
      "line_hint": "<line number or range if known>",
      "description": "<clear description of the issue>",
      "suggestion": "<concrete fix>"
    }
  ],
  "approved_files": ["path/to/file.py"],
  "must_fix_before_merge": ["<issue description>"]
}
"""


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        code_changes: list[dict] = input_.context.get("code_changes", [])
        rag_context: str = input_.context.get("rag_context", "")
        task_plan: dict = input_.context.get("plan", {})
        diff: str = input_.context.get("diff", "")

        if not code_changes and not diff:
            return self._make_error(input_, "code_changes or diff required in context")

        user_message = self._build_message(
            instruction=input_.instruction,
            code_changes=code_changes,
            diff=diff,
            rag_context=rag_context,
            task_plan=task_plan,
        )

        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
            max_tokens=3000,
        )

        raw = response.get("content", "")
        token_usage = response.get("usage", {})

        review = self._parse_review(raw)
        if review is None:
            return self._make_error(input_, f"Failed to parse review output: {raw[:200]}")

        return self._make_output(input_, result=review, token_usage=token_usage)

    def _build_message(
        self,
        instruction: str,
        code_changes: list[dict],
        diff: str,
        rag_context: str,
        task_plan: dict,
    ) -> str:
        parts = [f"## Original task\n{instruction}"]
        if task_plan:
            parts.append(f"## Acceptance criteria\n{json.dumps(task_plan.get('acceptance_criteria', []), indent=2)}")
        if diff:
            parts.append(f"## Diff\n```diff\n{diff}\n```")
        if code_changes:
            for change in code_changes:
                path = change.get("path", "unknown")
                content = change.get("content", "")
                lang = path.rsplit(".", 1)[-1] if "." in path else ""
                parts.append(f"### {path}\n```{lang}\n{content}\n```")
        if rag_context:
            parts.append(f"## Existing codebase context\n{rag_context}")
        return "\n\n".join(parts)

    def _parse_review(self, raw: str) -> dict | None:
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
            if "verdict" not in data:
                return None
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("reviewer.parse_failed", error=str(exc))
            return None