from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior software engineering planner inside the Kodiak autonomous agent system.
Your job is to decompose a software development task into a precise, ordered list of subtasks.

Rules:
- Each subtask must be atomic and independently verifiable.
- Order subtasks so earlier ones unblock later ones.
- Include a testing subtask for every implementation subtask.
- Identify files likely to be modified for each subtask.
- Set complexity: low | medium | high based on estimated LLM effort.
- Output ONLY valid JSON — no prose, no markdown fences.

Output schema:
{
  "goal": "<one-sentence restatement of the overall goal>",
  "acceptance_criteria": ["<criterion>", ...],
  "subtasks": [
    {
      "id": "st-1",
      "title": "<short title>",
      "description": "<what needs to be done>",
      "type": "implementation | test | documentation | review",
      "complexity": "low | medium | high",
      "depends_on": [],
      "likely_files": ["path/to/file.py"]
    }
  ],
  "estimated_total_complexity": "low | medium | high",
  "requires_architecture_review": true | false
}
"""


@dataclass
class SubTask:
    id: str
    title: str
    description: str
    type: str
    complexity: str
    depends_on: list[str]
    likely_files: list[str]


@dataclass
class TaskPlan:
    goal: str
    acceptance_criteria: list[str]
    subtasks: list[SubTask]
    estimated_total_complexity: str
    requires_architecture_review: bool


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        repo_context = input_.context.get("repo_context", "")
        similar_tasks = input_.context.get("similar_tasks", [])

        user_message = self._build_user_message(
            instruction=input_.instruction,
            repo_context=repo_context,
            similar_tasks=similar_tasks,
        )

        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
            max_tokens=4096,
        )

        raw = response.get("content", "")
        token_usage = response.get("usage", {})

        plan = self._parse_plan(raw)
        if plan is None:
            return self._make_error(input_, f"Failed to parse plan from LLM output: {raw[:200]}")

        return self._make_output(
            input_,
            result={"plan": self._plan_to_dict(plan)},
            token_usage=token_usage,
        )

    def _build_user_message(
        self,
        instruction: str,
        repo_context: str,
        similar_tasks: list[dict],
    ) -> str:
        parts = [f"## Task\n{instruction}"]
        if repo_context:
            parts.append(f"## Repository context\n{repo_context}")
        if similar_tasks:
            examples = json.dumps(similar_tasks[:3], indent=2)
            parts.append(f"## Similar past tasks (for reference)\n{examples}")
        return "\n\n".join(parts)

    def _parse_plan(self, raw: str) -> TaskPlan | None:
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
            subtasks = [
                SubTask(
                    id=st["id"],
                    title=st["title"],
                    description=st["description"],
                    type=st.get("type", "implementation"),
                    complexity=st.get("complexity", "medium"),
                    depends_on=st.get("depends_on", []),
                    likely_files=st.get("likely_files", []),
                )
                for st in data.get("subtasks", [])
            ]
            return TaskPlan(
                goal=data.get("goal", ""),
                acceptance_criteria=data.get("acceptance_criteria", []),
                subtasks=subtasks,
                estimated_total_complexity=data.get("estimated_total_complexity", "medium"),
                requires_architecture_review=data.get("requires_architecture_review", False),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("planner.parse_failed", error=str(exc))
            return None

    def _plan_to_dict(self, plan: TaskPlan) -> dict:
        return {
            "goal": plan.goal,
            "acceptance_criteria": plan.acceptance_criteria,
            "subtasks": [
                {
                    "id": st.id,
                    "title": st.title,
                    "description": st.description,
                    "type": st.type,
                    "complexity": st.complexity,
                    "depends_on": st.depends_on,
                    "likely_files": st.likely_files,
                }
                for st in plan.subtasks
            ],
            "estimated_total_complexity": plan.estimated_total_complexity,
            "requires_architecture_review": plan.requires_architecture_review,
        }