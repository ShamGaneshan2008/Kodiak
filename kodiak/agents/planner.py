from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior software engineering planner inside the Kodiak autonomous agent system.
Your job is to transform a natural-language software engineering goal into a precise,
machine-readable execution plan for autonomous agents.

Rules:
- Each subtask must be atomic and independently verifiable.
- Order subtasks so earlier ones unblock later ones.
- Include repository-inspection steps before any implementation step.
- Include a testing or verification step for every implementation step.
- Identify files that should be inspected before each subtask.
- Identify files likely to be modified for each implementation or test subtask.
- Choose tools for every step. Prefer existing Kodiak tools and agents.
- Model dependencies explicitly with depends_on task ids.
- Assign the same parallel_group to tasks that can run concurrently.
- Set complexity: low | medium | high based on estimated LLM effort.
- Do not perform code modification. Planning output only.
- Output ONLY valid JSON - no prose, no markdown fences.

Output schema:
{
  "plan_version": "1.0",
  "goal": "<one-sentence restatement of the overall goal>",
  "acceptance_criteria": ["<criterion>", "..."],
  "repository_files": ["path/to/file.py"],
  "execution_order": ["st-1", "st-2"],
  "parallel_groups": [["st-2", "st-3"]],
  "estimated_dependencies": [
    {
      "from_task": "st-1",
      "to_task": "st-2",
      "reason": "<why st-2 depends on st-1>"
    }
  ],
  "subtasks": [
    {
      "id": "st-1",
      "title": "<short title>",
      "description": "<what needs to be done>",
      "type": "inspection | research | implementation | test | documentation | review",
      "complexity": "low | medium | high",
      "depends_on": [],
      "files_to_inspect": ["path/to/file.py"],
      "tools": [
        {
          "name": "context_builder",
          "purpose": "<why this tool is needed>",
          "required_capability": "repository_context",
          "inputs": {"query": "<task-specific query>"},
          "risk_level": "low | medium | high"
        }
      ],
      "parallel_group": "pg-1",
      "can_run_parallel": true,
      "likely_files": ["path/to/file.py"]
    }
  ],
  "estimated_total_complexity": "low | medium | high",
  "requires_architecture_review": true | false
}
"""


@dataclass
class PlannedTool:
    name: str
    purpose: str
    required_capability: str | None
    inputs: dict[str, Any]
    risk_level: str


@dataclass
class TaskDependency:
    from_task: str
    to_task: str
    reason: str


@dataclass
class SubTask:
    id: str
    title: str
    description: str
    type: str
    complexity: str
    depends_on: list[str] = field(default_factory=list)
    likely_files: list[str] = field(default_factory=list)
    files_to_inspect: list[str] = field(default_factory=list)
    tools: list[PlannedTool] = field(default_factory=list)
    parallel_group: str | None = None
    can_run_parallel: bool = False


@dataclass
class TaskPlan:
    goal: str
    acceptance_criteria: list[str]
    subtasks: list[SubTask]
    estimated_total_complexity: str
    requires_architecture_review: bool
    plan_version: str = "1.0"
    repository_files: list[str] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    estimated_dependencies: list[TaskDependency] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return PlannerAgent.plan_to_dict(self)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        repo_context = input_.context.get("repo_context", "")
        similar_tasks = input_.context.get("similar_tasks", [])
        available_tools = input_.context.get("available_tools", [])

        user_message = self._build_user_message(
            instruction=input_.instruction,
            repo_context=repo_context,
            similar_tasks=similar_tasks,
            available_tools=available_tools,
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
            result={"plan": self.plan_to_dict(plan)},
            token_usage=token_usage,
            metadata={
                "plan_version": plan.plan_version,
                "steps": len(plan.subtasks),
                "parallel_groups": len(plan.parallel_groups),
            },
        )

    def _build_user_message(
        self,
        instruction: str,
        repo_context: str,
        similar_tasks: list[dict],
        available_tools: list[dict] | list[str],
    ) -> str:
        parts = [f"## Task\n{instruction}"]
        if repo_context:
            parts.append(f"## Repository context\n{repo_context}")
        if available_tools:
            parts.append(
                "## Available tools\n"
                f"{json.dumps(available_tools, indent=2, sort_keys=True, default=str)}"
            )
        if similar_tasks:
            examples = json.dumps(similar_tasks[:3], indent=2)
            parts.append(f"## Similar past tasks (for reference)\n{examples}")
        return "\n\n".join(parts)

    def _parse_plan(self, raw: str) -> TaskPlan | None:
        try:
            data = json.loads(self._strip_json_fence(raw))
            if not isinstance(data, dict):
                return None

            subtasks = [
                self._parse_subtask(st)
                for st in data.get("subtasks", [])
                if isinstance(st, dict)
            ]
            task_ids = {task.id for task in subtasks}
            for task in subtasks:
                task.depends_on = [dep for dep in task.depends_on if dep in task_ids]

            execution_order = self._coerce_string_list(data.get("execution_order"))
            if execution_order:
                execution_order = [task_id for task_id in execution_order if task_id in task_ids]
            if not execution_order:
                execution_order = self._topological_order(subtasks)

            parallel_groups = self._coerce_parallel_groups(data.get("parallel_groups"), task_ids)
            if not parallel_groups:
                parallel_groups = self._parallel_groups(subtasks, execution_order)

            repository_files = self._coerce_string_list(data.get("repository_files"))
            if not repository_files:
                repository_files = self._repository_files(subtasks)

            dependencies = [
                self._parse_dependency(item)
                for item in data.get("estimated_dependencies", [])
                if isinstance(item, dict)
            ]
            dependencies = [
                dep
                for dep in dependencies
                if dep.from_task in task_ids and dep.to_task in task_ids
            ]
            if not dependencies:
                dependencies = self._dependencies_from_tasks(subtasks)

            return TaskPlan(
                goal=str(data.get("goal", "")),
                acceptance_criteria=self._coerce_string_list(
                    data.get("acceptance_criteria")
                ),
                subtasks=subtasks,
                estimated_total_complexity=str(
                    data.get("estimated_total_complexity", "medium")
                ),
                requires_architecture_review=bool(
                    data.get("requires_architecture_review", False)
                ),
                plan_version=str(data.get("plan_version", "1.0")),
                repository_files=repository_files,
                execution_order=execution_order,
                parallel_groups=parallel_groups,
                estimated_dependencies=dependencies,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("planner.parse_failed", error=str(exc))
            return None

    @staticmethod
    def plan_to_dict(plan: TaskPlan) -> dict[str, Any]:
        return {
            "plan_version": plan.plan_version,
            "goal": plan.goal,
            "acceptance_criteria": plan.acceptance_criteria,
            "repository_files": plan.repository_files,
            "execution_order": plan.execution_order,
            "parallel_groups": plan.parallel_groups,
            "estimated_dependencies": [
                {
                    "from_task": dep.from_task,
                    "to_task": dep.to_task,
                    "reason": dep.reason,
                }
                for dep in plan.estimated_dependencies
            ],
            "subtasks": [
                {
                    "id": st.id,
                    "title": st.title,
                    "description": st.description,
                    "type": st.type,
                    "complexity": st.complexity,
                    "depends_on": st.depends_on,
                    "files_to_inspect": st.files_to_inspect,
                    "tools": [
                        {
                            "name": tool.name,
                            "purpose": tool.purpose,
                            "required_capability": tool.required_capability,
                            "inputs": tool.inputs,
                            "risk_level": tool.risk_level,
                        }
                        for tool in st.tools
                    ],
                    "parallel_group": st.parallel_group,
                    "can_run_parallel": st.can_run_parallel,
                    "likely_files": st.likely_files,
                }
                for st in plan.subtasks
            ],
            "estimated_total_complexity": plan.estimated_total_complexity,
            "requires_architecture_review": plan.requires_architecture_review,
        }

    def _plan_to_dict(self, plan: TaskPlan) -> dict[str, Any]:
        return self.plan_to_dict(plan)

    def _parse_subtask(self, st: dict[str, Any]) -> SubTask:
        task_type = str(st.get("type", "implementation"))
        likely_files = self._coerce_string_list(st.get("likely_files"))
        files_to_inspect = self._coerce_string_list(st.get("files_to_inspect"))
        if not files_to_inspect:
            files_to_inspect = list(likely_files)
        tools = [
            self._parse_tool(tool)
            for tool in st.get("tools", [])
            if isinstance(tool, dict)
        ]
        if not tools:
            tools = self._default_tools_for_task(task_type, st)
        return SubTask(
            id=str(st["id"]),
            title=str(st["title"]),
            description=str(st["description"]),
            type=task_type,
            complexity=str(st.get("complexity", "medium")),
            depends_on=self._coerce_string_list(st.get("depends_on")),
            likely_files=likely_files,
            files_to_inspect=files_to_inspect,
            tools=tools,
            parallel_group=(
                str(st["parallel_group"]) if st.get("parallel_group") is not None else None
            ),
            can_run_parallel=bool(st.get("can_run_parallel", False)),
        )

    def _parse_tool(self, tool: dict[str, Any]) -> PlannedTool:
        inputs = tool.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {"value": inputs}
        required_capability = tool.get("required_capability")
        return PlannedTool(
            name=str(tool.get("name", "planner")),
            purpose=str(tool.get("purpose", "")),
            required_capability=(
                str(required_capability) if required_capability is not None else None
            ),
            inputs=inputs,
            risk_level=str(tool.get("risk_level", "low")),
        )

    def _parse_dependency(self, item: dict[str, Any]) -> TaskDependency:
        return TaskDependency(
            from_task=str(item.get("from_task", "")),
            to_task=str(item.get("to_task", "")),
            reason=str(item.get("reason", "")),
        )

    def _default_tools_for_task(
        self,
        task_type: str,
        st: dict[str, Any],
    ) -> list[PlannedTool]:
        tool_by_type = {
            "inspection": ("context_builder", "repository_context"),
            "research": ("semantic_search", "repository_search"),
            "implementation": ("coder", "code_generation"),
            "test": ("tester", "test_generation"),
            "documentation": ("coder", "documentation_edit"),
            "review": ("reviewer", "code_review"),
        }
        name, capability = tool_by_type.get(task_type, ("planner", "task_planning"))
        return [
            PlannedTool(
                name=name,
                purpose=f"Support planned task: {st.get('title', '')}",
                required_capability=capability,
                inputs={"task_id": str(st.get("id", ""))},
                risk_level="low" if task_type in {"inspection", "research", "review"} else "medium",
            )
        ]

    def _strip_json_fence(self, raw: str) -> str:
        clean = raw.strip()
        if not clean.startswith("```"):
            return clean
        first_newline = clean.find("\n")
        if first_newline == -1:
            return clean.strip("`")
        clean = clean[first_newline + 1 :]
        if clean.endswith("```"):
            clean = clean[:-3]
        return clean.strip()

    def _coerce_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None]

    def _coerce_parallel_groups(
        self,
        value: Any,
        task_ids: set[str],
    ) -> list[list[str]]:
        if not isinstance(value, list):
            return []
        groups: list[list[str]] = []
        for group in value:
            if not isinstance(group, list):
                continue
            ids = [str(task_id) for task_id in group if str(task_id) in task_ids]
            if ids:
                groups.append(ids)
        return groups

    def _topological_order(self, subtasks: list[SubTask]) -> list[str]:
        pending = {task.id: set(task.depends_on) for task in subtasks}
        order: list[str] = []
        while pending:
            ready = sorted(task_id for task_id, deps in pending.items() if not deps)
            if not ready:
                order.extend(sorted(pending))
                break
            order.extend(ready)
            for task_id in ready:
                pending.pop(task_id, None)
            for deps in pending.values():
                deps.difference_update(ready)
        return order

    def _parallel_groups(
        self,
        subtasks: list[SubTask],
        execution_order: list[str],
    ) -> list[list[str]]:
        tasks = {task.id: task for task in subtasks}
        remaining = list(execution_order)
        completed: set[str] = set()
        groups: list[list[str]] = []
        while remaining:
            ready = [
                task_id
                for task_id in remaining
                if set(tasks[task_id].depends_on).issubset(completed)
            ]
            if not ready:
                ready = [remaining[0]]
            groups.append(ready)
            completed.update(ready)
            remaining = [task_id for task_id in remaining if task_id not in ready]
        return groups

    def _repository_files(self, subtasks: list[SubTask]) -> list[str]:
        files: list[str] = []
        for task in subtasks:
            files.extend(task.files_to_inspect)
            files.extend(task.likely_files)
        return sorted(dict.fromkeys(path for path in files if path))

    def _dependencies_from_tasks(self, subtasks: list[SubTask]) -> list[TaskDependency]:
        dependencies: list[TaskDependency] = []
        for task in subtasks:
            dependencies.extend(
                TaskDependency(
                    from_task=dependency,
                    to_task=task.id,
                    reason="Declared task dependency.",
                )
                for dependency in task.depends_on
            )
        return dependencies
