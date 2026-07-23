from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ExecutableTask(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    agent_type: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[uuid.UUID] = Field(default_factory=list)
    plan_step_id: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    files_to_inspect: list[str] = Field(default_factory=list)
    parallel_group: str | None = None
    can_run_parallel: bool = False


class ExecutionPlan(BaseModel):
    goal: str
    tasks: list[ExecutableTask]
    execution_order: list[uuid.UUID]
    parallel_groups: list[list[uuid.UUID]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def machine_readable(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "execution_order": [str(task_id) for task_id in self.execution_order],
            "parallel_groups": [
                [str(task_id) for task_id in group] for group in self.parallel_groups
            ],
            "tasks": [
                {
                    "id": str(task.id),
                    "name": task.name,
                    "agent_type": task.agent_type,
                    "dependencies": [str(dep) for dep in task.dependencies],
                    "plan_step_id": task.plan_step_id,
                    "tool_names": task.tool_names,
                    "files_to_inspect": task.files_to_inspect,
                    "parallel_group": task.parallel_group,
                    "can_run_parallel": task.can_run_parallel,
                    "input_data": task.input_data,
                }
                for task in self.tasks
            ],
            "metadata": self.metadata,
        }


class TaskPlanner:
    async def plan(self, goal: str, context: dict[str, Any] | None = None) -> list[ExecutableTask]:
        return (await self.plan_execution(goal, context)).tasks

    async def plan_execution(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        ctx = context or {}
        planner_plan = self._extract_planner_plan(ctx)
        if planner_plan is not None:
            return self._plan_from_planner_output(goal, planner_plan, ctx)

        tasks = self._legacy_tasks(goal, ctx)
        return ExecutionPlan(
            goal=goal,
            tasks=tasks,
            execution_order=[task.id for task in tasks],
            parallel_groups=self._parallel_groups_from_tasks(tasks),
            metadata={"source": "heuristic"},
        )

    def _plan_from_planner_output(
        self,
        goal: str,
        planner_plan: Mapping[str, Any],
        ctx: dict[str, Any],
    ) -> ExecutionPlan:
        subtasks = self._subtasks(planner_plan)
        task_by_step_id: dict[str, ExecutableTask] = {}

        for subtask in subtasks:
            step_id = str(subtask.get("id", uuid.uuid4().hex))
            task = ExecutableTask(
                name=self._task_name(subtask),
                agent_type=self._agent_type(subtask),
                input_data=self._input_data(goal, subtask, planner_plan, ctx),
                plan_step_id=step_id,
                tool_names=self._tool_names(subtask),
                files_to_inspect=self._string_list(subtask.get("files_to_inspect")),
                parallel_group=(
                    str(subtask["parallel_group"])
                    if subtask.get("parallel_group") is not None
                    else None
                ),
                can_run_parallel=bool(subtask.get("can_run_parallel", False)),
            )
            task_by_step_id[step_id] = task

        for subtask in subtasks:
            step_id = str(subtask.get("id", ""))
            task = task_by_step_id.get(step_id)
            if task is None:
                continue
            task.dependencies = [
                task_by_step_id[dependency].id
                for dependency in self._string_list(subtask.get("depends_on"))
                if dependency in task_by_step_id
            ]

        execution_order = [
            task_by_step_id[step_id].id
            for step_id in self._string_list(planner_plan.get("execution_order"))
            if step_id in task_by_step_id
        ]
        if not execution_order:
            execution_order = [task.id for task in task_by_step_id.values()]

        parallel_groups = self._planner_parallel_groups(
            planner_plan.get("parallel_groups"),
            task_by_step_id,
        )
        if not parallel_groups:
            parallel_groups = self._parallel_groups_from_tasks(list(task_by_step_id.values()))

        logger.info(
            "planner_execution_plan_created",
            tasks=len(task_by_step_id),
            source="planner_agent",
        )
        return ExecutionPlan(
            goal=str(planner_plan.get("goal") or goal),
            tasks=list(task_by_step_id.values()),
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            metadata={
                "source": "planner_agent",
                "plan_version": planner_plan.get("plan_version", "1.0"),
                "acceptance_criteria": self._string_list(planner_plan.get("acceptance_criteria")),
                "repository_files": self._string_list(planner_plan.get("repository_files")),
                "estimated_dependencies": planner_plan.get("estimated_dependencies", []),
                "estimated_total_complexity": planner_plan.get(
                    "estimated_total_complexity",
                    "medium",
                ),
                "requires_architecture_review": bool(
                    planner_plan.get("requires_architecture_review", False)
                ),
            },
        )

    def _legacy_tasks(self, goal: str, ctx: dict[str, Any]) -> list[ExecutableTask]:
        if "implement" in goal.lower() or "create" in goal.lower():
            return self._plan_implementation(goal, ctx)
        if "fix" in goal.lower() or "debug" in goal.lower():
            return self._plan_debugging(goal, ctx)
        if "review" in goal.lower():
            return self._plan_review(goal, ctx)
        return self._plan_generic(goal, ctx)

    def _plan_implementation(self, goal: str, ctx: dict[str, Any]) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            name="retrieve_context",
            agent_type="retrieval",
            input_data={"task": goal, **ctx},
            tool_names=["context_builder"],
            can_run_parallel=False,
        )
        plan = ExecutableTask(
            name="create_plan",
            agent_type="planner",
            input_data={"task": goal, **ctx},
            dependencies=[retrieval.id],
            tool_names=["planner"],
        )
        code = ExecutableTask(
            name="write_code",
            agent_type="coder",
            input_data={"task": goal, **ctx},
            dependencies=[plan.id],
            tool_names=["coder"],
        )
        test = ExecutableTask(
            name="write_tests",
            agent_type="tester",
            input_data={"task": goal, **ctx},
            dependencies=[code.id],
            tool_names=["tester"],
            parallel_group="verify",
            can_run_parallel=True,
        )
        review = ExecutableTask(
            name="review_code",
            agent_type="reviewer",
            input_data={"task": goal, **ctx},
            dependencies=[code.id],
            tool_names=["reviewer"],
            parallel_group="verify",
            can_run_parallel=True,
        )
        logger.info("implementation_plan_created", tasks=5)
        return [retrieval, plan, code, test, review]

    def _plan_debugging(self, goal: str, ctx: dict[str, Any]) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            name="retrieve_context",
            agent_type="retrieval",
            input_data={"task": goal, **ctx},
            tool_names=["context_builder"],
        )
        debug = ExecutableTask(
            name="analyze_error",
            agent_type="debugger",
            input_data={"task": goal, **ctx},
            dependencies=[retrieval.id],
            tool_names=["debugger"],
        )
        fix = ExecutableTask(
            name="apply_fix",
            agent_type="coder",
            input_data={"task": goal, **ctx},
            dependencies=[debug.id],
            tool_names=["coder"],
        )
        test = ExecutableTask(
            name="verify_fix",
            agent_type="tester",
            input_data={"task": goal, **ctx},
            dependencies=[fix.id],
            tool_names=["tester"],
        )
        logger.info("debugging_plan_created", tasks=4)
        return [retrieval, debug, fix, test]

    def _plan_review(self, goal: str, ctx: dict[str, Any]) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            name="get_code",
            agent_type="retrieval",
            input_data={"task": goal, **ctx},
            tool_names=["context_builder"],
        )
        review = ExecutableTask(
            name="execute_review",
            agent_type="reviewer",
            input_data={"task": goal, **ctx},
            dependencies=[retrieval.id],
            tool_names=["reviewer"],
        )
        logger.info("review_plan_created", tasks=2)
        return [retrieval, review]

    def _plan_generic(self, goal: str, ctx: dict[str, Any]) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            name="retrieve_context",
            agent_type="retrieval",
            input_data={"task": goal, **ctx},
            tool_names=["context_builder"],
        )
        plan = ExecutableTask(
            name="create_plan",
            agent_type="planner",
            input_data={"task": goal, **ctx},
            dependencies=[retrieval.id],
            tool_names=["planner"],
        )
        exec_task = ExecutableTask(
            name="execute_task",
            agent_type="coder",
            input_data={"task": goal, **ctx},
            dependencies=[plan.id],
            tool_names=["coder"],
        )
        logger.info("generic_plan_created", tasks=3)
        return [retrieval, plan, exec_task]

    def _extract_planner_plan(self, ctx: Mapping[str, Any]) -> Mapping[str, Any] | None:
        direct = ctx.get("plan") or ctx.get("execution_plan") or ctx.get("task_plan")
        if isinstance(direct, Mapping):
            if "subtasks" in direct:
                return direct
            nested = direct.get("plan")
            if isinstance(nested, Mapping):
                return nested
        planner_output = ctx.get("planner_output")
        if isinstance(planner_output, Mapping):
            nested = planner_output.get("plan")
            if isinstance(nested, Mapping):
                return nested
            result = planner_output.get("result")
            if isinstance(result, Mapping) and isinstance(result.get("plan"), Mapping):
                return result["plan"]
        return None

    def _subtasks(self, planner_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        subtasks = planner_plan.get("subtasks", [])
        if not isinstance(subtasks, Sequence) or isinstance(subtasks, (str, bytes)):
            return []
        return [subtask for subtask in subtasks if isinstance(subtask, Mapping)]

    def _task_name(self, subtask: Mapping[str, Any]) -> str:
        title = str(subtask.get("title") or subtask.get("id") or "planned_task")
        return title.lower().replace(" ", "_")

    def _agent_type(self, subtask: Mapping[str, Any]) -> str:
        task_type = str(subtask.get("type", "implementation")).lower()
        return {
            "inspection": "research",
            "research": "research",
            "implementation": "coder",
            "test": "tester",
            "documentation": "coder",
            "review": "reviewer",
        }.get(task_type, "coder")

    def _input_data(
        self,
        goal: str,
        subtask: Mapping[str, Any],
        planner_plan: Mapping[str, Any],
        ctx: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task": goal,
            "plan": planner_plan,
            "subtask": dict(subtask),
            "files_to_inspect": self._string_list(subtask.get("files_to_inspect")),
            "likely_files": self._string_list(subtask.get("likely_files")),
            **{
                key: value
                for key, value in ctx.items()
                if key not in {"plan", "execution_plan", "task_plan", "planner_output"}
            },
        }

    def _tool_names(self, subtask: Mapping[str, Any]) -> list[str]:
        tools = subtask.get("tools", [])
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            return []
        names: list[str] = []
        for tool in tools:
            if isinstance(tool, Mapping) and tool.get("name"):
                names.append(str(tool["name"]))
        return names

    def _planner_parallel_groups(
        self,
        raw_groups: Any,
        task_by_step_id: Mapping[str, ExecutableTask],
    ) -> list[list[uuid.UUID]]:
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            return []
        groups: list[list[uuid.UUID]] = []
        for group in raw_groups:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                continue
            ids = [
                task_by_step_id[str(step_id)].id
                for step_id in group
                if str(step_id) in task_by_step_id
            ]
            if ids:
                groups.append(ids)
        return groups

    def _parallel_groups_from_tasks(
        self,
        tasks: list[ExecutableTask],
    ) -> list[list[uuid.UUID]]:
        remaining = list(tasks)
        completed: set[uuid.UUID] = set()
        groups: list[list[uuid.UUID]] = []
        while remaining:
            ready = [task for task in remaining if set(task.dependencies).issubset(completed)]
            if not ready:
                ready = [remaining[0]]
            groups.append([task.id for task in ready])
            completed.update(task.id for task in ready)
            remaining = [task for task in remaining if task not in ready]
        return groups

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        return [str(item) for item in value if item is not None]
