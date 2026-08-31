"""
Planning pipeline for the Kodiak autonomous agent system.

This module implements the Planning subsystem capabilities:
- Goal decomposition (supporting hierarchical tasks)
- Task dependency management & cycle detection
- Task prioritization
- Task estimation & confidence scoring
- Plan validation & error reporting
- Plan optimization & transitive reduction
- Dynamic replanning preserving completed work
- Plan serialization and deserialization
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

import structlog
from pydantic import BaseModel, Field

from kodiak.orchestration.planning.exceptions import (
    DependencyCycleError,
    PlanValidationError,
    ReplanningError,
)
from kodiak.orchestration.task_planner import ExecutableTask, ExecutionPlan

logger = structlog.get_logger(__name__)


class ValidationResult(BaseModel):
    """Result of plan validation containing diagnostic information."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DependencyGraph:
    """Manages task dependency relationships, cycle detection, and ordering.

    Attributes:
        tasks: Dictionary mapping task ID to ExecutableTask.
        adj: Adjacency list mapping task ID to list of dependent task IDs.
        in_degree: Mapping of task ID to number of incoming dependencies.
    """

    def __init__(self, tasks: Sequence[ExecutableTask]) -> None:
        self.tasks: dict[uuid.UUID, ExecutableTask] = {task.id: task for task in tasks}
        self.adj: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
        self.in_degree: dict[uuid.UUID, int] = {task.id: 0 for task in tasks}
        self._build_graph()

    def _build_graph(self) -> None:
        """Build graph adjacency list and in-degree map from task dependencies."""
        for task in self.tasks.values():
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.adj[dep_id].append(task.id)
                    self.in_degree[task.id] += 1

    def detect_cycles(self) -> list[list[uuid.UUID]]:
        """Detect dependency cycles in the task graph using DFS.

        Returns:
            List of cycle paths (each cycle path is a list of task UUIDs).

        Raises:
            DependencyCycleError: If any cycle is detected.
        """
        visited: set[uuid.UUID] = set()
        rec_stack: set[uuid.UUID] = set()
        path: list[uuid.UUID] = []
        cycles: list[list[uuid.UUID]] = []

        def dfs(node: uuid.UUID) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for task_id in self.tasks:
            if task_id not in visited:
                dfs(task_id)

        if cycles:
            formatted_cycle = [str(node) for node in cycles[0]]
            msg = f"Dependency cycle detected in planning tasks: {' -> '.join(formatted_cycle)}"
            logger.error("dependency_cycle_detected", cycle=formatted_cycle)
            raise DependencyCycleError(msg, cycle_path=formatted_cycle)

        return cycles

    def topological_sort(self) -> list[uuid.UUID]:
        """Perform topological sorting on the task graph (Kahn's algorithm).

        Returns:
            Topologically sorted list of task UUIDs.

        Raises:
            DependencyCycleError: If a cycle prevents complete ordering.
        """
        in_degree = dict(self.in_degree)
        queue: deque[uuid.UUID] = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        order: list[uuid.UUID] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.tasks):
            missing = [str(t_id) for t_id in self.tasks if t_id not in order]
            raise DependencyCycleError(
                f"Dependency cycle detected involving tasks: {missing}",
                cycle_path=missing,
            )

        return order

    def compute_parallel_groups(self) -> list[list[uuid.UUID]]:
        """Compute parallel execution groups by topological levels.

        Returns:
            List of levels, where each level is a list of task UUIDs that can execute concurrently.
        """
        remaining = set(self.tasks.keys())
        completed: set[uuid.UUID] = set()
        groups: list[list[uuid.UUID]] = []

        while remaining:
            ready = [
                task_id
                for task_id in remaining
                if set(self.tasks[task_id].dependencies).issubset(completed)
            ]
            if not ready:
                # Fallback to single task to avoid infinite loop in edge cases
                first = next(iter(sorted(remaining, key=lambda u: str(u))))
                ready = [first]

            # Sort ready tasks deterministically by name / id
            ready.sort(key=lambda u: (self.tasks[u].name, str(u)))
            groups.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        return groups

    def get_downstream_reach(self, task_id: uuid.UUID) -> set[uuid.UUID]:
        """Compute all downstream task IDs dependent directly or indirectly on `task_id`.

        Args:
            task_id: The root task UUID to trace downstream.

        Returns:
            Set of downstream task UUIDs.
        """
        reached: set[uuid.UUID] = set()
        queue: deque[uuid.UUID] = deque([task_id])
        while queue:
            current = queue.popleft()
            for child in self.adj.get(current, []):
                if child not in reached:
                    reached.add(child)
                    queue.append(child)
        return reached


class TaskDecomposer:
    """Handles goal decomposition into atomic and hierarchical executable tasks."""

    def decompose(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> list[ExecutableTask]:
        """Decompose a user goal into executable tasks.

        Args:
            goal: The high-level goal description.
            context: Context bag optionally containing planner outputs or domain details.

        Returns:
            List of ExecutableTask instances, preserving any parent/child hierarchy.
        """
        ctx = context or {}
        planner_plan = self._extract_planner_plan(ctx)
        if planner_plan is not None:
            return self._decompose_from_planner_output(goal, planner_plan, ctx)

        return self._legacy_heuristic_decomposition(goal, ctx)

    def _extract_planner_plan(self, ctx: dict[str, Any]) -> dict[str, Any] | None:
        """Extract structured plan from context if available."""
        direct = ctx.get("plan") or ctx.get("execution_plan") or ctx.get("task_plan")
        if isinstance(direct, dict):
            if "subtasks" in direct:
                return direct
            nested = direct.get("plan")
            if isinstance(nested, dict):
                return nested

        planner_output = ctx.get("planner_output")
        if isinstance(planner_output, dict):
            nested = planner_output.get("plan")
            if isinstance(nested, dict):
                return nested
            result = planner_output.get("result")
            if isinstance(result, dict) and isinstance(result.get("plan"), dict):
                return result["plan"]
        return None

    def _decompose_from_planner_output(
        self,
        goal: str,
        planner_plan: dict[str, Any],
        ctx: dict[str, Any],
    ) -> list[ExecutableTask]:
        """Decompose LLM planner output into structured tasks with hierarchy support."""
        raw_subtasks = planner_plan.get("subtasks", [])
        if not isinstance(raw_subtasks, list):
            raw_subtasks = []

        step_id_to_task: dict[str, ExecutableTask] = {}
        subtask_id_to_step_id: dict[str, str] = {}

        # First pass: Instantiate tasks
        for subtask in raw_subtasks:
            if not isinstance(subtask, dict):
                continue
            step_id = str(subtask.get("id", uuid.uuid4().hex))
            task_name = str(subtask.get("title") or subtask.get("id") or "planned_task")
            task_name_clean = task_name.lower().replace(" ", "_")

            task_type = str(subtask.get("type", "implementation")).lower()
            agent_type = {
                "inspection": "research",
                "research": "research",
                "implementation": "coder",
                "test": "tester",
                "documentation": "coder",
                "review": "reviewer",
            }.get(task_type, "coder")

            files_to_inspect = self._to_string_list(subtask.get("files_to_inspect"))
            likely_files = self._to_string_list(subtask.get("likely_files"))
            if not files_to_inspect:
                files_to_inspect = list(likely_files)

            tools = subtask.get("tools", [])
            tool_names: list[str] = []
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict) and tool.get("name"):
                        tool_names.append(str(tool["name"]))

            task = ExecutableTask(
                name=task_name_clean,
                agent_type=agent_type,
                input_data={
                    "task": goal,
                    "plan": planner_plan,
                    "subtask": dict(subtask),
                    "files_to_inspect": files_to_inspect,
                    "likely_files": likely_files,
                },
                plan_step_id=step_id,
                tool_names=tool_names,
                files_to_inspect=files_to_inspect,
                parallel_group=(
                    str(subtask["parallel_group"])
                    if subtask.get("parallel_group") is not None
                    else None
                ),
                can_run_parallel=bool(subtask.get("can_run_parallel", False)),
                description=str(subtask.get("description", "")),
                estimated_complexity=str(subtask.get("complexity", "medium")),
            )

            step_id_to_task[step_id] = task
            subtask_id_to_step_id[step_id] = step_id

        # Second pass: Connect dependencies & hierarchy (parent/child)
        for subtask in raw_subtasks:
            if not isinstance(subtask, dict):
                continue
            step_id = str(subtask.get("id", ""))
            task = step_id_to_task.get(step_id)
            if task is None:
                continue

            depends_on_raw = self._to_string_list(subtask.get("depends_on"))
            task.dependencies = [
                step_id_to_task[dep_step].id
                for dep_step in depends_on_raw
                if dep_step in step_id_to_task
            ]

            parent_step = subtask.get("parent_id")
            if parent_step and str(parent_step) in step_id_to_task:
                parent_task = step_id_to_task[str(parent_step)]
                task.parent_id = parent_task.id
                if task.id not in parent_task.subtask_ids:
                    parent_task.subtask_ids.append(task.id)

        return list(step_id_to_task.values())

    def _legacy_heuristic_decomposition(
        self,
        goal: str,
        ctx: dict[str, Any],
    ) -> list[ExecutableTask]:
        """Heuristic decomposition for implementation, debugging, review, or generic goals."""
        goal_lower = goal.lower()
        if "implement" in goal_lower or "create" in goal_lower:
            retrieval = ExecutableTask(
                name="retrieve_context",
                agent_type="retrieval",
                input_data={"task": goal, **ctx},
                tool_names=["context_builder"],
                description="Retrieve repository context for implementation",
                estimated_complexity="low",
                priority="high",
            )
            plan_task = ExecutableTask(
                name="create_plan",
                agent_type="planner",
                input_data={"task": goal, **ctx},
                dependencies=[retrieval.id],
                tool_names=["planner"],
                description="Formulate detailed task plan",
                estimated_complexity="medium",
                priority="high",
            )
            code_task = ExecutableTask(
                name="write_code",
                agent_type="coder",
                input_data={"task": goal, **ctx},
                dependencies=[plan_task.id],
                tool_names=["coder"],
                description="Implement features and changes",
                estimated_complexity="high",
                priority="medium",
            )
            test_task = ExecutableTask(
                name="write_tests",
                agent_type="tester",
                input_data={"task": goal, **ctx},
                dependencies=[code_task.id],
                tool_names=["tester"],
                parallel_group="verify",
                can_run_parallel=True,
                description="Generate unit and integration tests",
                estimated_complexity="medium",
                priority="medium",
            )
            review_task = ExecutableTask(
                name="review_code",
                agent_type="reviewer",
                input_data={"task": goal, **ctx},
                dependencies=[code_task.id],
                tool_names=["reviewer"],
                parallel_group="verify",
                can_run_parallel=True,
                description="Perform static code review and validation",
                estimated_complexity="low",
                priority="low",
            )
            return [retrieval, plan_task, code_task, test_task, review_task]

        if "fix" in goal_lower or "debug" in goal_lower:
            retrieval = ExecutableTask(
                name="retrieve_context",
                agent_type="retrieval",
                input_data={"task": goal, **ctx},
                tool_names=["context_builder"],
                description="Gather error logs and context",
                estimated_complexity="low",
            )
            debug = ExecutableTask(
                name="analyze_error",
                agent_type="debugger",
                input_data={"task": goal, **ctx},
                dependencies=[retrieval.id],
                tool_names=["debugger"],
                description="Perform root cause analysis",
                estimated_complexity="high",
            )
            fix = ExecutableTask(
                name="apply_fix",
                agent_type="coder",
                input_data={"task": goal, **ctx},
                dependencies=[debug.id],
                tool_names=["coder"],
                description="Apply corrective code fix",
                estimated_complexity="medium",
            )
            test = ExecutableTask(
                name="verify_fix",
                agent_type="tester",
                input_data={"task": goal, **ctx},
                dependencies=[fix.id],
                tool_names=["tester"],
                description="Verify fix with automated tests",
                estimated_complexity="medium",
            )
            return [retrieval, debug, fix, test]

        # Generic default plan
        retrieval = ExecutableTask(
            name="retrieve_context",
            agent_type="retrieval",
            input_data={"task": goal, **ctx},
            tool_names=["context_builder"],
            description="Retrieve task context",
            estimated_complexity="low",
        )
        plan = ExecutableTask(
            name="create_plan",
            agent_type="planner",
            input_data={"task": goal, **ctx},
            dependencies=[retrieval.id],
            tool_names=["planner"],
            description="Create execution plan",
            estimated_complexity="medium",
        )
        exec_task = ExecutableTask(
            name="execute_task",
            agent_type="coder",
            input_data={"task": goal, **ctx},
            dependencies=[plan.id],
            tool_names=["coder"],
            description="Execute core task",
            estimated_complexity="medium",
        )
        return [retrieval, plan, exec_task]

    def _to_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None]


class TaskPrioritizer:
    """Calculates task priorities and orders tasks based on dependencies and critical paths."""

    _TYPE_WEIGHTS: dict[str, float] = {
        "retrieval": 20.0,
        "research": 18.0,
        "planner": 15.0,
        "debugger": 12.0,
        "coder": 10.0,
        "tester": 5.0,
        "reviewer": 2.0,
    }

    _PRIORITY_WEIGHTS: dict[str, float] = {
        "critical": 40.0,
        "high": 30.0,
        "medium": 20.0,
        "low": 10.0,
    }

    def prioritize(
        self,
        tasks: Sequence[ExecutableTask],
        graph: DependencyGraph,
    ) -> list[ExecutableTask]:
        """Calculate priority scores for tasks and sort execution order.

        Args:
            tasks: Sequence of ExecutableTask instances.
            graph: Built DependencyGraph for the tasks.

        Returns:
            List of ExecutableTask instances with updated priority scores.
        """
        task_list = list(tasks)
        for task in task_list:
            downstream_count = len(graph.get_downstream_reach(task.id))
            type_weight = self._TYPE_WEIGHTS.get(task.agent_type.lower(), 5.0)
            priority_weight = self._PRIORITY_WEIGHTS.get(str(task.priority).lower(), 20.0)

            # Unblocking downstream tasks increases priority significantly
            score = priority_weight + type_weight + (downstream_count * 15.0)
            task.priority_score = score

        return task_list


class TaskEstimator:
    """Estimates task complexity, execution requirements, duration, and confidence."""

    _COMPLEXITY_DURATION: dict[str, float] = {
        "low": 15.0,
        "medium": 45.0,
        "high": 120.0,
    }

    def estimate(
        self,
        tasks: Sequence[ExecutableTask],
    ) -> tuple[list[ExecutableTask], str, float]:
        """Estimate execution parameters for tasks and calculate plan totals.

        Args:
            tasks: Sequence of ExecutableTask instances.

        Returns:
            Tuple of (updated_tasks, estimated_total_complexity, estimated_total_duration_seconds).
        """
        task_list = list(tasks)
        total_duration = 0.0
        complexity_counts: dict[str, int] = defaultdict(int)

        for task in task_list:
            comp = task.estimated_complexity.lower()
            if comp not in self._COMPLEXITY_DURATION:
                comp = "medium"
                task.estimated_complexity = "medium"

            duration = self._COMPLEXITY_DURATION[comp]
            task.estimated_duration = duration
            total_duration += duration
            complexity_counts[comp] += 1

            # Estimate confidence: missing tools/inputs reduces confidence slightly
            confidence = 1.0
            if not task.tool_names:
                confidence -= 0.1
            if not task.files_to_inspect:
                confidence -= 0.05
            task.confidence_score = max(0.5, round(confidence, 2))

        # Total complexity heuristic
        if complexity_counts["high"] > 0 or len(task_list) > 6:
            total_complexity = "high"
        elif complexity_counts["medium"] > 1 or len(task_list) > 3:
            total_complexity = "medium"
        else:
            total_complexity = "low"

        return task_list, total_complexity, total_duration


class PlanValidator:
    """Validates plan consistency, dependency integrity, and execution feasibility."""

    def validate(
        self,
        goal: str,
        tasks: Sequence[ExecutableTask],
        execution_order: Sequence[uuid.UUID],
    ) -> ValidationResult:
        """Validate an execution plan for internal consistency.

        Args:
            goal: The overall goal string.
            tasks: List of tasks in the plan.
            execution_order: Ordered task UUIDs.

        Returns:
            ValidationResult with boolean validity status, errors, and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not goal or not goal.strip():
            errors.append("Goal description is empty.")

        if not tasks:
            errors.append("Plan contains no tasks.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        task_ids = {task.id for task in tasks}

        for task in tasks:
            if not task.name:
                errors.append(f"Task {task.id} is missing a name.")
            if not task.agent_type:
                errors.append(f"Task {task.name} ({task.id}) is missing an agent_type.")

            # Validate dependencies
            for dep in task.dependencies:
                if dep not in task_ids:
                    errors.append(
                        f"Task '{task.name}' references non-existent dependency task ID: {dep}"
                    )
                if dep == task.id:
                    errors.append(f"Task '{task.name}' cannot depend on itself.")

        # Validate cycle presence via DependencyGraph
        try:
            graph = DependencyGraph(tasks)
            graph.detect_cycles()
        except DependencyCycleError as exc:
            errors.append(f"Cycle validation error: {exc.message}")

        # Validate ordering completeness
        if len(execution_order) != len(tasks):
            warnings.append(
                "Execution order length "
                f"({len(execution_order)}) differs from total task count ({len(tasks)})."
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


class PlanOptimizer:
    """Optimizes task ordering, removes redundant dependencies, and constructs parallel groups."""

    def optimize(
        self,
        tasks: Sequence[ExecutableTask],
        graph: DependencyGraph,
    ) -> tuple[list[ExecutableTask], list[uuid.UUID], list[list[uuid.UUID]]]:
        """Optimize dependencies via transitive reduction and refine parallel execution groups.

        Args:
            tasks: Sequence of ExecutableTask instances.
            graph: Built DependencyGraph.

        Returns:
            Tuple of (optimized_tasks, execution_order, parallel_groups).
        """
        task_map = {task.id: task for task in tasks}

        # Transitive reduction: remove redundant direct dependencies
        for task in task_map.values():
            direct_deps = list(task.dependencies)
            redundant: set[uuid.UUID] = set()

            for dep_a in direct_deps:
                # Reachable downstream nodes from dep_a
                reach_a = graph.get_downstream_reach(dep_a)
                for dep_b in direct_deps:
                    if dep_a != dep_b and dep_b in reach_a:
                        # A direct dependency on dep_a is redundant for this task.
                        redundant.add(dep_a)

            if redundant:
                task.dependencies = [d for d in task.dependencies if d not in redundant]
                logger.debug(
                    "transitive_reduction_applied",
                    task_name=task.name,
                    removed_count=len(redundant),
                )

        # Re-build graph after reduction to compute fresh order and parallel groups
        reduced_graph = DependencyGraph(list(task_map.values()))
        execution_order = reduced_graph.topological_sort()

        # Sort tasks at each level of parallel groups by priority_score descending
        raw_parallel_groups = reduced_graph.compute_parallel_groups()
        parallel_groups: list[list[uuid.UUID]] = []
        for group in raw_parallel_groups:
            sorted_group = sorted(
                group,
                key=lambda u: (task_map[u].priority_score, task_map[u].name),
                reverse=True,
            )
            parallel_groups.append(sorted_group)

        return list(task_map.values()), execution_order, parallel_groups


class PlanReplanner:
    """Dynamic replanner that adapts plans upon task failures while preserving completed work."""

    def replan(
        self,
        current_plan: ExecutionPlan,
        execution_result: Any,
    ) -> ExecutionPlan:
        """Produce an updated ExecutionPlan upon failure or runtime updates.

        Args:
            current_plan: Existing ExecutionPlan.
            execution_result: Result object or dict indicating task outcomes/failures.

        Returns:
            An updated ExecutionPlan preserving completed tasks and replacing/repairing failed ones.

        Raises:
            ReplanningError: If replanning fails or no recovery strategy is found.
        """
        failed_task_id: str | None = None
        failure_error: str = "Unknown error"

        # Handle ExecutionResult or dict
        if hasattr(execution_result, "task_id") and hasattr(execution_result, "is_success"):
            if execution_result.is_success:
                logger.info("replanning_skipped_execution_succeeded")
                return current_plan
            failed_task_id = str(execution_result.task_id)
            if hasattr(execution_result, "error") and execution_result.error:
                failure_error = str(execution_result.error)
        elif isinstance(execution_result, dict):
            failed_task_id = str(
                execution_result.get("task_id") or execution_result.get("failed_task_id", "")
            )
            failure_error = str(
                execution_result.get("error") or execution_result.get("message", "Failure")
            )
            reflection = execution_result.get("reflection")
            if isinstance(reflection, dict) and reflection.get("root_cause"):
                failure_error = str(reflection["root_cause"])
            elif isinstance(execution_result.get("error"), dict):
                nested = execution_result["error"]
                reflection = nested.get("reflection")
                if isinstance(reflection, dict) and reflection.get("root_cause"):
                    failure_error = str(reflection["root_cause"])

        task_map = {str(t.id): t for t in current_plan.tasks}

        # Find target failed task
        target_task: ExecutableTask | None = None
        if failed_task_id and failed_task_id in task_map:
            target_task = task_map[failed_task_id]
        else:
            # Find first failed task by status attribute
            for t in current_plan.tasks:
                if t.status in ("failed", "cancelled"):
                    target_task = t
                    break

        if target_task is None:
            msg = (
                "Replanning requested but no failed task was identified"
                " in execution result or plan."
            )
            logger.warning("replanning_no_failed_task", goal=current_plan.goal)
            raise ReplanningError(msg)

        logger.info(
            "dynamic_replanning_triggered",
            failed_task=target_task.name,
            failed_task_id=str(target_task.id),
            error=failure_error,
        )

        # Build list of preserved tasks (completed ones remain untouched)
        new_tasks: list[ExecutableTask] = []
        for t in current_plan.tasks:
            if t.id == target_task.id:
                continue
            # Keep existing tasks
            new_tasks.append(t)

        # Create replacement recovery tasks: Debug -> Fix
        debug_task = ExecutableTask(
            name=f"debug_{target_task.name}",
            agent_type="debugger",
            input_data={
                "task": current_plan.goal,
                "failed_task_name": target_task.name,
                "error_details": failure_error,
                "original_input": target_task.input_data,
            },
            dependencies=list(target_task.dependencies),
            tool_names=["debugger"],
            description=f"Analyze failure in task: {target_task.name}",
            estimated_complexity="high",
            priority="critical",
            parent_id=target_task.parent_id,
        )

        recovery_fix_task = ExecutableTask(
            name=f"repair_{target_task.name}",
            agent_type="coder",
            input_data={
                "task": current_plan.goal,
                "failed_task_name": target_task.name,
                "original_input": target_task.input_data,
            },
            dependencies=[debug_task.id],
            tool_names=["coder"],
            description=f"Apply recovery fix for failed task: {target_task.name}",
            estimated_complexity="medium",
            priority="high",
            parent_id=target_task.parent_id,
        )

        new_tasks.extend([debug_task, recovery_fix_task])

        # Rewire downstream dependencies that depended on the failed task
        for t in new_tasks:
            if target_task.id in t.dependencies:
                t.dependencies.remove(target_task.id)
                t.dependencies.append(recovery_fix_task.id)

        # Re-run pipeline over new task set
        graph = DependencyGraph(new_tasks)
        graph.detect_cycles()

        prioritizer = TaskPrioritizer()
        prioritizer.prioritize(new_tasks, graph)

        estimator = TaskEstimator()
        updated_tasks, total_comp, total_dur = estimator.estimate(new_tasks)

        optimizer = PlanOptimizer()
        final_tasks, execution_order, parallel_groups = optimizer.optimize(updated_tasks, graph)

        validator = PlanValidator()
        val_res = validator.validate(current_plan.goal, final_tasks, execution_order)

        updated_metadata = dict(current_plan.metadata)
        updated_metadata["replanned"] = True
        updated_metadata["replan_count"] = int(updated_metadata.get("replan_count", 0)) + 1
        updated_metadata["last_failed_task"] = target_task.name

        return ExecutionPlan(
            goal=current_plan.goal,
            tasks=final_tasks,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            metadata=updated_metadata,
            acceptance_criteria=current_plan.acceptance_criteria,
            requires_architecture_review=current_plan.requires_architecture_review,
            estimated_total_complexity=total_comp,
            estimated_total_duration=total_dur,
            validation_result=val_res.to_dict(),
        )


class PlanSerializer:
    """Safe serialization and deserialization of ExecutionPlan objects to/from dict and JSON."""

    @staticmethod
    def to_dict(plan: ExecutionPlan) -> dict[str, Any]:
        """Serialize an ExecutionPlan to a JSON-serializable dictionary.

        Args:
            plan: The ExecutionPlan to serialize.

        Returns:
            Dictionary containing serialized plan structure.
        """
        return plan.machine_readable()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExecutionPlan:
        """Deserialize a dictionary into an ExecutionPlan object.

        Args:
            data: Dictionary representation of a plan.

        Returns:
            Deserialized ExecutionPlan instance.
        """
        goal = str(data.get("goal", ""))
        metadata = dict(data.get("metadata", {}))
        acceptance_criteria = [str(ac) for ac in data.get("acceptance_criteria", [])]
        requires_architecture_review = bool(data.get("requires_architecture_review", False))
        estimated_total_complexity = str(data.get("estimated_total_complexity", "medium"))
        estimated_total_duration = float(data.get("estimated_total_duration", 0.0))
        validation_result = data.get("validation_result")

        raw_tasks = data.get("tasks", [])
        tasks: list[ExecutableTask] = []
        id_map: dict[str, uuid.UUID] = {}

        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue

            raw_id = raw.get("id")
            task_uuid = uuid.UUID(str(raw_id)) if raw_id else uuid.uuid4()
            id_map[str(raw_id)] = task_uuid

            parent_id_raw = raw.get("parent_id")
            parent_uuid = uuid.UUID(str(parent_id_raw)) if parent_id_raw else None

            subtask_ids_raw = raw.get("subtask_ids", [])
            subtask_uuids = [uuid.UUID(str(sid)) for sid in subtask_ids_raw if sid is not None]

            dep_uuids = [
                uuid.UUID(str(dep)) for dep in raw.get("dependencies", []) if dep is not None
            ]

            task = ExecutableTask(
                id=task_uuid,
                name=str(raw.get("name", "unnamed_task")),
                agent_type=str(raw.get("agent_type", "coder")),
                input_data=dict(raw.get("input_data", {})),
                dependencies=dep_uuids,
                plan_step_id=raw.get("plan_step_id"),
                tool_names=[str(t) for t in raw.get("tool_names", [])],
                files_to_inspect=[str(f) for f in raw.get("files_to_inspect", [])],
                parallel_group=raw.get("parallel_group"),
                can_run_parallel=bool(raw.get("can_run_parallel", False)),
                description=str(raw.get("description", "")),
                priority=str(raw.get("priority", "medium")),
                priority_score=float(raw.get("priority_score", 0.0)),
                estimated_complexity=str(raw.get("estimated_complexity", "medium")),
                estimated_duration=float(raw.get("estimated_duration", 0.0)),
                confidence_score=float(raw.get("confidence_score", 1.0)),
                parent_id=parent_uuid,
                subtask_ids=subtask_uuids,
                status=str(raw.get("status", "pending")),
            )
            tasks.append(task)

        raw_order = data.get("execution_order", [])
        execution_order = [uuid.UUID(str(tid)) for tid in raw_order if tid is not None]

        raw_pgroups = data.get("parallel_groups", [])
        parallel_groups: list[list[uuid.UUID]] = []
        if isinstance(raw_pgroups, list):
            for group in raw_pgroups:
                if isinstance(group, list):
                    parallel_groups.append(
                        [uuid.UUID(str(tid)) for tid in group if tid is not None]
                    )

        return ExecutionPlan(
            goal=goal,
            tasks=tasks,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            metadata=metadata,
            acceptance_criteria=acceptance_criteria,
            requires_architecture_review=requires_architecture_review,
            estimated_total_complexity=estimated_total_complexity,
            estimated_total_duration=estimated_total_duration,
            validation_result=validation_result,
        )

    @classmethod
    def to_json(cls, plan: ExecutionPlan) -> str:
        """Serialize an ExecutionPlan to a JSON string.

        Args:
            plan: The ExecutionPlan instance.

        Returns:
            JSON string representation of the plan.
        """
        return json.dumps(cls.to_dict(plan), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ExecutionPlan:
        """Deserialize a JSON string into an ExecutionPlan instance.

        Args:
            json_str: JSON string representation of a plan.

        Returns:
            Deserialized ExecutionPlan instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class PlanningPipeline:
    """Main entry point and orchestrator for the Kodiak Planning subsystem.

    Coordinates goal decomposition, dependency cycle detection & ordering,
    task prioritization, estimation, validation, optimization, and dynamic replanning.
    """

    def __init__(
        self,
        decomposer: TaskDecomposer | None = None,
        prioritizer: TaskPrioritizer | None = None,
        estimator: TaskEstimator | None = None,
        validator: PlanValidator | None = None,
        optimizer: PlanOptimizer | None = None,
        replanner: PlanReplanner | None = None,
    ) -> None:
        self.decomposer = decomposer or TaskDecomposer()
        self.prioritizer = prioritizer or TaskPrioritizer()
        self.estimator = estimator or TaskEstimator()
        self.validator = validator or PlanValidator()
        self.optimizer = optimizer or PlanOptimizer()
        self.replanner = replanner or PlanReplanner()

    async def plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Generate a complete, validated, optimized ExecutionPlan for a user goal.

        Args:
            goal: User goal or task prompt.
            context: Optional context dictionary.

        Returns:
            Validated and optimized ExecutionPlan.

        Raises:
            DependencyCycleError: If tasks contain dependency cycles.
            PlanValidationError: If the generated plan fails strict validation.
        """
        ctx = context or {}
        logger.info("planning_pipeline_started", goal=goal)

        # Stage 1: Decomposition
        tasks = self.decomposer.decompose(goal, ctx)

        # Stage 2: Dependencies & Cycle Detection
        graph = DependencyGraph(tasks)
        graph.detect_cycles()

        # Stage 3: Prioritization
        self.prioritizer.prioritize(tasks, graph)

        # Stage 4: Estimation
        estimated_tasks, total_complexity, total_duration = self.estimator.estimate(tasks)

        # Stage 5: Optimization & Ordering
        final_tasks, execution_order, parallel_groups = self.optimizer.optimize(
            estimated_tasks, graph
        )

        # Stage 6: Validation
        validation_res = self.validator.validate(goal, final_tasks, execution_order)
        if not validation_res.is_valid:
            logger.error("plan_validation_failed", errors=validation_res.errors)
            raise PlanValidationError(
                f"Generated plan failed validation: {'; '.join(validation_res.errors)}",
                errors=validation_res.errors,
                warnings=validation_res.warnings,
            )

        # Assemble plan metadata
        metadata = {
            "source": "planning_pipeline",
            "task_count": len(final_tasks),
            "parallel_group_count": len(parallel_groups),
        }
        if "planner_output" in ctx:
            metadata["source"] = "planner_agent"

        plan = ExecutionPlan(
            goal=goal,
            tasks=final_tasks,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            metadata=metadata,
            acceptance_criteria=ctx.get("acceptance_criteria", []),
            requires_architecture_review=bool(ctx.get("requires_architecture_review", False)),
            estimated_total_complexity=total_complexity,
            estimated_total_duration=total_duration,
            validation_result=validation_res.to_dict(),
        )

        logger.info(
            "planning_pipeline_completed",
            tasks=len(final_tasks),
            complexity=total_complexity,
        )
        return plan

    async def replan(
        self,
        current_plan: ExecutionPlan,
        execution_result: Any,
    ) -> ExecutionPlan:
        """Dynamic replanning entry point.

        Args:
            current_plan: The plan currently being executed.
            execution_result: Failure result or outcome structure.

        Returns:
            An updated, valid ExecutionPlan.
        """
        return self.replanner.replan(current_plan, execution_result)
