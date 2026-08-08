# kodiak/orchestration/planning/prioritizer.py
"""Task Prioritization and Parallel Group Calculation engines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from kodiak.db.models.task import TaskPriority
from .graph import DependencyGraph, DependencyGraphBuilder

logger = structlog.get_logger(__name__)

__all__ = ["TaskPrioritizer", "ParallelGroupCalculator"]


class TaskPrioritizer:
    """Calculates task execution priorities based on DAG topology and critical paths."""

    def __init__(self, high_fanout_threshold: int = 2) -> None:
        self.high_fanout_threshold = high_fanout_threshold

    def prioritize_tasks(
        self,
        tasks: Sequence[Any],
        graph: DependencyGraph | None = None,
        durations: dict[str, float] | None = None,
    ) -> dict[str, TaskPriority]:
        """Assign TaskPriority ratings to tasks based on critical path position and dependency fan-out.

        Args:
            tasks: Sequence of task objects or dicts.
            graph: Optional pre-built DependencyGraph.
            durations: Optional duration estimates per task.

        Returns:
            Dictionary mapping task_id -> TaskPriority.
        """
        dag = graph or DependencyGraphBuilder.build_from_tasks(tasks)
        crit_path_nodes, _ = dag.calculate_critical_path(durations)
        crit_set = set(crit_path_nodes)
        depths = dag.calculate_depths()

        priorities: dict[str, TaskPriority] = {}

        for task in tasks:
            tid = str(task.id if hasattr(task, "id") else task.get("id") if isinstance(task, dict) else str(task))
            fanout = len(dag.get_dependents(tid))
            depth = depths.get(tid, 0)

            if tid in crit_set and fanout >= self.high_fanout_threshold:
                priority = TaskPriority.CRITICAL
            elif tid in crit_set or fanout >= self.high_fanout_threshold:
                priority = TaskPriority.HIGH
            elif depth <= 1:
                priority = TaskPriority.MEDIUM
            else:
                priority = TaskPriority.LOW

            priorities[tid] = priority

        logger.debug("tasks_prioritized", total_tasks=len(tasks))
        return priorities


class ParallelGroupCalculator:
    """Calculates optimal parallel execution groups for DAG task workloads."""

    def __init__(self, max_parallel_width: int = 5) -> None:
        self.max_parallel_width = max_parallel_width

    def calculate_parallel_groups(
        self,
        tasks: Sequence[Any],
        graph: DependencyGraph | None = None,
    ) -> list[list[str]]:
        """Group tasks into executable parallel batches that satisfy dependency ordering.

        Args:
            tasks: Sequence of task objects.
            graph: Optional pre-built DependencyGraph.

        Returns:
            List of parallel task ID batches.
        """
        dag = graph or DependencyGraphBuilder.build_from_tasks(tasks)
        top_order = dag.topological_sort()

        tasks_by_id = {
            str(t.id if hasattr(t, "id") else t.get("id") if isinstance(t, dict) else str(t)): t
            for t in tasks
        }

        remaining = list(top_order)
        completed: set[str] = set()
        groups: list[list[str]] = []

        while remaining:
            # Find all tasks whose dependencies are satisfied
            ready: list[str] = []
            for tid in remaining:
                deps = set(dag.get_dependencies(tid))
                if deps.issubset(completed):
                    # Check for file write conflicts with already selected ready tasks
                    if not self._has_file_conflict(tid, ready, tasks_by_id):
                        ready.append(tid)

                    if len(ready) >= self.max_parallel_width:
                        break

            if not ready:
                ready = [remaining[0]]

            groups.append(ready)
            completed.update(ready)
            remaining = [tid for tid in remaining if tid not in ready]

        logger.debug("parallel_groups_calculated", group_count=len(groups))
        return groups

    def _has_file_conflict(
        self,
        target_id: str,
        current_batch: list[str],
        tasks_by_id: dict[str, Any],
    ) -> bool:
        """Check if target_id modifies the same likely_files as any task in current_batch."""
        target_task = tasks_by_id.get(target_id)
        if not target_task:
            return False

        target_files = set(self._get_files(target_task))
        if not target_files:
            return False

        for batch_id in current_batch:
            batch_task = tasks_by_id.get(batch_id)
            if not batch_task:
                continue
            batch_files = set(self._get_files(batch_task))
            if target_files & batch_files:
                return True

        return False

    @staticmethod
    def _get_files(task: Any) -> list[str]:
        if hasattr(task, "likely_files"):
            return list(getattr(task, "likely_files"))
        if isinstance(task, dict):
            return list(task.get("likely_files", []))
        return []
