# kodiak/orchestration/planning/graph.py
"""Dependency Graph representation and DAG analysis algorithms for planning."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

import structlog

from .models import TaskDependencyType

logger = structlog.get_logger(__name__)

__all__ = ["DependencyEdge", "DependencyGraph", "DependencyGraphBuilder"]


class DependencyEdge:
    """Directed edge in a task dependency graph."""

    def __init__(
        self,
        from_node: str,
        to_node: str,
        dependency_type: TaskDependencyType = TaskDependencyType.HARD,
        reason: str = "",
    ) -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.dependency_type = dependency_type
        self.reason = reason


class DependencyGraph:
    """Directed Acyclic Graph (DAG) for managing task dependency networks."""

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        # outgoing edges: node_id -> list of child node_ids that depend on node_id
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        # incoming edges: node_id -> list of parent node_ids that node_id depends on
        self._incoming: dict[str, list[str]] = defaultdict(list)
        self._edges: list[DependencyEdge] = []

    def add_node(self, node_id: str) -> None:
        """Register a node in the graph."""
        self._nodes.add(node_id)
        if node_id not in self._outgoing:
            self._outgoing[node_id] = []
        if node_id not in self._incoming:
            self._incoming[node_id] = []

    def add_dependency(
        self,
        from_node: str,
        to_node: str,
        dependency_type: TaskDependencyType = TaskDependencyType.HARD,
        reason: str = "",
    ) -> None:
        """Add a directed dependency edge: to_node depends on from_node (from_node -> to_node)."""
        self.add_node(from_node)
        self.add_node(to_node)

        if to_node not in self._outgoing[from_node]:
            self._outgoing[from_node].append(to_node)
        if from_node not in self._incoming[to_node]:
            self._incoming[to_node].append(from_node)

        edge = DependencyEdge(
            from_node=from_node,
            to_node=to_node,
            dependency_type=dependency_type,
            reason=reason,
        )
        self._edges.append(edge)

    @property
    def nodes(self) -> set[str]:
        """Get set of all node IDs."""
        return set(self._nodes)

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get parent node IDs that node_id depends on."""
        return list(self._incoming.get(node_id, []))

    def get_dependents(self, node_id: str) -> list[str]:
        """Get child node IDs that depend on node_id."""
        return list(self._outgoing.get(node_id, []))

    def has_cycle(self) -> tuple[bool, list[str]]:
        """Detect cycles using Depth First Search (DFS).

        Returns:
            Tuple of (has_cycle: bool, cycle_path: list[str]).
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._outgoing.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    path.append(neighbor)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in sorted(self._nodes):
            if node not in visited:
                if dfs(node):
                    return True, path

        return False, []

    def topological_sort(self) -> list[str]:
        """Perform topological sort (Kahn's Algorithm).

        Returns:
            List of node IDs in valid topological execution order.

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        in_degree = {node: len(self._incoming[node]) for node in self._nodes}
        queue = deque(sorted([node for node, degree in in_degree.items() if degree == 0]))
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for neighbor in sorted(self._outgoing[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            has_cyc, cyc_path = self.has_cycle()
            path_str = " -> ".join(cyc_path) if cyc_path else "unknown"
            raise ValueError(f"Cycle detected in dependency graph: {path_str}")

        return order

    def calculate_depths(self) -> dict[str, int]:
        """Calculate maximum dependency depth (distance from root nodes) for each node."""
        depths: dict[str, int] = {}
        top_order = self.topological_sort()

        for node in top_order:
            parents = self._incoming[node]
            if not parents:
                depths[node] = 0
            else:
                depths[node] = max(depths[p] for p in parents) + 1

        return depths

    def calculate_critical_path(
        self, node_durations: dict[str, float] | None = None
    ) -> tuple[list[str], float]:
        """Compute the critical path (longest execution path duration and sequence).

        Args:
            node_durations: Optional mapping of node_id to estimated duration seconds.

        Returns:
            Tuple of (critical_path_nodes: list[str], total_duration_seconds: float).
        """
        durations = node_durations or {}
        top_order = self.topological_sort()

        earliest_start: dict[str, float] = {node: 0.0 for node in self._nodes}
        earliest_finish: dict[str, float] = {}
        predecessor: dict[str, str | None] = {node: None for node in self._nodes}

        for node in top_order:
            node_dur = durations.get(node, 1.0)
            parents = self._incoming[node]
            if parents:
                max_parent_finish = 0.0
                best_parent: str | None = None
                for parent in parents:
                    p_finish = earliest_finish[parent]
                    if p_finish > max_parent_finish:
                        max_parent_finish = p_finish
                        best_parent = parent
                earliest_start[node] = max_parent_finish
                predecessor[node] = best_parent

            earliest_finish[node] = earliest_start[node] + node_dur

        if not earliest_finish:
            return [], 0.0

        end_node = max(earliest_finish, key=earliest_finish.get)  # type: ignore[arg-type]
        max_duration = earliest_finish[end_node]

        path: list[str] = []
        curr: str | None = end_node
        while curr is not None:
            path.append(curr)
            curr = predecessor[curr]

        path.reverse()
        return path, max_duration

    def transitive_reduction(self) -> DependencyGraph:
        """Create a transitively reduced DAG by removing redundant direct dependency edges."""
        reduced = DependencyGraph()
        for node in self._nodes:
            reduced.add_node(node)

        top_order = self.topological_sort()
        # Compute reachability
        reachable: dict[str, set[str]] = {node: set() for node in self._nodes}

        for u in reversed(top_order):
            for v in self._outgoing[u]:
                reachable[u].add(v)
                reachable[u].update(reachable[v])

        for edge in self._edges:
            u, v = edge.from_node, edge.to_node
            # If v is reachable from u via intermediate node w (w != v),
            # edge (u, v) is redundant
            is_redundant = False
            for w in self._outgoing[u]:
                if w != v and v in reachable[w]:
                    is_redundant = True
                    break

            if not is_redundant:
                reduced.add_dependency(
                    from_node=u,
                    to_node=v,
                    dependency_type=edge.dependency_type,
                    reason=edge.reason,
                )

        return reduced


class DependencyGraphBuilder:
    """Factory helper for building DependencyGraph instances from various task inputs."""

    @staticmethod
    def build_from_tasks(tasks: Sequence[Any]) -> DependencyGraph:
        """Build a DependencyGraph from a list of tasks.

        Supports ExecutableTask, HierarchicalTaskNode, SubTask, TaskPlan, or dicts.
        """
        graph = DependencyGraph()
        task_ids: set[str] = set()

        for t in tasks:
            tid = str(t.id if hasattr(t, "id") else t.get("id") if isinstance(t, dict) else str(t))
            graph.add_node(tid)
            task_ids.add(tid)

        for t in tasks:
            tid = str(t.id if hasattr(t, "id") else t.get("id") if isinstance(t, dict) else str(t))

            deps: list[str] = []
            if hasattr(t, "dependencies"):
                deps = [str(d) for d in t.dependencies]
            elif hasattr(t, "depends_on"):
                deps = [str(d) for d in t.depends_on]
            elif isinstance(t, dict):
                deps = [str(d) for d in t.get("dependencies", t.get("depends_on", []))]

            for dep in deps:
                if dep in task_ids and dep != tid:
                    graph.add_dependency(from_node=dep, to_node=tid)

        logger.debug("dependency_graph_built", nodes=len(graph.nodes))
        return graph
