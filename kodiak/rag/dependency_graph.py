"""
In-memory dependency graph for Kodiak V3 repository indexes.

The graph in this module consumes :class:`kodiak.rag.repository_index.RepositoryIndex`
instances produced by :class:`kodiak.rag.repository_index.RepositoryIndexer`. It never
rescans source files, executes repository code, calls an LLM, computes embeddings,
performs semantic search, modifies repository files, or accesses a database.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from kodiak.rag.repository_index import ImportInfo, ModuleInfo, RepositoryIndex

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DependencyEdge:
    """A directed relationship from one indexed module to another."""

    source: str
    target: str
    kind: str
    line: int | None = None
    symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this edge."""
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "line": self.line,
            "symbol": self.symbol,
        }


@dataclass
class DependencyNode:
    """A repository module and the relationships discovered for it."""

    module_name: str
    path: Path
    relative_path: Path
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    parent_classes: set[str] = field(default_factory=set)
    inherited_classes: set[str] = field(default_factory=set)
    cross_module_references: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this node."""
        return {
            "module_name": self.module_name,
            "path": str(self.path),
            "relative_path": self.relative_path.as_posix(),
            "dependencies": sorted(self.dependencies),
            "dependents": sorted(self.dependents),
            "imports": sorted(self.imports),
            "functions": sorted(self.functions),
            "classes": sorted(self.classes),
            "parent_classes": sorted(self.parent_classes),
            "inherited_classes": sorted(self.inherited_classes),
            "cross_module_references": sorted(self.cross_module_references),
        }


@dataclass(frozen=True)
class ClassRelationship:
    """A resolved inheritance relationship between indexed classes."""

    child_class: str
    child_module: str
    parent_class: str
    parent_module: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation of this relationship."""
        return {
            "child_class": self.child_class,
            "child_module": self.child_module,
            "parent_class": self.parent_class,
            "parent_module": self.parent_module,
        }


class DependencyGraph:
    """
    Build and query module-level relationships from a repository index.

    Edges are directed from a module to the internal module it depends on. For
    example, if ``kodiak.api.main`` imports ``kodiak.config.tracing``, then
    ``kodiak.api.main`` has a dependency on ``kodiak.config.tracing`` and
    ``kodiak.config.tracing`` has ``kodiak.api.main`` as a dependent.
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._nodes: dict[str, DependencyNode] = {}
        self._edges: set[DependencyEdge] = set()
        self._module_names: set[str] = set()
        self._class_to_module: dict[str, str] = {}
        self._pending_class_relationships: list[ClassRelationship] = []
        self._class_relationships: tuple[ClassRelationship, ...] = ()

    @property
    def nodes(self) -> dict[str, DependencyNode]:
        """Return all graph nodes keyed by module name."""
        return dict(self._nodes)

    @property
    def edges(self) -> tuple[DependencyEdge, ...]:
        """Return all dependency edges in deterministic order."""
        return tuple(
            sorted(
                self._edges,
                key=lambda edge: (
                    edge.source,
                    edge.target,
                    edge.kind,
                    edge.line or -1,
                    edge.symbol or "",
                ),
            )
        )

    @property
    def class_relationships(self) -> tuple[ClassRelationship, ...]:
        """Return resolved parent-child class relationships."""
        return self._class_relationships

    def build_graph(self, repository_index: RepositoryIndex) -> DependencyGraph:
        """
        Populate this graph from a RepositoryIndexer result.

        Args:
            repository_index: Structural repository index created by
                :class:`RepositoryIndexer`.

        Returns:
            This graph instance for fluent use.
        """
        self._reset()
        self._module_names = {module.module_name for module in repository_index.modules}
        self._index_classes(repository_index.modules)

        for module in repository_index.modules:
            self._add_module(module)

        for module in repository_index.modules:
            self._add_import_dependencies(module)
            self._add_inheritance_dependencies(module)

        self._class_relationships = tuple(
            sorted(
                self._collect_class_relationships(),
                key=lambda item: (
                    item.parent_module,
                    item.parent_class,
                    item.child_module,
                    item.child_class,
                ),
            )
        )

        logger.info(
            "dependency_graph_built",
            modules=len(self._nodes),
            edges=len(self._edges),
            cycles=len(self.detect_cycles()),
            orphans=len(self.get_orphan_modules()),
        )
        return self

    @classmethod
    def from_index(cls, repository_index: RepositoryIndex) -> DependencyGraph:
        """
        Create and build a dependency graph from an existing repository index.

        This convenience constructor intentionally accepts an index, not a
        repository path, so callers do not accidentally trigger a second scan.
        """
        return cls().build_graph(repository_index)

    def get_dependencies(self, module: str) -> set[str]:
        """Return internal modules directly imported or inherited by ``module``."""
        node = self._nodes.get(module)
        return set(node.dependencies) if node else set()

    def get_dependents(self, module: str) -> set[str]:
        """Return internal modules that directly depend on ``module``."""
        node = self._nodes.get(module)
        return set(node.dependents) if node else set()

    def get_all_dependencies(self, module: str) -> set[str]:
        """Return the transitive dependency closure for ``module``."""
        visited: set[str] = set()
        self._visit_dependencies(module, visited)
        visited.discard(module)
        return visited

    def get_all_dependents(self, module: str) -> set[str]:
        """Return the transitive dependent closure for ``module``."""
        visited: set[str] = set()
        self._visit_dependents(module, visited)
        visited.discard(module)
        return visited

    def get_function_definitions(self, module: str) -> set[str]:
        """Return function and method qualified names defined in ``module``."""
        node = self._nodes.get(module)
        return set(node.functions) if node else set()

    def get_parent_classes(self, module: str) -> set[str]:
        """Return resolved parent classes referenced by classes in ``module``."""
        node = self._nodes.get(module)
        return set(node.parent_classes) if node else set()

    def get_inherited_classes(self, module: str) -> set[str]:
        """Return classes that inherit from classes defined in ``module``."""
        node = self._nodes.get(module)
        return set(node.inherited_classes) if node else set()

    def get_cross_module_references(self, module: str) -> set[str]:
        """Return resolved internal module references discovered for ``module``."""
        node = self._nodes.get(module)
        return set(node.cross_module_references) if node else set()

    def detect_cycles(self) -> list[list[str]]:
        """Return circular dependency paths, each closed by repeating the start."""
        visited: set[str] = set()
        active: set[str] = set()
        path: list[str] = []
        cycles: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        for module in sorted(self._nodes):
            if module not in visited:
                self._detect_cycles(module, visited, active, path, cycles, seen)

        return cycles

    def get_orphan_modules(self) -> set[str]:
        """Return modules with neither internal dependencies nor dependents."""
        return {
            name
            for name, node in self._nodes.items()
            if not node.dependencies and not node.dependents
        }

    def get_entry_point_modules(self) -> set[str]:
        """
        Return graph roots: modules not imported by any other indexed module.

        These modules are useful starting points for repository traversal because
        no other internal module depends on them. Orphan modules are included.
        """
        return {name for name, node in self._nodes.items() if not node.dependents}

    def topological_order(self) -> list[str]:
        """
        Return modules in dependency-first topological order.

        Raises:
            ValueError: If the graph contains one or more circular dependencies.
        """
        cycles = self.detect_cycles()
        if cycles:
            raise ValueError(f"Cannot topologically sort graph with cycles: {cycles}")

        dependents_by_module: dict[str, set[str]] = defaultdict(set)
        incoming_count: dict[str, int] = {module: 0 for module in self._nodes}

        for module, node in self._nodes.items():
            incoming_count[module] = len(node.dependencies)
            for dependency in node.dependencies:
                dependents_by_module[dependency].add(module)

        ready: deque[str] = deque(
            sorted(module for module, count in incoming_count.items() if count == 0)
        )
        ordered: list[str] = []

        while ready:
            module = ready.popleft()
            ordered.append(module)
            for dependent in sorted(dependents_by_module[module]):
                incoming_count[dependent] -= 1
                if incoming_count[dependent] == 0:
                    ready.append(dependent)

        return ordered

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this graph."""
        return {
            "nodes": {name: node.to_dict() for name, node in sorted(self._nodes.items())},
            "edges": [edge.to_dict() for edge in self.edges],
            "class_relationships": [
                relationship.to_dict() for relationship in self._class_relationships
            ],
            "cycles": self.detect_cycles(),
            "orphan_modules": sorted(self.get_orphan_modules()),
            "entry_point_modules": sorted(self.get_entry_point_modules()),
        }

    def _reset(self) -> None:
        self._nodes = {}
        self._edges = set()
        self._module_names = set()
        self._class_to_module = {}
        self._pending_class_relationships = []
        self._class_relationships = ()

    def _add_module(self, module: ModuleInfo) -> None:
        self._nodes[module.module_name] = DependencyNode(
            module_name=module.module_name,
            path=module.path,
            relative_path=module.relative_path,
            imports={self._format_import(import_info) for import_info in module.imports},
            functions={function.qualname for function in module.functions},
            classes={class_info.qualname for class_info in module.classes},
        )

    def _add_import_dependencies(self, module: ModuleInfo) -> None:
        for import_info in module.imports:
            for dependency in self._resolve_import(module.module_name, import_info):
                self._add_edge(
                    source=module.module_name,
                    target=dependency,
                    kind="import",
                    line=import_info.line,
                    symbol=self._format_import(import_info),
                )

    def _add_inheritance_dependencies(self, module: ModuleInfo) -> None:
        for class_info in module.classes:
            for base in class_info.bases:
                resolved = self._resolve_class_reference(module.module_name, base)
                if resolved is None:
                    continue

                parent_module, parent_class = resolved
                child_class = f"{module.module_name}.{class_info.qualname}"
                self._nodes[module.module_name].parent_classes.add(parent_class)
                self._nodes[parent_module].inherited_classes.add(child_class)
                self._pending_class_relationships.append(
                    ClassRelationship(
                        child_class=child_class,
                        child_module=module.module_name,
                        parent_class=parent_class,
                        parent_module=parent_module,
                    )
                )
                if parent_module != module.module_name:
                    self._add_edge(
                        source=module.module_name,
                        target=parent_module,
                        kind="inheritance",
                        line=class_info.line,
                        symbol=f"{class_info.name} -> {parent_class}",
                    )

    def _add_edge(
        self,
        *,
        source: str,
        target: str,
        kind: str,
        line: int | None = None,
        symbol: str | None = None,
    ) -> None:
        if source == target or source not in self._nodes or target not in self._nodes:
            return

        self._nodes[source].dependencies.add(target)
        self._nodes[source].cross_module_references.add(target)
        self._nodes[target].dependents.add(source)
        self._edges.add(
            DependencyEdge(source=source, target=target, kind=kind, line=line, symbol=symbol)
        )

    def _index_classes(self, modules: tuple[ModuleInfo, ...]) -> None:
        for module in modules:
            for class_info in module.classes:
                qualified = f"{module.module_name}.{class_info.qualname}"
                self._class_to_module[qualified] = module.module_name
                self._class_to_module[class_info.qualname] = module.module_name
                self._class_to_module[class_info.name] = module.module_name

    def _collect_class_relationships(self) -> list[ClassRelationship]:
        return self._pending_class_relationships

    def _resolve_import(self, current_module: str, import_info: ImportInfo) -> set[str]:
        candidates: set[str] = set()
        if import_info.kind == "import":
            candidates.update(self._clean_alias(name) for name in import_info.names)
        else:
            base_module = self._resolve_from_module(current_module, import_info)
            candidates.add(base_module)
            candidates.update(
                f"{base_module}.{self._clean_alias(name)}"
                if base_module
                else self._clean_alias(name)
                for name in import_info.names
            )

        return {
            resolved
            for candidate in candidates
            if candidate
            for resolved in [self._resolve_module_reference(candidate)]
            if resolved is not None and resolved != current_module
        }

    def _resolve_from_module(self, current_module: str, import_info: ImportInfo) -> str:
        if import_info.level <= 0:
            return import_info.module

        current_parts = current_module.split(".")
        package_parts = current_parts[:-1]
        keep_count = max(len(package_parts) - import_info.level + 1, 0)
        relative_parts = package_parts[:keep_count]
        if import_info.module:
            relative_parts.extend(import_info.module.split("."))
        return ".".join(part for part in relative_parts if part)

    def _resolve_module_reference(self, reference: str) -> str | None:
        normalized = self._strip_reference_suffix(reference)
        if normalized in self._module_names:
            return normalized

        parts = normalized.split(".")
        while len(parts) > 1:
            parts.pop()
            candidate = ".".join(parts)
            if candidate in self._module_names:
                return candidate
        return None

    def _resolve_class_reference(self, current_module: str, reference: str) -> tuple[str, str] | None:
        normalized = self._strip_reference_suffix(reference)
        candidates = (
            normalized,
            f"{current_module}.{normalized}",
            normalized.split(".")[-1],
        )

        for candidate in candidates:
            module_name = self._class_to_module.get(candidate)
            if module_name is not None:
                qualified_class = candidate if "." in candidate else f"{module_name}.{candidate}"
                return module_name, qualified_class

        module_name = self._resolve_module_reference(normalized)
        if module_name is not None:
            class_name = normalized.split(".")[-1]
            return module_name, f"{module_name}.{class_name}"
        return None

    def _visit_dependencies(self, module: str, visited: set[str]) -> None:
        if module in visited:
            return
        visited.add(module)
        for dependency in self.get_dependencies(module):
            self._visit_dependencies(dependency, visited)

    def _visit_dependents(self, module: str, visited: set[str]) -> None:
        if module in visited:
            return
        visited.add(module)
        for dependent in self.get_dependents(module):
            self._visit_dependents(dependent, visited)

    def _detect_cycles(
        self,
        module: str,
        visited: set[str],
        active: set[str],
        path: list[str],
        cycles: list[list[str]],
        seen: set[tuple[str, ...]],
    ) -> None:
        visited.add(module)
        active.add(module)
        path.append(module)

        for dependency in sorted(self.get_dependencies(module)):
            if dependency not in visited:
                self._detect_cycles(dependency, visited, active, path, cycles, seen)
            elif dependency in active:
                cycle_start = path.index(dependency)
                cycle = path[cycle_start:] + [dependency]
                canonical = self._canonical_cycle(cycle)
                if canonical not in seen:
                    seen.add(canonical)
                    cycles.append(cycle)

        path.pop()
        active.remove(module)

    @staticmethod
    def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
        body = cycle[:-1]
        rotations = [tuple(body[index:] + body[:index]) for index in range(len(body))]
        return min(rotations)

    @staticmethod
    def _clean_alias(name: str) -> str:
        return name.split(" as ", 1)[0].strip()

    @staticmethod
    def _format_import(import_info: ImportInfo) -> str:
        if import_info.kind == "import":
            return f"import {', '.join(import_info.names)}"

        prefix = "." * import_info.level
        return f"from {prefix}{import_info.module} import {', '.join(import_info.names)}"

    @staticmethod
    def _strip_reference_suffix(reference: str) -> str:
        return reference.split("[", 1)[0].split("(", 1)[0].strip()
