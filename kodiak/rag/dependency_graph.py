import ast
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class DependencyNode(BaseModel):
    name: str
    dependencies: set[str] = Field(default_factory=set)


class DependencyGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, DependencyNode] = {}

    def add_node(self, name: str) -> None:
        if name not in self._nodes:
            self._nodes[name] = DependencyNode(name=name)

    def add_dependency(self, module: str, depends_on: str) -> None:
        self.add_node(module)
        self.add_node(depends_on)
        self._nodes[module].dependencies.add(depends_on)

    def get_dependencies(self, module: str) -> set[str]:
        node = self._nodes.get(module)
        return node.dependencies if node else set()

    def get_all_dependencies(self, module: str) -> set[str]:
        visited: set[str] = set()
        self._dfs(module, visited)
        visited.discard(module)
        return visited

    def detect_cycles(self) -> list[list[str]]:
        visited: set[str] = set()
        stack: set[str] = set()
        cycles: list[list[str]] = []

        for node in self._nodes:
            if node not in visited:
                self._detect_cycles_dfs(node, visited, stack, [], cycles)

        return cycles

    def parse_file(self, file_path: Path) -> int:
        if not file_path.is_file() or file_path.suffix != ".py":
            return 0

        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        module_name = file_path.stem
        count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.add_dependency(module_name, alias.name.split(".")[0])
                    count += 1
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.add_dependency(module_name, node.module.split(".")[0])
                    count += 1

        return count

    def _dfs(self, node: str, visited: set[str]) -> None:
        if node in visited:
            return
        visited.add(node)
        for dep in self.get_dependencies(node):
            self._dfs(dep, visited)

    def _detect_cycles_dfs(
        self, node: str, visited: set[str], stack: set[str], path: list[str], cycles: list[list[str]]
    ) -> None:
        visited.add(node)
        stack.add(node)
        path.append(node)

        for dep in self.get_dependencies(node):
            if dep not in visited:
                self._detect_cycles_dfs(dep, visited, stack, path, cycles)
            elif dep in stack:
                cycle_start = path.index(dep)
                cycles.append(path[cycle_start:] + [dep])

        path.pop()
        stack.remove(node)