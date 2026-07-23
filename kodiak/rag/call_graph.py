import ast
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CallNode(BaseModel):
    name: str
    file_path: str
    callers: set[str] = Field(default_factory=set)
    callees: set[str] = Field(default_factory=set)


class CallGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, CallNode] = {}

    def _get_or_create(self, name: str, file_path: str = "") -> CallNode:
        if name not in self._nodes:
            self._nodes[name] = CallNode(name=name, file_path=file_path)
        elif file_path and not self._nodes[name].file_path:
            self._nodes[name].file_path = file_path
        return self._nodes[name]

    def add_call(self, caller: str, callee: str, caller_file: str = "") -> None:
        caller_node = self._get_or_create(caller, caller_file)
        callee_node = self._get_or_create(callee)
        caller_node.callees.add(callee)
        callee_node.callers.add(caller)

    def get_callees(self, func_name: str) -> set[str]:
        node = self._nodes.get(func_name)
        return node.callees if node else set()

    def get_callers(self, func_name: str) -> set[str]:
        node = self._nodes.get(func_name)
        return node.callers if node else set()

    def is_recursive(self, func_name: str) -> bool:
        return func_name in self.get_callees(func_name)

    def get_transitive_callees(self, func_name: str) -> set[str]:
        visited: set[str] = set()
        self._dfs_callees(func_name, visited)
        visited.discard(func_name)
        return visited

    def parse_file(self, file_path: Path) -> int:
        if not file_path.is_file() or file_path.suffix != ".py":
            return 0

        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = node.name
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee = self._extract_call_name(child)
                        if callee:
                            self.add_call(caller, callee, str(file_path))
                            count += 1
        logger.debug("call_graph_parsed", path=str(file_path), calls=count)
        return count

    def _dfs_callees(self, func_name: str, visited: set[str]) -> None:
        if func_name in visited:
            return
        visited.add(func_name)
        for callee in self.get_callees(func_name):
            self._dfs_callees(callee, visited)

    @staticmethod
    def _extract_call_name(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None
