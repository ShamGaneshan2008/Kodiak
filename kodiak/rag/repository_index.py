"""
AST-based repository indexing for Kodiak V3.

This module builds a lightweight, in-memory structural index for Python
repositories. It intentionally avoids LLM calls, embeddings, database access,
semantic search, and code execution. The index is suitable for later RAG
components that need reliable repository structure before choosing a retrieval
strategy.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
    }
)


@dataclass(frozen=True)
class ImportInfo:
    """A single import statement discovered in a Python module."""

    module: str
    names: tuple[str, ...]
    line: int
    kind: str
    level: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this import."""
        return {
            "module": self.module,
            "names": list(self.names),
            "line": self.line,
            "kind": self.kind,
            "level": self.level,
        }


@dataclass(frozen=True)
class FunctionInfo:
    """A function, async function, or method discovered in a Python module."""

    name: str
    qualname: str
    line: int
    end_line: int
    signature: str
    docstring: str | None = None
    is_async: bool = False
    class_name: str | None = None
    decorators: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this function."""
        return {
            "name": self.name,
            "qualname": self.qualname,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "is_async": self.is_async,
            "class_name": self.class_name,
            "decorators": list(self.decorators),
        }


@dataclass(frozen=True)
class ClassInfo:
    """A class definition discovered in a Python module."""

    name: str
    qualname: str
    line: int
    end_line: int
    bases: tuple[str, ...] = field(default_factory=tuple)
    docstring: str | None = None
    decorators: tuple[str, ...] = field(default_factory=tuple)
    methods: tuple[FunctionInfo, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this class."""
        return {
            "name": self.name,
            "qualname": self.qualname,
            "line": self.line,
            "end_line": self.end_line,
            "bases": list(self.bases),
            "docstring": self.docstring,
            "decorators": list(self.decorators),
            "methods": [method.to_dict() for method in self.methods],
        }


@dataclass(frozen=True)
class ModuleInfo:
    """Structural information extracted from one Python source file."""

    module_name: str
    path: Path
    relative_path: Path
    line_count: int
    docstring: str | None = None
    imports: tuple[ImportInfo, ...] = field(default_factory=tuple)
    classes: tuple[ClassInfo, ...] = field(default_factory=tuple)
    functions: tuple[FunctionInfo, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this module."""
        return {
            "module_name": self.module_name,
            "path": str(self.path),
            "relative_path": self.relative_path.as_posix(),
            "line_count": self.line_count,
            "docstring": self.docstring,
            "imports": [import_info.to_dict() for import_info in self.imports],
            "classes": [class_info.to_dict() for class_info in self.classes],
            "functions": [function.to_dict() for function in self.functions],
        }


@dataclass(frozen=True)
class IndexingError:
    """A non-fatal error encountered while scanning or parsing a repository."""

    path: Path
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation of this error."""
        return {"path": str(self.path), "message": self.message}


@dataclass(frozen=True)
class RepositoryIndex:
    """An in-memory structural index for a Python repository."""

    root_path: Path
    modules: tuple[ModuleInfo, ...] = field(default_factory=tuple)
    errors: tuple[IndexingError, ...] = field(default_factory=tuple)

    @property
    def module_count(self) -> int:
        """Return the number of Python modules indexed successfully."""
        return len(self.modules)

    @property
    def class_count(self) -> int:
        """Return the total number of classes indexed across all modules."""
        return sum(len(module.classes) for module in self.modules)

    @property
    def function_count(self) -> int:
        """Return the total number of functions and methods indexed."""
        return sum(len(module.functions) for module in self.modules)

    @property
    def import_count(self) -> int:
        """Return the total number of import statements indexed."""
        return sum(len(module.imports) for module in self.modules)

    @property
    def has_errors(self) -> bool:
        """Return whether any files could not be scanned or parsed."""
        return bool(self.errors)

    def get_module(self, path: str | Path) -> ModuleInfo | None:
        """Return a module by absolute or repository-relative path."""
        requested = Path(path)
        for module in self.modules:
            if requested == module.path or requested == module.relative_path:
                return module
            if requested.as_posix() == module.relative_path.as_posix():
                return module
        return None

    def find_classes(self, name: str) -> list[ClassInfo]:
        """Return all classes matching ``name`` or a qualified class name."""
        return [
            class_info
            for module in self.modules
            for class_info in module.classes
            if class_info.name == name or class_info.qualname == name
        ]

    def find_functions(self, name: str) -> list[FunctionInfo]:
        """Return all functions matching ``name`` or a qualified function name."""
        return [
            function
            for module in self.modules
            for function in module.functions
            if function.name == name or function.qualname == name
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the repository index."""
        return {
            "root_path": str(self.root_path),
            "module_count": self.module_count,
            "class_count": self.class_count,
            "function_count": self.function_count,
            "import_count": self.import_count,
            "modules": [module.to_dict() for module in self.modules],
            "errors": [error.to_dict() for error in self.errors],
        }


class RepositoryIndexer:
    """
    Build an in-memory AST index for Python files under a repository root.

    The indexer only reads source files. It does not execute repository code,
    write files, access databases, call LLMs, compute embeddings, or perform
    semantic search.
    """

    def __init__(self, ignored_dirs: set[str] | frozenset[str] | None = None) -> None:
        """
        Initialize a repository indexer.

        Args:
            ignored_dirs: Optional directory names to skip while scanning. When
                omitted, Kodiak's default ignored directory set is used.
        """
        self.ignored_dirs = frozenset(ignored_dirs or DEFAULT_IGNORED_DIRS)

    def index(self, root_path: str | Path) -> RepositoryIndex:
        """
        Recursively scan ``root_path`` and return an in-memory repository index.

        Args:
            root_path: Local repository root to scan.

        Raises:
            ValueError: If ``root_path`` does not exist or is not a directory.
        """
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Repository root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Repository root is not a directory: {root}")

        modules: list[ModuleInfo] = []
        errors: list[IndexingError] = []

        logger.info("repository_index_start", root_path=str(root))
        for file_path in self.iter_python_files(root):
            try:
                modules.append(self.index_file(file_path, root))
            except (OSError, UnicodeError, SyntaxError) as exc:
                errors.append(IndexingError(path=file_path, message=str(exc)))
                logger.warning(
                    "repository_index_file_failed",
                    path=str(file_path),
                    error=str(exc),
                )

        modules.sort(key=lambda module: module.relative_path.as_posix())
        index = RepositoryIndex(
            root_path=root,
            modules=tuple(modules),
            errors=tuple(errors),
        )
        logger.info(
            "repository_index_complete",
            root_path=str(root),
            modules=index.module_count,
            classes=index.class_count,
            functions=index.function_count,
            imports=index.import_count,
            errors=len(index.errors),
        )
        return index

    def index_repository(self, root_path: str | Path) -> RepositoryIndex:
        """
        Recursively scan ``root_path`` and return an in-memory repository index.

        This is an explicit alias for callers that prefer a verb matching the
        class name.
        """
        return self.index(root_path)

    def iter_python_files(self, root_path: str | Path) -> tuple[Path, ...]:
        """
        Return Python files under ``root_path`` while pruning ignored directories.

        Directory read errors are logged and skipped so that one inaccessible
        path does not prevent indexing the rest of the repository.
        """
        root = Path(root_path)
        files: list[Path] = []
        self._collect_python_files(root, files)
        files.sort(key=lambda path: path.as_posix())
        return tuple(files)

    def index_file(self, file_path: str | Path, root_path: str | Path | None = None) -> ModuleInfo:
        """
        Parse one Python file and return its module index.

        Args:
            file_path: Python source file to parse.
            root_path: Optional repository root used to compute relative paths
                and dotted module names. Defaults to the file's parent.

        Raises:
            SyntaxError: If the file is not valid Python.
            OSError: If the file cannot be read.
        """
        path = Path(file_path).resolve()
        root = Path(root_path).resolve() if root_path is not None else path.parent
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
        relative_path = self._safe_relative_path(path, root)

        visitor = _ModuleVisitor()
        visitor.visit(tree)

        return ModuleInfo(
            module_name=self._module_name(relative_path),
            path=path,
            relative_path=relative_path,
            line_count=len(source.splitlines()),
            docstring=ast.get_docstring(tree),
            imports=tuple(visitor.imports),
            classes=tuple(visitor.classes),
            functions=tuple(visitor.functions),
        )

    def _collect_python_files(self, directory: Path, files: list[Path]) -> None:
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            logger.warning(
                "repository_index_directory_skipped",
                path=str(directory),
                error=str(exc),
            )
            return

        for child in children:
            if child.is_dir():
                if child.name in self.ignored_dirs:
                    continue
                self._collect_python_files(child, files)
            elif child.is_file() and child.suffix == ".py":
                files.append(child.resolve())

    @staticmethod
    def _safe_relative_path(path: Path, root: Path) -> Path:
        try:
            return path.relative_to(root)
        except ValueError:
            return Path(path.name)

    @staticmethod
    def _module_name(relative_path: Path) -> str:
        parts = list(relative_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)


class _ModuleVisitor(ast.NodeVisitor):
    """AST visitor that extracts imports, classes, functions, and docstrings."""

    def __init__(self) -> None:
        self.imports: list[ImportInfo] = []
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self._class_stack: list[str] = []
        self._function_stack: list[str] = []
        self._class_methods: dict[str, list[FunctionInfo]] = {}

    def visit_Import(self, node: ast.Import) -> None:
        """Record an ``import`` statement."""
        self.imports.append(
            ImportInfo(
                module="",
                names=tuple(self._format_alias(alias) for alias in node.names),
                line=node.lineno,
                kind="import",
            )
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record a ``from ... import ...`` statement."""
        self.imports.append(
            ImportInfo(
                module=node.module or "",
                names=tuple(self._format_alias(alias) for alias in node.names),
                line=node.lineno,
                kind="from",
                level=node.level,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record a class and visit nested definitions."""
        qualname = self._qualname(node.name)
        self._class_methods.setdefault(qualname, [])
        self._class_stack.append(node.name)

        self.generic_visit(node)

        self.classes.append(
            ClassInfo(
                name=node.name,
                qualname=qualname,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                bases=tuple(self._unparse(base) for base in node.bases),
                docstring=ast.get_docstring(node),
                decorators=tuple(self._unparse(decorator) for decorator in node.decorator_list),
                methods=tuple(self._class_methods.get(qualname, ())),
            )
        )
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record a synchronous function or method."""
        self._record_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record an asynchronous function or method."""
        self._record_function(node, is_async=True)

    def _record_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool,
    ) -> None:
        class_name = self._class_stack[-1] if self._class_stack else None
        function = FunctionInfo(
            name=node.name,
            qualname=self._qualname(node.name),
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            signature=self._function_signature(node, is_async=is_async),
            docstring=ast.get_docstring(node),
            is_async=is_async,
            class_name=class_name,
            decorators=tuple(self._unparse(decorator) for decorator in node.decorator_list),
        )
        self.functions.append(function)
        if class_name is not None and not self._function_stack:
            class_qualname = ".".join(self._class_stack)
            self._class_methods.setdefault(class_qualname, []).append(function)

        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def _qualname(self, name: str) -> str:
        parts = [*self._class_stack, *self._function_stack, name]
        return ".".join(parts)

    @staticmethod
    def _format_alias(alias: ast.alias) -> str:
        if alias.asname:
            return f"{alias.name} as {alias.asname}"
        return alias.name

    @staticmethod
    def _function_signature(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool,
    ) -> str:
        prefix = "async def" if is_async else "def"
        returns = f" -> {_ModuleVisitor._unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({_ModuleVisitor._unparse(node.args)}){returns}"

    @staticmethod
    def _unparse(node: ast.AST | None) -> str:
        if node is None:
            return ""
        try:
            return ast.unparse(node)
        except Exception as exc:
            logger.debug("repository_index_unparse_failed", error=str(exc))
            return ""
