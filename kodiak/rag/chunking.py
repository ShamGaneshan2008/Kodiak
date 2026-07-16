"""
Repository-index based semantic chunking for Kodiak V3.

This module converts objects produced by ``RepositoryIndexer`` into semantic
chunks. It does not scan the repository, parse Python syntax, execute code,
compute embeddings, search, call an LLM, modify files, or access databases.
Source text is read only for files already present in the supplied
``RepositoryIndex`` so chunks can include exact code snippets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from kodiak.rag.repository_index import ClassInfo, FunctionInfo, ImportInfo, ModuleInfo, RepositoryIndex

logger = structlog.get_logger(__name__)


class ChunkSymbolType(str, Enum):
    """Logical source object represented by a repository chunk."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CONSTANT_BLOCK = "constant_block"


@dataclass(frozen=True)
class RepositoryChunk:
    """A semantic chunk derived from an indexed repository object."""

    id: str
    module_path: str
    file_path: Path
    symbol_name: str
    symbol_type: ChunkSymbolType
    source_code: str
    docstring: str | None
    start_line: int
    end_line: int
    imports: tuple[str, ...] = field(default_factory=tuple)
    parent_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this chunk."""
        return {
            "id": self.id,
            "module_path": self.module_path,
            "file_path": str(self.file_path),
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type.value,
            "source_code": self.source_code,
            "docstring": self.docstring,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "imports": list(self.imports),
            "parent_class": self.parent_class,
            "metadata": self.metadata,
        }


class Chunker:
    """
    Convert ``RepositoryIndex`` models into semantic chunks.

    The chunker trusts the structural information produced by
    ``RepositoryIndexer``. It reads indexed files only to slice source text by
    the line numbers already present on ``ModuleInfo``, ``ClassInfo``, and
    ``FunctionInfo`` objects.
    """

    def __init__(self, include_module_chunks: bool = True) -> None:
        """
        Initialize a repository chunker.

        Args:
            include_module_chunks: Whether ``chunk_repository`` should include a
                module-level chunk for each indexed file in addition to class
                and function chunks.
        """
        self.include_module_chunks = include_module_chunks
        self._source_cache: dict[Path, tuple[str, ...]] = {}

    def chunk_repository(self, repository_index: RepositoryIndex) -> tuple[RepositoryChunk, ...]:
        """
        Chunk every indexed module in ``repository_index``.

        Args:
            repository_index: Repository structure created by
                ``RepositoryIndexer``.

        Returns:
            Chunks sorted by module path, start line, symbol type, and symbol
            name for deterministic downstream processing.
        """
        chunks: list[RepositoryChunk] = []
        for module in repository_index.modules:
            chunks.extend(self.chunk_module(module))

        ordered = tuple(
            sorted(
                chunks,
                key=lambda chunk: (
                    chunk.module_path,
                    chunk.start_line,
                    chunk.symbol_type.value,
                    chunk.symbol_name,
                ),
            )
        )
        logger.info(
            "repository_chunking_complete",
            modules=repository_index.module_count,
            chunks=len(ordered),
        )
        return ordered

    def chunk_module(self, module: ModuleInfo) -> tuple[RepositoryChunk, ...]:
        """
        Chunk one indexed module into module, class, and function chunks.

        Args:
            module: Indexed module produced by ``RepositoryIndexer``.

        Returns:
            Semantic chunks for the module in source order.
        """
        source_lines = self._source_lines(module.path)
        imports = self._format_imports(module.imports)
        chunks: list[RepositoryChunk] = []

        if self.include_module_chunks:
            chunks.append(self._module_chunk(module, source_lines, imports))

        for class_info in module.classes:
            chunks.append(self.chunk_class(module, class_info))

        for function_info in module.functions:
            chunks.append(self.chunk_function(module, function_info))

        return tuple(sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.symbol_type.value)))

    def chunk_class(self, module: ModuleInfo, class_info: ClassInfo) -> RepositoryChunk:
        """
        Build a chunk for one indexed class.

        Args:
            module: Module containing the class.
            class_info: Class metadata from ``RepositoryIndexer``.

        Returns:
            A class chunk containing source, inheritance metadata, methods, and
            import context.
        """
        source_lines = self._source_lines(module.path)
        imports = self._format_imports(module.imports)
        parent_class = self._parent_from_qualname(class_info.qualname)
        metadata: dict[str, Any] = {
            "bases": list(class_info.bases),
            "decorators": list(class_info.decorators),
            "methods": [method.qualname for method in class_info.methods],
            "relative_path": module.relative_path.as_posix(),
            "line_count": class_info.end_line - class_info.line + 1,
        }

        return self._build_chunk(
            module=module,
            symbol_name=class_info.qualname,
            symbol_type=ChunkSymbolType.CLASS,
            source_lines=source_lines,
            start_line=class_info.line,
            end_line=class_info.end_line,
            docstring=class_info.docstring,
            imports=imports,
            parent_class=parent_class,
            metadata=metadata,
        )

    def chunk_function(self, module: ModuleInfo, function_info: FunctionInfo) -> RepositoryChunk:
        """
        Build a chunk for one indexed function, async function, or method.

        Args:
            module: Module containing the function.
            function_info: Function metadata from ``RepositoryIndexer``.

        Returns:
            A function chunk containing source, signature metadata, decorators,
            parent class context, and import context.
        """
        source_lines = self._source_lines(module.path)
        imports = self._format_imports(module.imports)
        symbol_type = (
            ChunkSymbolType.ASYNC_FUNCTION
            if function_info.is_async
            else ChunkSymbolType.FUNCTION
        )
        metadata: dict[str, Any] = {
            "signature": function_info.signature,
            "decorators": list(function_info.decorators),
            "is_async": function_info.is_async,
            "relative_path": module.relative_path.as_posix(),
            "line_count": function_info.end_line - function_info.line + 1,
        }

        return self._build_chunk(
            module=module,
            symbol_name=function_info.qualname,
            symbol_type=symbol_type,
            source_lines=source_lines,
            start_line=function_info.line,
            end_line=function_info.end_line,
            docstring=function_info.docstring,
            imports=imports,
            parent_class=function_info.class_name,
            metadata=metadata,
        )

    def _module_chunk(
        self,
        module: ModuleInfo,
        source_lines: tuple[str, ...],
        imports: tuple[str, ...],
    ) -> RepositoryChunk:
        metadata: dict[str, Any] = {
            "relative_path": module.relative_path.as_posix(),
            "line_count": module.line_count,
            "class_count": len(module.classes),
            "function_count": len(module.functions),
            "import_count": len(module.imports),
            "classes": [class_info.qualname for class_info in module.classes],
            "functions": [function.qualname for function in module.functions],
            "constant_blocks": [],
            "constant_block_note": (
                "RepositoryIndex does not currently expose constant definitions; "
                "chunking avoids reparsing files to infer them."
            ),
        }
        end_line = max(module.line_count, len(source_lines), 1)

        return self._build_chunk(
            module=module,
            symbol_name=module.module_name,
            symbol_type=ChunkSymbolType.MODULE,
            source_lines=source_lines,
            start_line=1,
            end_line=end_line,
            docstring=module.docstring,
            imports=imports,
            parent_class=None,
            metadata=metadata,
        )

    def _build_chunk(
        self,
        *,
        module: ModuleInfo,
        symbol_name: str,
        symbol_type: ChunkSymbolType,
        source_lines: tuple[str, ...],
        start_line: int,
        end_line: int,
        docstring: str | None,
        imports: tuple[str, ...],
        parent_class: str | None,
        metadata: dict[str, Any],
    ) -> RepositoryChunk:
        bounded_start = max(start_line, 1)
        bounded_end = max(end_line, bounded_start)
        source_code = self._slice_source(source_lines, bounded_start, bounded_end)
        chunk_id = self._chunk_id(
            module_path=module.module_name,
            file_path=module.path,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            start_line=bounded_start,
            end_line=bounded_end,
        )

        return RepositoryChunk(
            id=chunk_id,
            module_path=module.module_name,
            file_path=module.path,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            source_code=source_code,
            docstring=docstring,
            start_line=bounded_start,
            end_line=bounded_end,
            imports=imports,
            parent_class=parent_class,
            metadata=metadata,
        )

    def _source_lines(self, path: Path) -> tuple[str, ...]:
        resolved = path.expanduser().resolve()
        if resolved not in self._source_cache:
            try:
                self._source_cache[resolved] = tuple(
                    resolved.read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError as exc:
                logger.warning("chunk_source_read_failed", path=str(resolved), error=str(exc))
                self._source_cache[resolved] = ()
        return self._source_cache[resolved]

    @staticmethod
    def _slice_source(source_lines: tuple[str, ...], start_line: int, end_line: int) -> str:
        if not source_lines:
            return ""

        start_index = max(start_line - 1, 0)
        end_index = min(end_line, len(source_lines))
        if start_index >= len(source_lines) or end_index <= start_index:
            return ""
        return "\n".join(source_lines[start_index:end_index])

    @staticmethod
    def _format_imports(imports: tuple[ImportInfo, ...]) -> tuple[str, ...]:
        formatted: list[str] = []
        for import_info in imports:
            if import_info.kind == "import":
                formatted.append(f"import {', '.join(import_info.names)}")
                continue

            prefix = "." * import_info.level
            formatted.append(
                f"from {prefix}{import_info.module} import {', '.join(import_info.names)}"
            )
        return tuple(formatted)

    @staticmethod
    def _parent_from_qualname(qualname: str) -> str | None:
        parts = qualname.split(".")
        if len(parts) <= 1:
            return None
        return ".".join(parts[:-1])

    @staticmethod
    def _chunk_id(
        *,
        module_path: str,
        file_path: Path,
        symbol_name: str,
        symbol_type: ChunkSymbolType,
        start_line: int,
        end_line: int,
    ) -> str:
        raw = (
            f"{module_path}:{file_path.as_posix()}:{symbol_type.value}:"
            f"{symbol_name}:{start_line}:{end_line}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
