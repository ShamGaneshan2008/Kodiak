"""
kodiak/rag/chunker.py

Semantic chunking for code and text content.
Supports AST-aware splitting for Python, JS/TS, and fallback line-based chunking.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ChunkType(StrEnum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    BLOCK = "block"
    TEXT = "text"
    COMMENT = "comment"
    IMPORT = "import"


@dataclass
class Chunk:
    """A single semantic chunk of code or text."""

    content: str
    chunk_type: ChunkType
    file_path: str
    start_line: int
    end_line: int
    language: str
    name: str | None = None
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Stable deterministic ID based on content + location."""
        raw = f"{self.file_path}:{self.start_line}:{self.end_line}:{self.content[:64]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return len(self.content) // 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "name": self.name,
            "parent": self.parent,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


class PythonASTChunker:
    """AST-aware chunker for Python source files."""

    def __init__(self, max_tokens: int = 512, overlap_lines: int = 3) -> None:
        self.max_tokens = max_tokens
        self.overlap_lines = overlap_lines

    def chunk(self, source: str, file_path: str) -> list[Chunk]:
        lines = source.splitlines()
        chunks: list[Chunk] = []

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning("ast_parse_failed", file=file_path, error=str(exc))
            return self._fallback_chunk(source, file_path)

        # Collect top-level and class-level definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._node_to_chunk(node, lines, file_path, ChunkType.FUNCTION)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._node_to_chunk(node, lines, file_path, ChunkType.CLASS)
                if chunk:
                    chunks.append(chunk)

        # Collect module-level imports as a single chunk
        import_lines: list[int] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno)

        if import_lines:
            start = min(import_lines) - 1
            end = max(import_lines)
            content = "\n".join(lines[start:end])
            if content.strip():
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_type=ChunkType.IMPORT,
                        file_path=file_path,
                        start_line=start + 1,
                        end_line=end,
                        language="python",
                        name="imports",
                    )
                )

        # If nothing found, fall back
        if not chunks:
            return self._fallback_chunk(source, file_path)

        return sorted(chunks, key=lambda c: c.start_line)

    def _node_to_chunk(
        self,
        node: ast.AST,
        lines: list[str],
        file_path: str,
        chunk_type: ChunkType,
    ) -> Chunk | None:
        start = node.lineno - 1  # type: ignore[attr-defined]
        end = getattr(node, "end_lineno", start + 1)
        content = "\n".join(lines[start:end])

        if not content.strip():
            return None

        # If chunk is too large, split it
        if len(content) // 4 > self.max_tokens:
            content = self._truncate_with_ellipsis(content, self.max_tokens)

        parent = None
        if chunk_type == ChunkType.FUNCTION:
            # Detect if nested inside a class
            for parent_node in ast.walk(ast.parse("\n".join(lines))):
                if isinstance(parent_node, ast.ClassDef):
                    cls_start = parent_node.lineno - 1
                    cls_end = getattr(parent_node, "end_lineno", cls_start)
                    if cls_start <= start <= cls_end:
                        parent = parent_node.name
                        break

        return Chunk(
            content=content,
            chunk_type=chunk_type,
            file_path=file_path,
            start_line=start + 1,
            end_line=end,
            language="python",
            name=getattr(node, "name", None),
            parent=parent,
        )

    def _truncate_with_ellipsis(self, content: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n# ... (truncated)"

    def _fallback_chunk(self, source: str, file_path: str) -> list[Chunk]:
        return LineBasedChunker(max_tokens=self.max_tokens, overlap_lines=self.overlap_lines).chunk(
            source, file_path, language="python"
        )


class LineBasedChunker:
    """Generic line-based chunker for any language."""

    def __init__(self, max_tokens: int = 512, overlap_lines: int = 5) -> None:
        self.max_tokens = max_tokens
        self.overlap_lines = overlap_lines

    def chunk(self, source: str, file_path: str, language: str = "text") -> list[Chunk]:
        lines = source.splitlines()
        chunks: list[Chunk] = []
        max_chars = self.max_tokens * 4

        current_lines: list[str] = []
        current_start = 1
        current_chars = 0

        for i, line in enumerate(lines, start=1):
            current_lines.append(line)
            current_chars += len(line) + 1  # +1 for newline

            if current_chars >= max_chars:
                content = "\n".join(current_lines)
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_type=ChunkType.BLOCK,
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i,
                        language=language,
                    )
                )
                # Overlap: keep last N lines
                overlap = current_lines[-self.overlap_lines :]
                current_start = i - self.overlap_lines + 1
                current_lines = overlap
                current_chars = sum(len(line) + 1 for line in overlap)

        # Remaining lines
        if current_lines:
            content = "\n".join(current_lines)
            if content.strip():
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_type=ChunkType.BLOCK,
                        file_path=file_path,
                        start_line=current_start,
                        end_line=len(lines),
                        language=language,
                    )
                )

        return chunks


LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".md": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "bash",
}


def detect_language(file_path: str) -> str:
    import os

    ext = os.path.splitext(file_path)[1].lower()
    return LANGUAGE_MAP.get(ext, "text")


class Chunker:
    """
    Main entry point. Dispatches to language-specific chunker.
    """

    def __init__(self, max_tokens: int = 512, overlap_lines: int = 5) -> None:
        self.max_tokens = max_tokens
        self.overlap_lines = overlap_lines
        self._python = PythonASTChunker(max_tokens=max_tokens, overlap_lines=overlap_lines)
        self._line = LineBasedChunker(max_tokens=max_tokens, overlap_lines=overlap_lines)

    def chunk_file(self, source: str, file_path: str) -> list[Chunk]:
        language = detect_language(file_path)
        log = logger.bind(file=file_path, language=language)

        if language == "python":
            chunks = self._python.chunk(source, file_path)
        else:
            chunks = self._line.chunk(source, file_path, language=language)

        log.debug("chunked_file", num_chunks=len(chunks))
        return chunks

    def chunk_text(self, text: str, source_id: str = "text") -> list[Chunk]:
        return self._line.chunk(text, source_id, language="text")

    def iter_chunks(self, source: str, file_path: str) -> Iterator[Chunk]:
        yield from self.chunk_file(source, file_path)
