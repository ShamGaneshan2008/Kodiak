"""
kodiak/rag/indexer.py

Orchestrates the full indexing pipeline:
  file content → chunker → embedder → vector store + symbol index

Supports incremental indexing via file-hash change detection.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
<<<<<<< HEAD
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
=======
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
>>>>>>> narasimha/refactor/3-rag-type-hints

import structlog

from kodiak.rag.chunker import Chunk, Chunker, detect_language
from kodiak.rag.embedder import Embedder
from kodiak.rag.symbol_index import SymbolIndex
from kodiak.rag.vector_store import VectorStore

logger = structlog.get_logger(__name__)

# Config

<<<<<<< HEAD
IGNORED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".exe",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".lock",
    ".sum",
    ".mod",
    ".min.js",
    ".min.css",
}

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "coverage",
    ".eggs",
=======
IGNORED_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".lock", ".sum", ".mod",
    ".min.js", ".min.css",
}

IGNORED_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".mypy_cache", ".pytest_cache",
    ".tox", "coverage", ".eggs",
>>>>>>> narasimha/refactor/3-rag-type-hints
}

MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB


# Result types


@dataclass
class FileIndexResult:
    file_path: str
    chunks: int
    skipped: bool
    reason: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class IndexingReport:
    repo_id: str
    total_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.indexed_files / self.total_files

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "total_files": self.total_files,
            "indexed_files": self.indexed_files,
            "skipped_files": self.skipped_files,
            "total_chunks": self.total_chunks,
            "errors": self.errors,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "success_rate": round(self.success_rate, 3),
        }


# ---------------------------------------------------------------------------
# File hash tracker (in-memory; swap for Redis in production)
# ---------------------------------------------------------------------------


class FileHashTracker:
    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def compute(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def has_changed(self, repo_id: str, file_path: str, content: str) -> bool:
        key = f"{repo_id}:{file_path}"
        new_hash = self.compute(content)
        old_hash = self._hashes.get(key)
        if old_hash == new_hash:
            return False
        self._hashes[key] = new_hash
        return True

    def mark(self, repo_id: str, file_path: str, content: str) -> None:
        key = f"{repo_id}:{file_path}"
        self._hashes[key] = self.compute(content)

    def remove(self, repo_id: str, file_path: str) -> None:
        self._hashes.pop(f"{repo_id}:{file_path}", None)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    """
    End-to-end indexing pipeline for a repository's source files.

    Usage::

        indexer = Indexer(embedder=embedder, vector_store=vs, symbol_index=si)
        report = await indexer.index_repo(repo_id="org/repo", root_path="/tmp/repo")
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        symbol_index: SymbolIndex,
        chunker: Chunker | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.symbol_index = symbol_index
        self.chunker = chunker or Chunker()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tracker = FileHashTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_repo(
        self,
        repo_id: str,
        root_path: str,
        incremental: bool = True,
    ) -> IndexingReport:
        """
        Walk a local clone of a repository and index all supported files.

        Args:
            repo_id: Unique repository identifier (e.g. ``"org/repo"``).
            root_path: Local path to the repository root.
            incremental: Skip files whose content has not changed since last run.
        """
        start = time.perf_counter()
        report = IndexingReport(repo_id=repo_id)

        files = list(self._iter_files(root_path))
        report.total_files = len(files)

        logger.info(
            "indexer_start",
            repo_id=repo_id,
            root=root_path,
            total_files=len(files),
            incremental=incremental,
        )

        tasks = [self._index_file(repo_id, fp, incremental) for fp in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for fp, result in zip(files, results, strict=False):
            if isinstance(result, Exception):
                report.errors.append(f"{fp}: {result}")
                report.skipped_files += 1
            elif isinstance(result, FileIndexResult):
                if result.skipped:
                    report.skipped_files += 1
                else:
                    report.indexed_files += 1
                    report.total_chunks += result.chunks

        report.elapsed_sec = time.perf_counter() - start
        logger.info("indexer_complete", **report.to_dict())
        return report

    async def index_file(
        self,
        repo_id: str,
        file_path: str,
        content: str,
        force: bool = False,
    ) -> FileIndexResult:
        """Index a single file by content string."""
        return await self._index_content(repo_id, file_path, content, force=force)

    async def delete_file(self, repo_id: str, file_path: str) -> None:
        """Remove a file's chunks from the vector store and symbol index."""
        await asyncio.gather(
            self.vector_store.delete_file(repo_id, file_path),
            self.symbol_index.delete_file(repo_id, file_path),
        )
        self._tracker.remove(repo_id, file_path)
        logger.info("indexer_file_deleted", repo_id=repo_id, file=file_path)

    async def reindex_repo(self, repo_id: str, root_path: str) -> IndexingReport:
        """Force full re-index (ignores change detection)."""
        return await self.index_repo(repo_id, root_path, incremental=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _index_file(
        self,
        repo_id: str,
        file_path: str,
        incremental: bool,
    ) -> FileIndexResult:
        async with self._semaphore:
            try:
                content = await asyncio.to_thread(
                    Path(file_path).read_text,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as exc:
                return FileIndexResult(
                    file_path=file_path,
                    chunks=0,
                    skipped=True,
                    reason=f"read_error: {exc}",
                )
            return await self._index_content(repo_id, file_path, content, force=not incremental)

    async def _index_content(
        self,
        repo_id: str,
        file_path: str,
        content: str,
        force: bool = False,
    ) -> FileIndexResult:
        start = time.perf_counter()
        log = logger.bind(repo_id=repo_id, file=file_path)

        # Skip unchanged files in incremental mode
        if not force and not self._tracker.has_changed(repo_id, file_path, content):
            return FileIndexResult(file_path=file_path, chunks=0, skipped=True, reason="unchanged")

        # Size guard
        if len(content.encode()) > MAX_FILE_SIZE_BYTES:
            log.warning("indexer_file_too_large", size=len(content))
            return FileIndexResult(file_path=file_path, chunks=0, skipped=True, reason="too_large")

        # 1. Chunk
        chunks: list[Chunk] = self.chunker.chunk_file(content, file_path)
        if not chunks:
            return FileIndexResult(file_path=file_path, chunks=0, skipped=True, reason="no_chunks")

        # 2. Delete stale chunks for this file
        await self.vector_store.delete_file(repo_id, file_path)

        # 3. Embed
        chunk_embedding_pairs = await self.embedder.embed_chunks(chunks)
        embedded_chunks = [c for c, _ in chunk_embedding_pairs]
        embeddings = [e for _, e in chunk_embedding_pairs]

        # 4. Store in vector store
        await self.vector_store.upsert_chunks(repo_id, embedded_chunks, embeddings)

        # 5. Update symbol index
        await self.symbol_index.index_chunks(repo_id, chunks)

        elapsed = (time.perf_counter() - start) * 1000
        log.debug("indexer_file_done", chunks=len(chunks), elapsed_ms=round(elapsed))

        return FileIndexResult(
            file_path=file_path,
            chunks=len(chunks),
            skipped=False,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _iter_files(self, root: str) -> Iterator[str]:
        """Yield file paths worth indexing under root."""
        from pathlib import Path as P

        def _walk(p: P) -> Iterator[str]:
            for item in p.iterdir():
                if item.is_dir():
                    if item.name not in IGNORED_DIRS:
                        yield from _walk(item)
                elif item.is_file():
                    if self._should_index(item):
                        yield str(item)

        return _walk(P(root))

    def _should_index(self, path: Path) -> bool:
        name = path.name
        suffix = path.suffix.lower()

        if suffix in IGNORED_EXTENSIONS:
            return False
        if name.startswith("."):
            return False
        if any(name.endswith(ext) for ext in {".min.js", ".min.css"}):
            return False
        if detect_language(str(path)) == "text" and suffix not in {".md", ".txt", ".rst"}:
            return False

        return True
