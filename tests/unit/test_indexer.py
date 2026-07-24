"""
Tests for kodiak/rag/indexer.py.

Covers the scenarios requested in issue #5: empty repositories, invalid
paths, duplicate indexing, and large repositories. Uses fakes for
embedder/vector_store/symbol_index rather than the real backends, both to
keep the tests fast/offline and because the real SymbolIndex class doesn't
actually implement the index_chunks()/delete_file() methods Indexer calls
on it (a separate, pre-existing gap -- out of scope here, but it's why a
fake is the right choice for testing Indexer's own logic in isolation).
"""

from __future__ import annotations

import pytest

from kodiak.rag.chunker import Chunk
from kodiak.rag.indexer import FileHashTracker, Indexer


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[Chunk]] = []

    async def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        self.calls.append(chunks)
        return [(c, [0.0, 0.0]) for c in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.upserted: list[tuple[str, list[Chunk]]] = []

    async def delete_file(self, repo_id: str, file_path: str) -> None:
        self.deleted.append((repo_id, file_path))

    async def upsert_chunks(
        self, repo_id: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        self.upserted.append((repo_id, chunks))


class FakeSymbolIndex:
    def __init__(self) -> None:
        self.indexed: list[tuple[str, list[Chunk]]] = []
        self.deleted: list[tuple[str, str]] = []

    async def index_chunks(self, repo_id: str, chunks: list[Chunk]) -> None:
        self.indexed.append((repo_id, chunks))

    async def delete_file(self, repo_id: str, file_path: str) -> None:
        self.deleted.append((repo_id, file_path))


def make_indexer() -> Indexer:
    return Indexer(
        embedder=FakeEmbedder(), vector_store=FakeVectorStore(), symbol_index=FakeSymbolIndex()
    )


# ---------------------------------------------------------------------------
# Empty repositories
# ---------------------------------------------------------------------------


async def test_empty_directory_produces_a_clean_zero_report(tmp_path) -> None:
    indexer = make_indexer()

    report = await indexer.index_repo("org/repo", str(tmp_path))

    assert report.total_files == 0
    assert report.indexed_files == 0
    assert report.skipped_files == 0
    assert report.total_chunks == 0
    assert report.errors == []
    assert report.success_rate == 0.0  # guarded div-by-zero, not NaN/crash


async def test_directory_with_only_ignored_files_is_also_empty(tmp_path) -> None:
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "notes.pyc").write_bytes(b"")
    (tmp_path / ".hidden").write_text("secret")

    report = await make_indexer().index_repo("org/repo", str(tmp_path))

    assert report.total_files == 0
    assert report.indexed_files == 0


# ---------------------------------------------------------------------------
# Invalid paths
# ---------------------------------------------------------------------------


async def test_nonexistent_root_path_raises_file_not_found(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        await make_indexer().index_repo("org/repo", str(missing))


async def test_root_path_that_is_a_file_not_a_directory_raises(tmp_path) -> None:
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("hello")

    with pytest.raises(NotADirectoryError):
        await make_indexer().index_repo("org/repo", str(a_file))


# ---------------------------------------------------------------------------
# Duplicate indexing
# ---------------------------------------------------------------------------


async def test_second_run_with_unchanged_content_skips_everything(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")
    indexer = make_indexer()

    first = await indexer.index_repo("org/repo", str(tmp_path))
    second = await indexer.index_repo("org/repo", str(tmp_path))

    assert first.indexed_files == 2
    assert first.skipped_files == 0
    assert second.indexed_files == 0
    assert second.skipped_files == 2


async def test_second_run_after_content_change_reindexes_only_the_changed_file(tmp_path) -> None:
    file_a = tmp_path / "a.py"
    file_a.write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")
    indexer = make_indexer()

    await indexer.index_repo("org/repo", str(tmp_path))
    file_a.write_text("def foo():\n    return 999\n")
    second = await indexer.index_repo("org/repo", str(tmp_path))

    assert second.indexed_files == 1
    assert second.skipped_files == 1


async def test_reindex_repo_forces_full_reindex_even_when_unchanged(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    indexer = make_indexer()

    await indexer.index_repo("org/repo", str(tmp_path))
    forced = await indexer.reindex_repo("org/repo", str(tmp_path))

    assert forced.indexed_files == 1
    assert forced.skipped_files == 0


def test_file_hash_tracker_detects_changes_and_stability() -> None:
    tracker = FileHashTracker()

    assert tracker.has_changed("repo", "a.py", "content-1") is True
    tracker.mark("repo", "a.py", "content-1")
    assert tracker.has_changed("repo", "a.py", "content-1") is False
    assert tracker.has_changed("repo", "a.py", "content-2") is True


# ---------------------------------------------------------------------------
# Large repositories
# ---------------------------------------------------------------------------


async def test_large_number_of_files_are_all_indexed_correctly(tmp_path) -> None:
    file_count = 120
    for i in range(file_count):
        (tmp_path / f"module_{i}.py").write_text(f"def fn_{i}():\n    return {i}\n")

    report = await make_indexer().index_repo("org/repo", str(tmp_path))

    assert report.total_files == file_count
    assert report.indexed_files == file_count
    assert report.skipped_files == 0
    assert report.errors == []
    assert report.total_chunks >= file_count  # at least one chunk per file


# ---------------------------------------------------------------------------
# General behavior (not explicitly requested, but directly adjacent and
# cheap to lock in while adding the coverage above)
# ---------------------------------------------------------------------------


async def test_delete_file_removes_from_vector_store_and_symbol_index_and_clears_tracker(
    tmp_path,
) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    indexer = make_indexer()
    await indexer.index_repo("org/repo", str(tmp_path))

    await indexer.delete_file("org/repo", str(tmp_path / "a.py"))

    assert (("org/repo", str(tmp_path / "a.py"))) in indexer.vector_store.deleted
    assert (("org/repo", str(tmp_path / "a.py"))) in indexer.symbol_index.deleted
    # tracker cleared -> re-indexing with the same content is no longer "unchanged"
    second = await indexer.index_repo("org/repo", str(tmp_path))
    assert second.indexed_files == 1


async def test_file_that_fails_to_read_is_recorded_as_an_error_not_a_crash(tmp_path) -> None:
    good = tmp_path / "good.py"
    good.write_text("def foo():\n    return 1\n")
    bad_dir_as_file_path = tmp_path / "phantom.py"
    # Never actually created on disk -> read_text() raises inside _index_file,
    # which is caught and turned into an error entry rather than propagating.
    indexer = make_indexer()

    files = list(indexer._iter_files(str(tmp_path)))
    files.append(str(bad_dir_as_file_path))

    result_good = await indexer._index_file("org/repo", str(good), incremental=True)
    result_bad = await indexer._index_file("org/repo", str(bad_dir_as_file_path), incremental=True)

    assert result_good.skipped is False
    assert result_bad.skipped is True
    assert result_bad.reason is not None and "read_error" in result_bad.reason
