"""
Tests for descriptive read-error messages in kodiak/rag/indexer.py (issue #2).

_index_file() used to report any read failure as just `f"read_error: {exc}"`
-- whatever the raw exception's repr happened to be. These tests confirm
common, recognizable causes (permission denied, file removed after the scan,
a directory where a file was expected) now get a message that explains the
cause and suggests a fix, while unrecognized exceptions still fall back to
something reasonable rather than crashing.
"""

from __future__ import annotations

import os
import sys

import pytest

from kodiak.rag.indexer import Indexer, _describe_read_error


class FakeEmbedder:
    async def embed_chunks(self, chunks):
        return [(c, [0.0]) for c in chunks]


class FakeVectorStore:
    async def delete_file(self, repo_id, file_path):
        pass

    async def upsert_chunks(self, repo_id, chunks, embeddings):
        pass


class FakeSymbolIndex:
    async def index_chunks(self, repo_id, chunks):
        pass

    async def delete_file(self, repo_id, file_path):
        pass


def make_indexer() -> Indexer:
    return Indexer(
        embedder=FakeEmbedder(), vector_store=FakeVectorStore(), symbol_index=FakeSymbolIndex()
    )


# ---------------------------------------------------------------------------
# _describe_read_error() unit-level: exact message content per exception type
# ---------------------------------------------------------------------------


def test_permission_error_explains_cause_and_suggests_a_fix():
    msg = _describe_read_error(PermissionError("denied"), "/repo/secret.py")
    assert msg.startswith("read_error: ")
    assert "permission" in msg.lower()
    assert "/repo/secret.py" in msg
    assert "permission" in msg.lower() and ("check" in msg.lower() or "access" in msg.lower())


def test_file_not_found_explains_it_may_have_been_deleted_since_scan():
    msg = _describe_read_error(FileNotFoundError("gone"), "/repo/deleted.py")
    assert "no longer exists" in msg
    assert "re-run indexing" in msg


def test_is_a_directory_error_explains_symlink_possibility():
    msg = _describe_read_error(IsADirectoryError("is a dir"), "/repo/oddly_named")
    assert "directory, not a file" in msg
    assert "symlink" in msg.lower()


def test_unicode_error_explains_encoding_possibility():
    msg = _describe_read_error(UnicodeError("bad bytes"), "/repo/binaryish.py")
    assert "decoded" in msg
    assert "encoding" in msg.lower()


def test_generic_os_error_still_includes_original_exception_text():
    exc = OSError("device not ready")
    msg = _describe_read_error(exc, "/repo/x.py")
    assert "OS error" in msg
    assert "device not ready" in msg


def test_unrecognized_exception_falls_back_without_crashing():
    msg = _describe_read_error(RuntimeError("something odd"), "/repo/x.py")
    assert msg.startswith("read_error: ")
    assert "something odd" in msg


# ---------------------------------------------------------------------------
# End-to-end through Indexer._index_file(), not just the helper in isolation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file permission bits",
)
async def test_permission_denied_file_is_skipped_with_descriptive_reason(tmp_path):
    unreadable = tmp_path / "unreadable.py"
    unreadable.write_text("def foo():\n    return 1\n")
    unreadable.chmod(0o000)
    indexer = make_indexer()

    try:
        result = await indexer._index_file("org/repo", str(unreadable), incremental=True)
    finally:
        unreadable.chmod(0o644)  # restore so tmp_path cleanup can remove it

    assert result.skipped is True
    assert result.reason is not None
    assert "permission" in result.reason.lower()


async def test_file_deleted_between_discovery_and_read_is_skipped_with_descriptive_reason(tmp_path):
    ghost = tmp_path / "ghost.py"  # never created on disk
    indexer = make_indexer()

    result = await indexer._index_file("org/repo", str(ghost), incremental=True)

    assert result.skipped is True
    assert result.reason is not None
    assert "no longer exists" in result.reason


async def test_path_that_is_a_directory_is_skipped_with_descriptive_reason(tmp_path):
    a_dir = tmp_path / "looks_like_a_file.py"
    a_dir.mkdir()
    indexer = make_indexer()

    result = await indexer._index_file("org/repo", str(a_dir), incremental=True)

    assert result.skipped is True
    assert result.reason is not None
    assert "directory, not a file" in result.reason
