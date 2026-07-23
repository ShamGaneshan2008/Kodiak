"""
Regression coverage for kodiak/rag/chunker.py.

Also serves as an import smoke test: this module previously landed on main
with an unresolved merge conflict (literal branch-name text left in the
source after a bad manual resolution) that made it -- and therefore every
module importing it -- fail to even parse. A basic test file here means CI
catches that class of breakage immediately instead of it going unnoticed.
"""

from __future__ import annotations

from kodiak.rag.chunker import Chunk, Chunker, ChunkType, detect_language


def test_chunker_python_file_produces_function_and_class_chunks() -> None:
    source = "def foo():\n    return 1\n\n\nclass Bar:\n    def baz(self):\n        return 2\n"
    chunks = Chunker().chunk_file(source, "example.py")

    types = {c.chunk_type for c in chunks}
    names = {c.name for c in chunks}
    assert ChunkType.FUNCTION in types
    assert ChunkType.CLASS in types
    assert "foo" in names
    assert "Bar" in names


def test_chunker_falls_back_to_line_based_for_invalid_python_syntax() -> None:
    # Syntax error -> PythonASTChunker.chunk() catches SyntaxError internally
    # and falls back to LineBasedChunker instead of raising.
    chunks = Chunker().chunk_file("def broken(:\n    pass\n", "broken.py")
    assert len(chunks) >= 1
    assert all(c.chunk_type == ChunkType.BLOCK for c in chunks)


def test_chunker_non_python_file_uses_line_based_chunking() -> None:
    chunks = Chunker().chunk_file("# Title\n\nSome text.\n", "README.md")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.BLOCK
    assert chunks[0].language == "markdown"


def test_chunker_empty_source_produces_no_chunks() -> None:
    assert Chunker().chunk_file("", "empty.py") == []


def test_detect_language_maps_known_extensions() -> None:
    assert detect_language("app.py") == "python"
    assert detect_language("index.ts") == "typescript"
    assert detect_language("README.md") == "markdown"
    assert detect_language("no_extension") == "text"


def test_chunk_id_is_stable_for_identical_chunks() -> None:
    kwargs = dict(
        content="return 1",
        chunk_type=ChunkType.FUNCTION,
        file_path="a.py",
        start_line=1,
        end_line=2,
        language="python",
    )
    assert Chunk(**kwargs).chunk_id == Chunk(**kwargs).chunk_id


def test_chunk_id_differs_for_different_content() -> None:
    base = dict(
        chunk_type=ChunkType.FUNCTION,
        file_path="a.py",
        start_line=1,
        end_line=2,
        language="python",
    )
    assert Chunk(content="return 1", **base).chunk_id != Chunk(content="return 2", **base).chunk_id
