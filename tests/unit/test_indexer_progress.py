"""
Tests for structured progress logging during repository indexing (issue #1).

Indexing previously logged only at the very start and very end of a run
(`indexer_start` / `indexer_complete`), giving no visibility into a
long-running scan. These tests confirm periodic `indexer_progress` events
are emitted with the repo id, files processed, total files, percent
complete, and elapsed time -- without changing what actually gets indexed.
"""

from __future__ import annotations

import kodiak.rag.indexer as indexer_module
from kodiak.rag.indexer import Indexer, _ProgressReporter


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


class CapturingLogger:
    """Records `.info(event, **fields)` calls; ignores `.bind()`/`.debug()`/`.warning()`."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def info(self, event, **fields):
        self.events.append((event, fields))

    def debug(self, event, **fields):
        pass

    def warning(self, event, **fields):
        pass

    def bind(self, **fields):
        return self


def make_indexer() -> Indexer:
    return Indexer(
        embedder=FakeEmbedder(), vector_store=FakeVectorStore(), symbol_index=FakeSymbolIndex()
    )


def write_files(tmp_path, count: int) -> None:
    for i in range(count):
        (tmp_path / f"module_{i}.py").write_text(f"def fn_{i}():\n    return {i}\n")


# ---------------------------------------------------------------------------
# _ProgressReporter unit-level behavior
# ---------------------------------------------------------------------------


def test_reporter_logs_at_each_ten_percent_threshold(monkeypatch):
    fake_log = CapturingLogger()
    monkeypatch.setattr(indexer_module, "logger", fake_log)

    reporter = _ProgressReporter(repo_id="org/repo", total=50)
    for _ in range(50):
        reporter.tick()

    progress_events = [e for e in fake_log.events if e[0] == "indexer_progress"]
    assert len(progress_events) == 10  # one per 10% step, not one per file

    percents = [fields["percent_complete"] for _, fields in progress_events]
    assert percents == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def test_reporter_always_logs_final_completion_even_off_threshold(monkeypatch):
    fake_log = CapturingLogger()
    monkeypatch.setattr(indexer_module, "logger", fake_log)

    reporter = _ProgressReporter(repo_id="org/repo", total=3)
    for _ in range(3):
        reporter.tick()

    progress_events = [e for e in fake_log.events if e[0] == "indexer_progress"]
    assert progress_events[-1][1]["files_processed"] == 3
    assert progress_events[-1][1]["total_files"] == 3
    assert progress_events[-1][1]["percent_complete"] == 100


def test_reporter_handles_zero_total_without_logging_or_crashing(monkeypatch):
    fake_log = CapturingLogger()
    monkeypatch.setattr(indexer_module, "logger", fake_log)

    reporter = _ProgressReporter(repo_id="org/repo", total=0)
    reporter.tick()  # must not raise a ZeroDivisionError

    assert [e for e in fake_log.events if e[0] == "indexer_progress"] == []


def test_reporter_events_include_repo_id_and_elapsed_time(monkeypatch):
    fake_log = CapturingLogger()
    monkeypatch.setattr(indexer_module, "logger", fake_log)

    reporter = _ProgressReporter(repo_id="my-org/my-repo", total=1)
    reporter.tick()

    _, fields = next(e for e in fake_log.events if e[0] == "indexer_progress")
    assert fields["repo_id"] == "my-org/my-repo"
    assert "elapsed_sec" in fields


# ---------------------------------------------------------------------------
# End-to-end through Indexer.index_repo()
# ---------------------------------------------------------------------------


async def test_index_repo_emits_start_progress_and_complete_events(tmp_path, monkeypatch):
    fake_log = CapturingLogger()
    monkeypatch.setattr(indexer_module, "logger", fake_log)
    write_files(tmp_path, 20)
    indexer = make_indexer()

    report = await indexer.index_repo("org/repo", str(tmp_path))

    event_names = [e[0] for e in fake_log.events]
    assert "indexer_start" in event_names
    assert "indexer_progress" in event_names
    assert "indexer_complete" in event_names
    # indexing behavior itself is unaffected by the added logging
    assert report.total_files == 20
    assert report.indexed_files == 20


async def test_index_repo_progress_reflects_correct_totals(tmp_path, monkeypatch):
    fake_log = CapturingLogger()
    monkeypatch.setattr(indexer_module, "logger", fake_log)
    write_files(tmp_path, 10)
    indexer = make_indexer()

    await indexer.index_repo("org/repo", str(tmp_path))

    progress_events = [fields for name, fields in fake_log.events if name == "indexer_progress"]
    assert progress_events  # at least one emitted
    last = progress_events[-1]
    assert last["files_processed"] == 10
    assert last["total_files"] == 10
    assert last["percent_complete"] == 100
