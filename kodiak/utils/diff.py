import difflib
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass(frozen=True)
class DiffStatistics:
    added_lines: int = 0
    removed_lines: int = 0
    changed_files: int = 0


class DiffResult(BaseModel):
    is_different: bool
    statistics: DiffStatistics = field(default_factory=DiffStatistics)
    unified_diff: str = ""


def unified_diff(
    old_text: str,
    new_text: str,
    old_name: str = "original",
    new_name: str = "modified",
    context_lines: int = 3,
) -> DiffResult:
    """Generate a unified diff between two strings."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_name,
            tofile=new_name,
            n=context_lines,
        )
    )

    diff_text = "".join(diff_lines)
    stats = diff_statistics(old_text, new_text)

    return DiffResult(
        is_different=len(diff_lines) > 0,
        statistics=stats,
        unified_diff=diff_text,
    )


def compare_text(old_text: str, new_text: str) -> bool:
    """Check if two text strings are exactly different."""
    return old_text != new_text


def diff_statistics(old_text: str, new_text: str) -> DiffStatistics:
    """Calculate line-level addition and removal statistics."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    added = 0
    removed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            removed += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1

    return DiffStatistics(added_lines=added, removed_lines=removed, changed_files=1)