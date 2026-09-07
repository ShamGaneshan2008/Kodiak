"""Lightweight file-content change tracking without optional RAG dependencies."""

from __future__ import annotations

import hashlib


class FileHashTracker:
    """Track content hashes by repository and relative path in memory."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def compute(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def has_changed(self, repo_id: str, file_path: str, content: str) -> bool:
        key = f"{repo_id}:{file_path}"
        new_hash = self.compute(content)
        if self._hashes.get(key) == new_hash:
            return False
        self._hashes[key] = new_hash
        return True

    def mark(self, repo_id: str, file_path: str, content: str) -> None:
        self._hashes[f"{repo_id}:{file_path}"] = self.compute(content)

    def remove(self, repo_id: str, file_path: str) -> None:
        self._hashes.pop(f"{repo_id}:{file_path}", None)
