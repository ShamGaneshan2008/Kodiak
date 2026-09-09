"""Repository-local, sanitized persistence for the Kodiak v1 workflow."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kodiak.orchestration.v1_models import ApprovalRequest

_SECRET_FIELD_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*"),
)


class LocalStateStore:
    """Persist state only beneath ``<repository>/.kodiak``.

    Malformed records are ignored when reading so a partially written local
    state file never prevents repository work. Writes of JSON documents are
    atomic on the same filesystem.
    """

    def __init__(self, repository_path: str | Path) -> None:
        root = Path(repository_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Repository path is not a directory: {root}")
        self.root = root
        self.state_dir = root / ".kodiak"
        self.approvals_path = self.state_dir / "approvals.json"
        self.history_path = self.state_dir / "task_history.jsonl"
        self.runs_dir = self.state_dir / "runs"
        self._assert_inside(self.state_dir)

    def history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return newest valid task records first."""
        if limit < 1:
            raise ValueError("History limit must be at least 1.")
        self._assert_inside(self.history_path)
        if not self.history_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return []
        for line in lines:
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                records.append(value)
        return list(reversed(records[-limit:]))

    def append_history(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append one sanitized JSONL task record."""
        payload = sanitize_for_storage(dict(record))
        self._assert_inside(self.history_path)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return payload

    def load_approvals(self) -> list[dict[str, Any]]:
        """Load valid approval objects, treating malformed storage as empty."""
        self._assert_inside(self.approvals_path)
        if not self.approvals_path.is_file():
            return []
        try:
            value = json.loads(self.approvals_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def save_approvals(self, approvals: Sequence[Mapping[str, Any]]) -> None:
        payload = sanitize_for_storage([dict(item) for item in approvals])
        self._write_json_atomic(self.approvals_path, payload)

    def approvals(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Compatibility view of approval records, optionally filtered by status."""
        approvals = self.load_approvals()
        if status is not None:
            approvals = [item for item in approvals if item.get("status") == status]
        return approvals

    def create_approval(
        self,
        *,
        task_id: str,
        action: str,
        reason: str,
        risk_level: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApprovalRequest:
        from datetime import UTC, datetime
        from uuid import uuid4

        from kodiak.orchestration.v1_models import ApprovalRequest

        now = datetime.now(UTC).isoformat()
        request = ApprovalRequest(
            approval_id=f"apr_{uuid4().hex[:12]}",
            task_id=task_id,
            action=action,
            reason=reason,
            risk_level=risk_level,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        approvals = self.load_approvals()
        approvals.append(request.to_dict())
        self.save_approvals(approvals)
        return request

    def update_approval(self, approval_id: str, status: str) -> dict[str, Any] | None:
        approvals = self.load_approvals()
        for item in approvals:
            if item.get("approval_id") != approval_id:
                continue
            item["status"] = status
            self.save_approvals(approvals)
            return item
        return None

    def update_approval_metadata(
        self, approval_id: str, values: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        approvals = self.load_approvals()
        for item in approvals:
            if item.get("approval_id") != approval_id:
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                item["metadata"] = metadata
            metadata.update(values)
            self.save_approvals(approvals)
            return item
        return None

    def save_run(self, task_id: str, record: Mapping[str, Any]) -> Path:
        """Save a sanitized per-run report under a traversal-safe filename."""
        safe_id = self._safe_task_id(task_id)
        path = self.runs_dir / f"{safe_id}.json"
        self._assert_inside(path)
        self._write_json_atomic(path, sanitize_for_storage(dict(record)))
        return path

    def load_run(self, task_id: str) -> dict[str, Any] | None:
        """Load one valid per-run report without allowing path traversal."""
        path = self.runs_dir / f"{self._safe_task_id(task_id)}.json"
        self._assert_inside(path)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def get_run(self, task_id: str) -> dict[str, Any] | None:
        """Compatibility alias for loading one task run."""
        return self.load_run(task_id)

    @staticmethod
    def _safe_task_id(task_id: str) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", task_id).strip(".-")
        if not safe_id or safe_id != task_id:
            raise ValueError("Task ID contains unsafe filename characters.")
        return safe_id

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        self._assert_inside(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _assert_inside(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing local-state access outside repository: {path}") from exc


def sanitize_for_storage(value: Any, *, field_name: str = "") -> Any:
    """Return JSON-compatible data with common secret fields and values redacted."""
    lowered = field_name.casefold()
    if any(fragment in lowered for fragment in _SECRET_FIELD_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_for_storage(item, field_name=str(key)) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_storage(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        clean = value
        for pattern in _SECRET_PATTERNS:
            clean = pattern.sub("[REDACTED]", clean)
        return clean
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)
