"""File-backed approvals and v1 risk classification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from kodiak.orchestration.local_storage import LocalStateStore

APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected"})
_DEPENDENCY_FILES = frozenset(
    {
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "pom.xml",
        "go.mod",
        "cargo.toml",
    }
)
_LOCK_SUFFIXES = (".lock", "-lock.json", ".lockb")


class ApprovalManager:
    """Create and resolve approval requests scoped to one repository."""

    def __init__(self, repository_path: str | Path) -> None:
        self.store = LocalStateStore(repository_path)

    def create(
        self,
        *,
        task_id: str,
        action: str,
        reason: str,
        risk_level: str = "high",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        request = {
            "approval_id": f"apr_{uuid4().hex[:12]}",
            "task_id": task_id,
            "action": action,
            "reason": reason,
            "risk_level": risk_level,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "metadata": dict(metadata or {}),
        }
        approvals = self.store.load_approvals()
        approvals.append(request)
        self.store.save_approvals(approvals)
        return request

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in APPROVAL_STATUSES:
            raise ValueError(f"Unknown approval status: {status}")
        approvals = self.store.load_approvals()
        if status is not None:
            approvals = [item for item in approvals if item.get("status") == status]
        return sorted(approvals, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get(self, approval_id: str) -> dict[str, Any]:
        for item in self.store.load_approvals():
            if item.get("approval_id") == approval_id:
                return item
        raise LookupError(f"Unknown approval ID: {approval_id}")

    def resolve(self, approval_id: str, status: str) -> dict[str, Any]:
        if status not in {"approved", "rejected"}:
            raise ValueError("Approval can only be resolved as approved or rejected.")
        approvals = self.store.load_approvals()
        for item in approvals:
            if item.get("approval_id") != approval_id:
                continue
            if item.get("status") != "pending":
                raise RuntimeError(
                    f"Approval {approval_id} is already {item.get('status', 'resolved')}."
                )
            item["status"] = status
            item["updated_at"] = datetime.now(UTC).isoformat()
            self.store.save_approvals(approvals)
            return item
        raise LookupError(f"Unknown approval ID: {approval_id}")

    @staticmethod
    def risk_reasons(
        *,
        action: str,
        paths: Iterable[str] = (),
        command: Iterable[str] = (),
    ) -> list[str]:
        """Describe every v1 policy condition requiring approval."""
        normalized_paths = [Path(path.replace("\\", "/")) for path in paths]
        reasons: list[str] = []
        action_key = action.casefold()
        if action_key in {"commit", "git_commit"}:
            reasons.append("creating a Git commit")
        if action_key in {"delete", "delete_files"}:
            reasons.append("deleting files")
        if action_key in {"push", "git_push"}:
            reasons.append("attempting a remote push (disabled in v1)")
        if len(normalized_paths) > 5:
            reasons.append("editing more than five files")
        for path in normalized_paths:
            lowered = path.as_posix().casefold()
            name = path.name.casefold()
            if name in _DEPENDENCY_FILES:
                reasons.append(f"changing dependency file {path.as_posix()}")
            if name.endswith(_LOCK_SUFFIXES):
                reasons.append(f"changing lock file {path.as_posix()}")
            if lowered.startswith(".github/workflows/"):
                reasons.append(f"changing CI/CD workflow {path.as_posix()}")
            if any(part.casefold() in {"auth", "security"} for part in path.parts):
                reasons.append(f"changing auth/security code {path.as_posix()}")
            if "migration" in lowered or lowered.startswith("alembic/versions/"):
                reasons.append(f"changing database migration {path.as_posix()}")
        risky_tokens = {
            "del",
            "format",
            "git clean",
            "git push",
            "git reset",
            "remove-item",
            "rm",
        }
        rendered = " ".join(command).casefold()
        if rendered and any(token in rendered for token in risky_tokens):
            reasons.append("running a risky shell command")
        return list(dict.fromkeys(reasons))
