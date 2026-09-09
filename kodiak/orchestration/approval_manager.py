"""File-backed approvals and v1 risk classification."""

from __future__ import annotations

import builtins
import hashlib
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


class ApprovalRecord(dict[str, Any]):
    """Dictionary approval record with the legacy attribute helpers."""

    @property
    def approval_id(self) -> str:
        return str(self["approval_id"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


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
    ) -> ApprovalRecord:
        now = datetime.now(UTC).isoformat()
        request = ApprovalRecord(
            {
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
        )
        approvals = self.store.load_approvals()
        approvals.append(request)
        self.store.save_approvals(approvals)
        return request

    def list(self, status: str | None = None) -> builtins.list[dict[str, Any]]:
        if status is not None and status not in APPROVAL_STATUSES:
            raise ValueError(f"Unknown approval status: {status}")
        approvals = self.store.load_approvals()
        if status is not None:
            approvals = [item for item in approvals if item.get("status") == status]
        return sorted(approvals, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get(self, approval_id: str) -> ApprovalRecord:
        for item in self.store.load_approvals():
            if item.get("approval_id") == approval_id:
                return ApprovalRecord(item)
        raise LookupError(f"Approval not found (Unknown approval ID): {approval_id}")

    def resolve(self, approval_id: str, status: str) -> ApprovalRecord:
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
            return ApprovalRecord(item)
        raise LookupError(f"Approval not found (Unknown approval ID): {approval_id}")

    @staticmethod
    def risk_reasons(
        *,
        action: str,
        paths: Iterable[str] = (),
        command: Iterable[str] = (),
    ) -> builtins.list[str]:
        """Describe every v1 policy condition requiring approval."""
        normalized_paths = [Path(path.replace("\\", "/")) for path in paths]
        reasons: builtins.list[str] = []
        action_key = action.casefold()
        if action_key in {"commit", "git_commit"}:
            reasons.append("creating a Git commit")
        if action_key in {"delete", "delete_file", "delete_files"}:
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
        if action_key == "risky_shell":
            reasons.append("running a risky shell command")
        if rendered and any(token in rendered for token in risky_tokens):
            reasons.append("running a risky shell command")
        return list(dict.fromkeys(reasons))

    @classmethod
    def requires_approval(
        cls,
        action: str,
        *,
        paths: Iterable[str] = (),
        command: Iterable[str] = (),
    ) -> bool:
        return bool(cls.risk_reasons(action=action, paths=paths, command=command))

    def execute_approved_commit(self, approval_id: str) -> str:
        """Execute an approved commit only while its file snapshot is unchanged."""
        approval = self.get(approval_id)
        if approval.get("status") != "approved":
            raise RuntimeError(f"Approval is not approved: {approval_id}")
        if approval.get("action") != "git_commit":
            raise ValueError("Only approved git_commit requests can be executed.")
        metadata = approval.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Approval metadata is malformed.")
        if metadata.get("executed_at"):
            raise RuntimeError(f"Approval was already executed: {approval_id}")

        files_value = metadata.get("files", metadata.get("paths"))
        hashes_value = metadata.get("after_hashes", metadata.get("sha256"))
        if not isinstance(files_value, list) or not isinstance(hashes_value, dict):
            raise ValueError("Approval does not contain a verifiable file snapshot.")
        files = [str(value) for value in files_value]
        for relative in files:
            target = (self.store.root / relative).resolve()
            try:
                target.relative_to(self.store.root)
            except ValueError as exc:
                raise ValueError(f"Approved file is outside the repository: {relative}") from exc
            if not target.is_file():
                raise ValueError(f"Approved file is missing: {relative}")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != hashes_value.get(relative):
                raise RuntimeError(f"Approved file changed after the request: {relative}")

        from kodiak.orchestration.local_agents import LocalGitAgent

        output = LocalGitAgent().commit(
            self.store.root,
            files,
            str(metadata.get("message", "kodiak: apply approved changes")),
        )
        self.store.update_approval_metadata(
            approval_id, {"executed_at": datetime.now(UTC).isoformat()}
        )
        return output
