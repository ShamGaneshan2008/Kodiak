"""Filesystem artifact and file-change verification."""

from __future__ import annotations

import time
from pathlib import Path

from kodiak.orchestration.verification.base import Verifier
from kodiak.orchestration.verification.models import (
    VerificationContext,
    VerificationEvidence,
    VerificationStatus,
)


class FileVerifier(Verifier):
    """Verify expected files exist and unexpected files were not modified."""

    name = "file"

    def applies(self, context: VerificationContext) -> bool:
        criteria = context.success_criteria
        return bool(
            criteria.get("expected_files")
            or criteria.get("required_artifacts")
            or criteria.get("unexpected_files")
        )

    async def verify(self, context: VerificationContext) -> VerificationEvidence:
        start = time.monotonic()
        root = context.workspace_root or Path.cwd()
        criteria = context.success_criteria

        expected = list(criteria.get("expected_files") or [])
        artifacts = list(criteria.get("required_artifacts") or [])
        unexpected = set(criteria.get("unexpected_files") or [])

        checked: list[str] = []
        missing: list[str] = []

        for rel_path in expected + artifacts:
            path = Path(rel_path)
            if not path.is_absolute():
                path = root / path
            checked.append(str(path))
            if not path.exists():
                missing.append(str(rel_path))

        if missing:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=f"Missing expected files: {', '.join(missing)}",
                files_checked=tuple(checked),
                artifacts_checked=tuple(artifacts),
            )

        changed_unexpected = [
            rel for rel in unexpected if (root / rel).exists()
        ]
        if changed_unexpected:
            return VerificationEvidence(
                verifier=self.name,
                status=VerificationStatus.FAILED,
                duration_seconds=time.monotonic() - start,
                message=f"Unexpected files present: {', '.join(changed_unexpected)}",
                files_checked=tuple(checked),
                metadata={"unexpected_files": changed_unexpected},
            )

        return VerificationEvidence(
            verifier=self.name,
            status=VerificationStatus.VERIFIED,
            duration_seconds=time.monotonic() - start,
            message="All expected files are present.",
            files_checked=tuple(checked),
            artifacts_checked=tuple(artifacts),
        )


__all__ = ["FileVerifier"]
