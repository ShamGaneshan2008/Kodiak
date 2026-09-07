"""Contract tests for the canonical release check."""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

from kodiak.evaluation.release_check import _run, audit_artifacts


def test_artifact_audit_detects_environment_file(tmp_path: Path) -> None:
    wheel = tmp_path / "sample.whl"
    sdist = tmp_path / "sample.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kodiak/__init__.py", "")
    with tarfile.open(sdist, "w:gz") as archive:
        environment = tmp_path / ".env"
        environment.write_text("TOKEN=example", encoding="utf-8")
        archive.add(environment, arcname="sample/.env")
    assert audit_artifacts(wheel, sdist) == ["sdist:sample/.env"]


def test_artifact_audit_accepts_clean_archives(tmp_path: Path) -> None:
    wheel = tmp_path / "sample.whl"
    sdist = tmp_path / "sample.tar.gz"
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1", encoding="utf-8")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(source, "kodiak/module.py")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="sample/kodiak/module.py")
    assert audit_artifacts(wheel, sdist) == []


def test_release_subprocess_timeout_is_a_normalized_failure(monkeypatch, tmp_path: Path) -> None:
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", time_out)

    result = _run("bounded", ["example"], tmp_path, timeout_seconds=0.01)

    assert not result.passed
    assert result.returncode == 124
