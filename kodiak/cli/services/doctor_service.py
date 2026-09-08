"""Sanitized diagnostics for the Kodiak CLI."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    name: str
    status: str
    detail: str
    required: bool = False


class DoctorService:
    """Inspect local prerequisites without ever reading provider key values."""

    def __init__(self, cwd: str | Path | None = None) -> None:
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self._checks: list[DiagnosticCheck] = []
        self._provider = {name: "missing" for name in PROVIDER_KEYS}

    def run(self) -> list[DiagnosticCheck]:
        self._checks = []
        version_ok = sys.version_info >= (3, 12)
        self._add(
            "Python",
            version_ok,
            platform.python_version(),
            required=True,
        )
        self._check_import("Kodiak import", "kodiak", required=True)
        self._check_import("CLI import", "kodiak.cli.app", required=True)
        self._add("Current directory", self.cwd.is_dir(), str(self.cwd), required=True)

        git_path = shutil.which("git")
        self._add("Git", bool(git_path), git_path or "not found")
        self._add("Git repository", self._is_git_repo(git_path), str(self.cwd))
        pytest_available = importlib.util.find_spec("pytest") is not None
        ruff_available = importlib.util.find_spec("ruff") is not None
        self._add("pytest", pytest_available, "available" if pytest_available else "missing")
        self._add("Ruff", ruff_available, "available" if ruff_available else "missing")
        docker_path = shutil.which("docker")
        self._add("Docker", bool(docker_path), docker_path or "optional; not found")
        state_dir = self.cwd / ".kodiak"
        state_status = "present" if state_dir.is_dir() else "not created yet"
        self._checks.append(DiagnosticCheck(".kodiak folder", "pass", state_status))

        self._provider = {
            name: "present" if bool(os.environ.get(name)) else "missing" for name in PROVIDER_KEYS
        }
        configured = [name for name, status in self._provider.items() if status == "present"]
        self._checks.append(
            DiagnosticCheck(
                "Provider keys",
                "pass",
                f"{len(configured)} present; optional for local v1",
            )
        )
        self._checks.append(
            DiagnosticCheck(
                "Platform",
                "pass",
                f"{platform.system()} {platform.release()} ({platform.machine()})",
            )
        )
        return list(self._checks)

    def as_dict(self) -> dict[str, Any]:
        if not self._checks:
            self.run()
        return {
            "checks": [asdict(item) for item in self._checks],
            "provider": {
                "configured": any(status == "present" for status in self._provider.values()),
                "keys": dict(self._provider),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        }

    def _check_import(self, name: str, module: str, *, required: bool) -> None:
        try:
            importlib.import_module(module)
        except Exception as exc:
            self._checks.append(DiagnosticCheck(name, "fail", type(exc).__name__, required))
        else:
            self._checks.append(DiagnosticCheck(name, "pass", module, required))

    def _is_git_repo(self, git_path: str | None) -> bool:
        if not git_path:
            return False
        try:
            result = subprocess.run(
                [git_path, "rev-parse", "--is-inside-work-tree"],
                cwd=self.cwd,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.strip().casefold() == "true"

    def _add(self, name: str, passed: bool, detail: str, *, required: bool = False) -> None:
        status = "pass" if passed else "fail" if required else "missing"
        self._checks.append(DiagnosticCheck(name, status, detail, required))
