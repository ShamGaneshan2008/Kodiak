"""Canonical local release-readiness check for maintainers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    command: tuple[str, ...]
    returncode: int


def audit_artifacts(wheel: Path, sdist: Path) -> list[str]:
    """Return forbidden archive entries without extracting untrusted paths."""
    forbidden = (".env", "__pycache__", ".pytest_cache", ".tmp/", "evaluation_reports/")
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_names = archive.getnames()
    return sorted(
        f"{kind}:{name}"
        for kind, names in (("wheel", wheel_names), ("sdist", sdist_names))
        for name in names
        if any(marker in name for marker in forbidden)
    )


def _run(
    name: str,
    command: list[str],
    root: Path,
    *,
    timeout_seconds: float = 900,
) -> CheckResult:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(name, False, tuple(command), 124)
    return CheckResult(name, completed.returncode == 0, tuple(command), completed.returncode)


def run_release_check(root: Path, *, full: bool = True) -> dict[str, object]:
    """Run quality, package, clean-install, and safe-user-flow release gates."""
    root = root.resolve()
    temp_root = root / ".tmp"
    temp_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="release-check-", dir=temp_root))
    dist = work / "dist"
    venv = work / "venv"
    python = Path(sys.executable)
    checks: list[CheckResult] = []

    commands = [
        ("ruff", [str(python), "-m", "ruff", "check", "."]),
        ("format", [str(python), "-m", "ruff", "format", "--check", "."]),
        ("alpha_suite", [str(python), "-m", "kodiak.evaluation.alpha_suite"]),
    ]
    if full:
        commands[2:2] = [
            (
                "unit_tests",
                [
                    str(python),
                    "-m",
                    "pytest",
                    "tests/unit",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={work / 'unit-temp'}",
                ],
            ),
            (
                "integration_tests",
                [
                    str(python),
                    "-m",
                    "pytest",
                    "tests/integration",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={work / 'integration-temp'}",
                ],
            ),
        ]
    for name, command in commands:
        result = _run(name, command, root)
        checks.append(result)
        if not result.passed:
            return _summary(checks, artifact_violations=[])

    build = _run("package_build", [str(python), "-m", "build", "--outdir", str(dist)], root)
    checks.append(build)
    if not build.passed:
        return _summary(checks, artifact_violations=[])

    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    violations = audit_artifacts(wheel, sdist)
    checks.append(CheckResult("artifact_audit", not violations, (), int(bool(violations))))
    if violations:
        return _summary(checks, artifact_violations=violations)

    venv_result = _run("create_venv", [str(python), "-m", "venv", str(venv)], root)
    checks.append(venv_result)
    if not venv_result.passed:
        return _summary(checks, artifact_violations=[])

    installed_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    installed_cli = venv / ("Scripts/kodiak.exe" if sys.platform == "win32" else "bin/kodiak")
    install = _run(
        "wheel_install", [str(installed_python), "-m", "pip", "install", str(wheel)], root
    )
    checks.append(install)
    if not install.passed:
        return _summary(checks, artifact_violations=[])

    for name, command in (
        ("cli_help", [str(installed_cli), "--help"]),
        ("doctor", [str(installed_cli), "doctor", "--json"]),
        (
            "quickstart",
            [str(installed_cli), "analyze", "analyze", "examples/quickstart", "--json"],
        ),
        ("installed_alpha_suite", [str(installed_python), "-m", "kodiak.evaluation.alpha_suite"]),
    ):
        result = _run(name, command, root)
        checks.append(result)
        if not result.passed:
            break
    return _summary(checks, artifact_violations=[])


def _summary(checks: list[CheckResult], artifact_violations: list[str]) -> dict[str, object]:
    return {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "artifact_violations": artifact_violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kodiak's canonical release checks.")
    parser.add_argument("--smoke", action="store_true", help="Skip full unit/integration suites.")
    args = parser.parse_args()
    report = run_release_check(Path.cwd(), full=not args.smoke)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
