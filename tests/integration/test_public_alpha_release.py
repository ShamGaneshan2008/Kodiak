"""Public-alpha CLI, version, documentation, and artifact-facing checks."""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from kodiak import __version__
from kodiak.agents.repository import RepositoryAnalyzerAgent
from kodiak.cli.app import app
from kodiak.config.settings import Settings

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


def test_top_level_help_and_doctor_are_usable() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "analyze" in help_result.stdout
    assert "doctor" in help_result.stdout
    doctor_result = runner.invoke(app, ["doctor"])
    assert doctor_result.exit_code == 0
    assert "Kodiak diagnostics" in doctor_result.stdout
    assert "Python" in doctor_result.stdout
    assert "Git" in doctor_result.stdout


def test_version_has_one_canonical_source() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "kodiak/__init__.py"
    assert Settings().APP_VERSION == __version__ == "0.1.0a1"


def test_documented_safe_quickstart_command(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = runner.invoke(app, ["analyze", "analyze", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert '"file_count": 1' in result.stdout


def test_release_documents_and_templates_exist() -> None:
    required = (
        "LICENSE",
        "CHANGELOG.md",
        "docs/PUBLIC_ALPHA.md",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/ISSUE_TEMPLATE/integration_proposal.md",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_example_configuration_contains_no_real_secret() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ghp_" not in example
    assert "sk-" not in example
    assert "BEGIN PRIVATE KEY" not in example


def test_sdist_excludes_environment_files() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "/.env*" in pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]


def test_repository_analysis_ignores_release_temp_environments(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".tmp" / "venv" / "Lib").mkdir(parents=True)
    (tmp_path / ".tmp" / "venv" / "Lib" / "dependency.py").write_text(
        "SHOULD_NOT_BE_SCANNED = True\n", encoding="utf-8"
    )
    (tmp_path / ".venv312" / "Lib").mkdir(parents=True)
    (tmp_path / ".venv312" / "Lib" / "versioned_dependency.py").write_text(
        "SHOULD_NOT_BE_SCANNED = True\n", encoding="utf-8"
    )
    analysis = RepositoryAnalyzerAgent()._scan(tmp_path)
    assert analysis.file_count == 1
    assert analysis.files[0].path == str(Path("src") / "app.py")
