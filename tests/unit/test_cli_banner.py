from __future__ import annotations

from io import BytesIO, StringIO, TextIOWrapper

from rich.console import Console
from typer.testing import CliRunner

from kodiak.cli.app import app
from kodiak.cli.ui.banner import get_banner, print_banner, should_show_banner

runner = CliRunner()


def test_full_banner_preserves_logo_geometry() -> None:
    banner = get_banner()

    assert "◉" in banner
    assert banner.count("●") == 4
    assert "╭╱" in banner and "╲╮" in banner
    assert "kodiak" in banner
    assert "Local AI software engineering agent" in banner


def test_compact_banner_exists_and_is_smaller() -> None:
    compact = get_banner(compact=True)

    assert "‹  ◉  ›" in compact
    assert compact.count("●") == 4
    assert len(compact.splitlines()) < len(get_banner().splitlines())


def test_print_banner_renders_plain_text_without_color_dependency() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=80)

    print_banner(console)

    output = stream.getvalue()
    assert "◉" in output
    assert "kodiak" in output
    assert "\x1b[" not in output


def test_print_banner_falls_back_safely_on_legacy_windows_encoding() -> None:
    buffer = BytesIO()
    stream = TextIOWrapper(buffer, encoding="cp1252")
    console = Console(file=stream, force_terminal=False, width=80)

    print_banner(console)
    stream.flush()

    output = buffer.getvalue().decode("cp1252")
    assert "+/" in output
    assert "O" in output
    assert "kodiak" in output


def test_banner_suppression_rules(monkeypatch) -> None:
    monkeypatch.delenv("KODIAK_NO_BANNER", raising=False)
    assert should_show_banner()
    assert not should_show_banner(json_mode=True)
    assert not should_show_banner(no_banner=True)

    monkeypatch.setenv("KODIAK_NO_BANNER", "1")
    assert not should_show_banner()


def test_no_banner_global_option_suppresses_startup_banner() -> None:
    result = runner.invoke(app, ["--no-banner", "agents", "list"])

    assert result.exit_code == 0
    assert "Local AI software engineering agent" not in result.stdout


def test_environment_suppresses_startup_banner(monkeypatch) -> None:
    monkeypatch.setenv("KODIAK_NO_BANNER", "1")

    result = runner.invoke(app, ["agents", "list"])

    assert result.exit_code == 0
    assert "Local AI software engineering agent" not in result.stdout


def test_json_mode_suppresses_startup_banner() -> None:
    result = runner.invoke(app, ["agents", "list", "--json"])

    assert result.exit_code == 0
    assert "Local AI software engineering agent" not in result.stdout


def test_normal_command_shows_banner_once() -> None:
    result = runner.invoke(app, ["agents", "list"])

    assert result.exit_code == 0
    assert result.stdout.count("Local AI software engineering agent") == 1
