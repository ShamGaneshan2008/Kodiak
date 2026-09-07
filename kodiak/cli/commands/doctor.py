"""CLI command for verifying the health of a Kodiak installation.

This module contains presentation logic only.

Health checks are NOT currently available from the CLI. The previous
implementation depended on ``kodiak.cli.services.doctor_service.DoctorService``,
which does not exist in this repository. There is no other existing
backend module that performs the Python/Git/Docker/API-key/connectivity/
GitHub-auth/configuration/workspace/sandbox checks this command
previously described. Rather than invent a replacement service, this
command reports the limitation clearly.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from importlib import metadata

import typer
from rich.console import Console
from rich.table import Table

console = Console()
error_console = Console(stderr=True)

app = typer.Typer()

_UNAVAILABLE_MESSAGE = (
    "Health checks are not available from the CLI. The backend service "
    "this command depended on (DoctorService) does not exist in this "
    "repository, and no other existing backend module performs these "
    "checks."
)


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit sanitized machine-readable JSON."),
) -> None:
    """Check the minimal installation without printing secret values."""
    python_ok = sys.version_info >= (3, 12)
    git_path = shutil.which("git")
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "local")
    provider_key = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }.get(provider.lower())
    provider_ready = provider_key is None or bool(os.getenv(provider_key))
    try:
        version = metadata.version("kodiak")
    except metadata.PackageNotFoundError:
        from kodiak import __version__

        version = __version__

    diagnostic = {
        "kodiak_version": version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": {"available": git_path is not None},
        "provider": {
            "name": provider,
            "configured": provider_ready,
            "required_variable": provider_key,
        },
        "optional_components": {"rag": _package_available("chromadb")},
    }
    if json_output:
        console.print_json(json.dumps(diagnostic))
    else:
        table = Table(title="Kodiak diagnostics")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        table.add_row("Kodiak", "OK", version)
        table.add_row("Python", "OK" if python_ok else "FAIL", platform.python_version())
        table.add_row("Git", "OK" if git_path else "FAIL", git_path or "Install Git and retry.")
        table.add_row(
            "Provider",
            "OK" if provider_ready else "OPTIONAL CONFIG MISSING",
            provider if provider_ready else f"Set {provider_key} to use {provider}.",
        )
        console.print(table)
    if not python_ok or git_path is None:
        raise typer.Exit(code=1)


def _package_available(package: str) -> bool:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        return False
    return True
