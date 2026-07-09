"""Reusable filesystem path utilities for the Kodiak command-line interface.

This module provides small, composable helpers for resolving, expanding,
and validating filesystem paths used by CLI commands. It performs no
printing, prompting, or business logic — callers are responsible for
interpreting and acting on the results.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "resolve_repository_path",
    "expand_user_path",
    "ensure_directory_exists",
    "normalize_path",
    "is_git_repository",
]


def expand_user_path(path: str | os.PathLike[str]) -> Path:
    """Expand a user home reference and environment variables in a path.

    Args:
        path: A path that may contain a leading ``~`` or environment
            variable references (e.g. ``$HOME``, ``%USERPROFILE%``).

    Returns:
        The expanded `Path`, not resolved against the filesystem.
    """
    expanded = os.path.expandvars(str(path))
    return Path(expanded).expanduser()


def normalize_path(path: str | os.PathLike[str]) -> Path:
    """Normalize a path by expanding and resolving it into an absolute form.

    Args:
        path: The path to normalize.

    Returns:
        An absolute, normalized `Path`. The path need not exist on disk.
    """
    return expand_user_path(path).resolve()


def ensure_directory_exists(path: str | os.PathLike[str]) -> Path:
    """Ensure that a directory exists, creating it and any parents if needed.

    Args:
        path: The directory path to ensure exists.

    Returns:
        The normalized `Path` to the directory.

    Raises:
        NotADirectoryError: If `path` already exists but is not a
            directory.
        OSError: If the directory cannot be created due to a filesystem
            error (e.g. permissions).
    """
    normalized = normalize_path(path)

    if normalized.exists() and not normalized.is_dir():
        raise NotADirectoryError(f"Path exists and is not a directory: {normalized}")

    normalized.mkdir(parents=True, exist_ok=True)
    return normalized


def resolve_repository_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the path to a repository, defaulting to the current directory.

    Args:
        path: An optional path to a repository. When omitted, the current
            working directory is used.

    Returns:
        The normalized, absolute `Path` to the repository.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        NotADirectoryError: If the resolved path exists but is not a
            directory.
    """
    normalized = normalize_path(path) if path is not None else normalize_path(Path.cwd())

    if not normalized.exists():
        raise FileNotFoundError(f"Repository path does not exist: {normalized}")
    if not normalized.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {normalized}")

    return normalized


def is_git_repository(path: str | os.PathLike[str] | None = None) -> bool:
    """Determine whether a given path is the root of a Git repository.

    This performs a lightweight filesystem check for a ``.git`` entry and
    does not shell out to Git or inspect repository internals.

    Args:
        path: An optional path to check. When omitted, the current
            working directory is used.

    Returns:
        True if a ``.git`` directory or file is present at `path`, False
        otherwise.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        NotADirectoryError: If the resolved path exists but is not a
            directory.
    """
    normalized = resolve_repository_path(path)
    return (normalized / ".git").exists()