"""Reusable validation helpers shared across Kodiak CLI commands.

This module provides small, composable validation functions used to
verify user-supplied input before it is passed to services or agents.
It performs no I/O beyond filesystem existence checks and raises the
shared CLI exception types defined in `kodiak.cli.utils.errors`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from kodiak.cli.utils.errors import RepositoryError, ValidationError

__all__ = [
    "validate_non_empty_string",
    "validate_directory",
    "validate_repository_path",
    "validate_issue_number",
    "validate_branch_name",
]

_BRANCH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
_BRANCH_NAME_FORBIDDEN_SEQUENCES = ("..", "//", "@{")


def validate_non_empty_string(value: str, field_name: str) -> str:
    """Validate that a string value is non-empty after stripping whitespace.

    Args:
        value: The string to validate.
        field_name: Human-readable name of the field, used in error
            messages.

    Returns:
        The stripped, validated string.

    Raises:
        ValidationError: If `value` is empty or consists only of
            whitespace.
    """
    stripped = value.strip()
    if not stripped:
        raise ValidationError(f"'{field_name}' must not be empty.")
    return stripped


def validate_directory(path: str | os.PathLike[str], field_name: str = "path") -> Path:
    """Validate that a path exists and is a directory.

    Args:
        path: The path to validate.
        field_name: Human-readable name of the field, used in error
            messages.

    Returns:
        The validated `Path`, expanded and resolved.

    Raises:
        ValidationError: If the path does not exist or is not a
            directory.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise ValidationError(f"'{field_name}' does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValidationError(f"'{field_name}' is not a directory: {resolved}")

    return resolved


def validate_repository_path(path: str | os.PathLike[str]) -> Path:
    """Validate that a path points to a usable Git repository directory.

    Args:
        path: The candidate repository path.

    Returns:
        The validated, resolved `Path` to the repository.

    Raises:
        ValidationError: If the path does not exist or is not a
            directory.
        RepositoryError: If the path exists but does not contain a
            ``.git`` entry.
    """
    resolved = validate_directory(path, field_name="repository path")

    if not (resolved / ".git").exists():
        raise RepositoryError(f"Not a Git repository: {resolved}")

    return resolved


def validate_issue_number(value: int | str) -> int:
    """Validate that a value represents a positive issue number.

    Args:
        value: The candidate issue number, as an integer or numeric
            string.

    Returns:
        The validated issue number as an `int`.

    Raises:
        ValidationError: If `value` cannot be parsed as an integer or is
            not positive.
    """
    try:
        issue_number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid issue number: '{value}'.") from exc

    if issue_number <= 0:
        raise ValidationError(f"Issue number must be positive, got: {issue_number}.")

    return issue_number


def validate_branch_name(name: str) -> str:
    """Validate that a string is a syntactically valid Git branch name.

    This performs a lightweight, conservative syntax check and does not
    consult an actual Git repository.

    Args:
        name: The candidate branch name.

    Returns:
        The validated, stripped branch name.

    Raises:
        ValidationError: If `name` is empty, contains disallowed
            characters or sequences, or has an invalid leading/trailing
            character.
    """
    stripped = validate_non_empty_string(name, field_name="branch name")

    if stripped.startswith("/") or stripped.endswith("/"):
        raise ValidationError(f"Branch name must not start or end with '/': '{stripped}'")
    if stripped.endswith("."):
        raise ValidationError(f"Branch name must not end with '.': '{stripped}'")
    if any(sequence in stripped for sequence in _BRANCH_NAME_FORBIDDEN_SEQUENCES):
        raise ValidationError(f"Branch name contains an invalid sequence: '{stripped}'")
    if not _BRANCH_NAME_PATTERN.match(stripped):
        raise ValidationError(f"Branch name contains invalid characters: '{stripped}'")

    return stripped
