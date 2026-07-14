"""Service layer for local Kodiak authentication state.

This module owns the on-disk storage of the CLI's authentication
credentials (the GitHub access token obtained via the OAuth flow in
:mod:`kodiak.auth.oauth`). It is a thin orchestration layer: it does not
perform any GitHub API calls or token exchange itself -- that remains the
responsibility of :mod:`kodiak.auth.oauth`. Its sole concern is reading,
writing, and removing the local credentials file used between CLI
invocations.

Credentials are stored as a single JSON file inside the user's Kodiak
config directory (``~/.kodiak/credentials.json`` by default), with file
permissions restricted to the owner only.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import structlog

__all__ = [
    "AuthService",
    "CredentialStorageError",
    "StoredCredentials",
]

logger = structlog.get_logger(__name__)

_DEFAULT_CONFIG_DIR: Final[Path] = Path.home() / ".kodiak"
_CREDENTIALS_FILENAME: Final[str] = "credentials.json"
_DIR_MODE: Final[int] = stat.S_IRWXU  # 0700
_FILE_MODE: Final[int] = stat.S_IRUSR | stat.S_IWUSR  # 0600


class CredentialStorageError(Exception):
    """Raised when stored credentials cannot be read, written, or removed."""


@dataclass
class StoredCredentials:
    """Local representation of a stored GitHub credential.

    Attributes:
        access_token: The GitHub access token used to authenticate API
            requests.
        username: The GitHub login associated with the token, if known.
        token_type: The OAuth token type (e.g. ``"bearer"``).
        scope: The space-delimited scopes granted to the token, if known.
    """

    access_token: str
    username: str | None = None
    token_type: str | None = None
    scope: str | None = None


class AuthService:
    """Manages local storage of the CLI's GitHub authentication state.

    This service reads and writes a single credentials file on disk. It is
    used by CLI commands such as ``kodiak login`` and ``kodiak logout`` to
    check whether a user is currently authenticated and to remove stored
    credentials on logout.

    Attributes:
        credentials_path: Absolute path to the credentials file on disk.
    """

    def __init__(self, credentials_path: Path | None = None) -> None:
        """Initialize the service.

        Args:
            credentials_path: Optional override for the credentials file
                location. Defaults to ``~/.kodiak/credentials.json``.
        """
        self.credentials_path: Path = (
            credentials_path or _DEFAULT_CONFIG_DIR / _CREDENTIALS_FILENAME
        )

    def has_stored_credentials(self) -> bool:
        """Check whether credentials are currently stored on disk.

        Returns:
            ``True`` if a non-empty credentials file exists, ``False``
            otherwise.
        """
        return self.credentials_path.is_file() and self.credentials_path.stat().st_size > 0

    def load_credentials(self) -> StoredCredentials | None:
        """Load stored credentials from disk.

        Returns:
            The stored credentials, or ``None`` if no credentials are
            stored.

        Raises:
            CredentialStorageError: If the credentials file exists but
                cannot be read or parsed.
        """
        if not self.has_stored_credentials():
            return None

        try:
            raw = self.credentials_path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise CredentialStorageError(
                f"Failed to read credentials file at {self.credentials_path}: {exc}"
            ) from exc

        try:
            return StoredCredentials(
                access_token=data["access_token"],
                username=data.get("username"),
                token_type=data.get("token_type"),
                scope=data.get("scope"),
            )
        except KeyError as exc:
            raise CredentialStorageError(
                f"Credentials file at {self.credentials_path} is missing required field: {exc}"
            ) from exc

    def store_credentials(self, credentials: StoredCredentials) -> None:
        """Persist credentials to disk, replacing any existing file.

        Args:
            credentials: The credentials to store.

        Raises:
            CredentialStorageError: If the credentials directory or file
                cannot be created or written.
        """
        try:
            self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.credentials_path.parent, _DIR_MODE)

            payload = {
                "access_token": credentials.access_token,
                "username": credentials.username,
                "token_type": credentials.token_type,
                "scope": credentials.scope,
            }

            self.credentials_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            os.chmod(self.credentials_path, _FILE_MODE)
        except OSError as exc:
            raise CredentialStorageError(
                f"Failed to write credentials file at {self.credentials_path}: {exc}"
            ) from exc

        logger.info("auth.credentials_stored", path=str(self.credentials_path))

    def clear_credentials(self) -> None:
        """Remove stored credentials from disk, if present.

        This is a no-op if no credentials file exists.

        Raises:
            CredentialStorageError: If the credentials file exists but
                cannot be removed.
        """
        if not self.credentials_path.exists():
            return

        try:
            self.credentials_path.unlink()
        except OSError as exc:
            raise CredentialStorageError(
                f"Failed to remove credentials file at {self.credentials_path}: {exc}"
            ) from exc

        logger.info("auth.credentials_cleared", path=str(self.credentials_path))