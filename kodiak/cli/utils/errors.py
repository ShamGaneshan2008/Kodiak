"""Centralized CLI exception hierarchy for the Kodiak command-line interface.

This module defines a lightweight, reusable set of exception classes used
throughout the CLI layer to signal well-defined failure conditions. It
contains no business logic, I/O, or presentation concerns — exceptions
raised here are expected to be caught and rendered by the CLI's
presentation layer.
"""

from __future__ import annotations


class KodiakCLIError(Exception):
    """Base class for all exceptions raised within the Kodiak CLI layer.

    All CLI-specific exceptions should inherit from this class so callers
    can catch a single base type when handling CLI-layer failures.

    Attributes:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception with a descriptive message.

        Args:
            message: Human-readable description of the error.
        """
        self.message = message
        super().__init__(message)


class ValidationError(KodiakCLIError):
    """Raised when user-supplied input fails validation.

    Examples include malformed arguments, invalid option combinations, or
    values that fail schema or type checks before further processing.
    """


class ConfigurationError(KodiakCLIError):
    """Raised when CLI configuration is missing, malformed, or invalid.

    Examples include an unreadable configuration file, an invalid field
    value, or a required configuration entry that has not been set.
    """


class RepositoryError(KodiakCLIError):
    """Raised when an operation involving a repository cannot proceed.

    Examples include an invalid repository path, an unresolvable
    repository reference, or a repository state that prevents the
    requested operation.
    """


class AuthenticationError(KodiakCLIError):
    """Raised when authentication or authorization fails.

    Examples include a missing or invalid API key, an expired credential,
    or insufficient permissions for the requested operation.
    """


class ServiceError(KodiakCLIError):
    """Raised when a downstream service invoked by the CLI fails.

    Examples include a service returning an unexpected response, a
    service being unreachable, or a service reporting an internal
    failure that the CLI cannot recover from.
    """


class CommandExecutionError(KodiakCLIError):
    """Raised when a CLI command fails to execute successfully.

    This is typically used as a catch-all for command-level failures that
    do not map cleanly to one of the more specific exception types above.
    """