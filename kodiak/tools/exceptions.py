"""Exceptions for the Kodiak Tool & Capability System."""

from __future__ import annotations


class ToolError(Exception):
    """Base exception for all tool system errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool cannot be found in the registry."""


class ToolRegistrationError(ToolError):
    """Raised when registering a tool fails (e.g., duplicate name or invalid metadata)."""


class ToolValidationError(ToolError):
    """Raised when tool input or parameters fail validation."""


class ToolPermissionError(ToolError):
    """Raised when tool execution is denied by the permission engine."""


class ToolTimeoutError(ToolError):
    """Raised when tool execution exceeds its allowed time limit."""


class ToolExecutionError(ToolError):
    """Raised when a tool encounters a runtime failure during execution."""


__all__ = [
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolValidationError",
    "ToolPermissionError",
    "ToolTimeoutError",
    "ToolExecutionError",
]
