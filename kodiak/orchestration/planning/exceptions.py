"""
Exceptions for the Kodiak Planning subsystem.
"""

from __future__ import annotations

from typing import Any, Sequence


class PlanningError(Exception):
    """Base exception for all planning errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DependencyCycleError(PlanningError):
    """Raised when a dependency cycle is detected in task dependencies."""

    def __init__(
        self,
        message: str,
        cycle_path: Sequence[str] | Sequence[Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = dict(details or {})
        if cycle_path is not None:
            merged_details["cycle_path"] = [str(node) for node in cycle_path]
        super().__init__(message, details=merged_details)
        self.cycle_path: list[str] = [str(node) for node in (cycle_path or [])]


class PlanValidationError(PlanningError):
    """Raised when a plan fails validation."""

    def __init__(
        self,
        message: str,
        errors: Sequence[str] | None = None,
        warnings: Sequence[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = dict(details or {})
        if errors is not None:
            merged_details["errors"] = list(errors)
        if warnings is not None:
            merged_details["warnings"] = list(warnings)
        super().__init__(message, details=merged_details)
        self.errors: list[str] = list(errors or [])
        self.warnings: list[str] = list(warnings or [])


class ReplanningError(PlanningError):
    """Raised when dynamic replanning fails to produce a valid updated plan."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)
