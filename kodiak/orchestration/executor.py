"""Backwards-compatibility shim for Kodiak's execution architecture refactor.

The execution engine implementation that previously lived in this module
was moved to :mod:`kodiak.orchestration.execution.engine` as part of the
execution architecture refactor. This module contains no execution logic
of its own; it exists solely to preserve legacy imports such as::

    from kodiak.orchestration.executor import ExecutionEngine

All behavior, state, and logic belong to the canonical engine module. This
shim only re-exports its public symbols and must not be extended with new
functionality, wrappers, or subclasses.
"""

from __future__ import annotations

from kodiak.orchestration.execution.engine import ExecutionEngine
from kodiak.orchestration.execution.exceptions import (
    ExecutionCancelledError,
    ExecutionEngineError,
    ExecutionTimeoutError,
    NonRetryableExecutionError,
    RetryExhaustedError,
)

__all__ = [
    "ExecutionEngine",
    "ExecutionEngineError",
    "ExecutionTimeoutError",
    "ExecutionCancelledError",
    "RetryExhaustedError",
    "NonRetryableExecutionError",
]
