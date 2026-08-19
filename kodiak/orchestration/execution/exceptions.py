"""
kodiak/execution/exceptions.py

Exception taxonomy for the Execution Engine. The engine itself does not
raise these for ordinary execution failures — `execute()` always returns
an `ExecutionResult` — but it raises/uses them internally to drive
control flow, and callers (Supervisor, tests, Agent Manager
implementations) can raise or catch them explicitly.
"""

from __future__ import annotations


class ExecutionEngineError(Exception):
    """Base class for all Execution Engine failures."""


class ExecutionTimeoutError(ExecutionEngineError):
    """Raised internally when a single attempt exceeds its allotted timeout."""


class ExecutionCancelledError(ExecutionEngineError):
    """Raised internally when execution is cancelled cooperatively."""


class RetryExhaustedError(ExecutionEngineError):
    """Raised by callers that want exception-based flow after retries are exhausted.

    The engine itself signals this case via `ExecutionOutcome.RETRY_EXHAUSTED`
    on the returned `ExecutionResult` rather than raising, but a Supervisor
    that prefers try/except control flow can raise this from the result.
    """

    def __init__(
        self, message: str, attempts: int, last_error: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class NonRetryableExecutionError(ExecutionEngineError):
    """Wraps an underlying exception explicitly marked as non-retryable.

    Agent Manager implementations can raise this to short-circuit the
    retry loop for errors that are known to be futile to retry (e.g. bad
    credentials, malformed task input) regardless of the configured
    `RetryPolicy`.
    """

    def __init__(self, message: str, cause: BaseException) -> None:
        super().__init__(message)
        self.cause = cause
