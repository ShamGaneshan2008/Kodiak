"""
Structured logging configuration using structlog.
Outputs JSON in production, pretty-printed console in development.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from kodiak.config.settings import Settings


def _add_app_context(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject static app-level fields into every log record."""
    from kodiak.config.settings import get_settings

    settings = get_settings()
    event_dict.setdefault("app", settings.APP_NAME)
    event_dict.setdefault("env", settings.ENVIRONMENT.value)
    event_dict.setdefault("version", settings.APP_VERSION)
    return event_dict


def configure_logging(settings: Settings | None = None) -> None:
    """
    Call once at application startup (main.py / worker entrypoint).
    Idempotent — safe to call multiple times.
    """
    if settings is None:
        from kodiak.config.settings import get_settings

        settings = get_settings()

    log_level: str = settings.LOG_LEVEL.value
    is_json: bool = settings.LOG_FORMAT == "json"

    # ── Shared processors (run for every log record) ──────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_app_context,
    ]

    # ── structlog configuration ───────────────────────────────────────────
    if is_json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level)),
        cache_logger_on_first_use=True,
    )

    # ── stdlib logging → structlog bridge ────────────────────────────────
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # ── Silence noisy third-party loggers ────────────────────────────────
    for noisy in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "openai._base_client",
        "sqlalchemy.engine",
        "celery.app.trace",
        "celery.worker.strategy",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.get_logger(__name__).info(
        "logging_configured",
        level=log_level,
        format=settings.LOG_FORMAT,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Factory used throughout the codebase.

    Usage::

        from kodiak.config.logging import get_logger
        log = get_logger(__name__)
        log.info("task_started", task_id=task_id)
    """
    return structlog.get_logger(name)
