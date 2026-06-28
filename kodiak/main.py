from __future__ import annotations

import asyncio
import logging
import signal
import sys
import types

import redis.asyncio as aioredis
import structlog

from kodiak.workers.beat_schedule import BEAT_SCHEDULE
from kodiak.workers.celery_app import celery_app

logger = structlog.get_logger(__name__)


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def validate_redis_connection() -> None:
    broker_url = celery_app.conf.broker_url
    if not broker_url:
        raise ValueError("CELERY_BROKER_URL is not configured")

    client = aioredis.from_url(broker_url)
    try:
        await client.ping()
        logger.info("redis_connection_validated", broker=broker_url)
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        raise
    finally:
        await client.aclose()


def validate_celery_config() -> None:
    celery_app.conf.beat_schedule = BEAT_SCHEDULE
    logger.info("celery_configuration_validated")


def start_worker() -> None:
    logger.info("starting_kodiak_worker")
    worker_args = [
        "worker",
        "--loglevel=info",
        "--queues=default,heavy,maintenance",
    ]
    celery_app.worker_main(worker_args)


def start_beat() -> None:
    logger.info("starting_kodiak_beat")
    beat_args = ["beat", "--loglevel=info"]
    celery_app.worker_main(beat_args)


def handle_shutdown(signum: int, frame: types.FrameType | None) -> None:
    logger.info("shutdown_signal_received", signal=signum)
    sys.exit(0)


def main() -> None:
    setup_logging()

    command = sys.argv[1] if len(sys.argv) > 1 else "worker"

    if command not in ("worker", "beat"):
        logger.error("invalid_command", command=command, valid_options=["worker", "beat"])
        sys.exit(1)

    try:
        asyncio.run(validate_redis_connection())
    except Exception:
        sys.exit(1)

    validate_celery_config()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if command == "worker":
        start_worker()
    else:
        start_beat()


if __name__ == "__main__":
    main()