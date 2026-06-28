from __future__ import annotations

import os
from typing import Any

from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "kodiak",
    broker=broker_url,
    backend=result_backend,
    include=[
        "kodiak.workers.tasks.memory_tasks",
        "kodiak.workers.tasks.repository_tasks",
        "kodiak.workers.tasks.learning_tasks",
        "kodiak.workers.tasks.maintenance_tasks",
        "kodiak.workers.tasks.metrics_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_extended=True,
    task_default_queue="default",
    task_queues={
        "default": {},
        "heavy": {},
        "maintenance": {},
    },
    task_default_exchange="tasks",
    task_default_routing_key="task.default",
)