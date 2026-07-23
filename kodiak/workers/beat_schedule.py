from __future__ import annotations

from typing import Any

from celery.schedules import crontab

BEAT_SCHEDULE: dict[str, dict[str, Any]] = {
    "repository_sync": {
        "task": "kodiak.workers.tasks.repository_tasks.sync_repositories",
        "schedule": crontab(minute="*/15"),
        "kwargs": {"limit": 100},
        "queue": "default",
    },
    "memory_consolidation": {
        "task": "kodiak.workers.tasks.memory_tasks.consolidate_memory",
        "schedule": crontab(minute="0", hour="*/2"),
        "queue": "heavy",
    },
    "learning_feedback_collection": {
        "task": "kodiak.workers.tasks.learning_tasks.collect_feedback",
        "schedule": crontab(minute="0", hour="*"),
        "queue": "default",
    },
    "metrics_cleanup": {
        "task": "kodiak.workers.tasks.metrics_tasks.cleanup_metrics",
        "schedule": crontab(minute="0", hour="2"),
        "queue": "maintenance",
    },
    "health_check": {
        "task": "kodiak.workers.tasks.maintenance_tasks.run_health_checks",
        "schedule": crontab(minute="*/5"),
        "queue": "maintenance",
    },
    "task_cleanup": {
        "task": "kodiak.workers.tasks.maintenance_tasks.cleanup_stale_tasks",
        "schedule": crontab(minute="30", hour="3"),
        "queue": "maintenance",
    },
}
