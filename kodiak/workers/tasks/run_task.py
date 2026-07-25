from __future__ import annotations

from kodiak.workers.celery_app import celery_app


@celery_app.task
def run_task_async(task_id: str, project_id: str):
    print(f"Running task {task_id} for project {project_id}")

    return {
        "task_id": task_id,
        "project_id": project_id,
        "status": "completed",
    }
