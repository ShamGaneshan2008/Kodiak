"""
kodiak/api/schemas/task.py

Pydantic schemas for the Task resource, derived from the tasks router
(kodiak/api/routers/tasks.py) as the source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from kodiak.db.models.task import TaskPriority, TaskStatus


class TaskCreate(BaseModel):
    title: str
    description: str | None = None
    task_type: str
    priority: TaskPriority = TaskPriority.MEDIUM
    github_issue_number: int | None = None


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    task_type: str | None = None
    priority: TaskPriority | None = None
    github_issue_number: int | None = None


class TaskApprove(BaseModel):
    approved: bool
    comment: str | None = None


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    created_by: uuid.UUID | None
    title: str
    description: str | None
    task_type: str
    status: TaskStatus
    priority: TaskPriority
    github_issue_number: int | None
    celery_task_id: str | None
    created_at: datetime
    updated_at: datetime


class TaskStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: TaskStatus