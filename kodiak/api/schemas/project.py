"""
kodiak/api/schemas/project.py

Pydantic schemas for the Project resource, derived from the projects router
(kodiak/api/routers/projects.py) and the Project model as source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from kodiak.db.models.project import ProjectStatus


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    repo_url: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    repo_url: str | None = None
    status: ProjectStatus | None = None


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    description: str | None
    repo_url: str | None
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime