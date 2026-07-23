"""
kodiak/api/routers/projects.py
"""

from __future__ import annotations

import uuid
from datetime import UTC

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kodiak.api.dependencies import CurrentUser, PaginationDep
from kodiak.api.schemas.common import PaginatedResponse
from kodiak.api.schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate
from kodiak.db.models.project import Project
from kodiak.db.session import get_db

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Project:
    project = Project(
        owner_id=current_user.id,
        name=body.name,
        description=body.description,
        repo_url=body.repo_url,
    )
    session.add(project)
    await session.flush()
    return project


@router.get("", response_model=PaginatedResponse[ProjectResponse])
async def list_projects(
    current_user: CurrentUser,
    pagination: PaginationDep,
    session: AsyncSession = Depends(get_db),
) -> PaginatedResponse[Project]:
    base = select(Project).where(
        Project.owner_id == current_user.id,
        Project.deleted_at.is_(None),
    )
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    result = await session.execute(
        base.order_by(Project.created_at.desc()).offset(pagination.offset).limit(pagination.limit)
    )
    return PaginatedResponse[Project].build(
        list(result.scalars().all()), total, pagination.page, pagination.page_size
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Project:
    return await _get_project(session, project_id, current_user.id)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Project:
    project = await _get_project(session, project_id, current_user.id)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(project, field, value)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> None:
    project = await _get_project(session, project_id, current_user.id)
    from datetime import datetime

    project.deleted_at = datetime.now(UTC)


async def _get_project(session: AsyncSession, project_id: uuid.UUID, user_id: uuid.UUID) -> Project:
    result = await session.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == user_id,
            Project.deleted_at.is_(None),
        )
    )
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project
