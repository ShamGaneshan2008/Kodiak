from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from kodiak.api.schemas.common import PaginatedResponse
from kodiak.api.schemas.task import (
    TaskApprove,
    TaskCreate,
    TaskResponse,
    TaskStatusResponse,
    TaskUpdate,
)
from kodiak.db.models.project import Project
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kodiak.api.dependencies import CurrentUser, PaginationDep, get_db
from kodiak.db.models.task import Task, TaskStatus

router = APIRouter(prefix="/projects/{project_id}/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    project_id: uuid.UUID,
    body: TaskCreate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Task:
    await _assert_project_access(session, project_id, current_user.id)
    task = Task(
        project_id=project_id,
        created_by=current_user.id,
        title=body.title,
        description=body.description,
        task_type=body.task_type,
        priority=body.priority,
        github_issue_number=body.github_issue_number,
        status=TaskStatus.PENDING,
    )
    session.add(task)
    await session.flush()

    # Dispatch to Celery — import here to avoid circular at module load
    try:
        from kodiak.workers.tasks.run_task import run_task_async

        celery_result = run_task_async.delay(str(task.id), str(project_id))
        task.celery_task_id = celery_result.id
    except Exception:
        pass  # worker unavailable in test/dev; task remains PENDING

    return task


@router.get("", response_model=PaginatedResponse[TaskResponse])
async def list_tasks(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    pagination: PaginationDep,
    session: AsyncSession = Depends(get_db),
    status_filter: TaskStatus | None = None,
) -> PaginatedResponse[TaskResponse]:
    await _assert_project_access(session, project_id, current_user.id)
    base = select(Task).where(
        Task.project_id == project_id,
        Task.deleted_at.is_(None),
    )
    if status_filter:
        base = base.where(Task.status == status_filter)

    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    result = await session.execute(
        base.order_by(Task.created_at.desc()).offset(pagination.offset).limit(pagination.limit)
    )
    return PaginatedResponse.build(
        list(result.scalars().all()), total, pagination.page, pagination.page_size
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Task:
    return await _get_task(session, task_id, project_id, current_user.id)


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Task:
    return await _get_task(session, task_id, project_id, current_user.id)


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    body: TaskUpdate,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Task:
    task = await _get_task(session, task_id, project_id, current_user.id)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(task, field, value)
    return task


@router.post("/{task_id}/approve", response_model=TaskResponse)
async def approve_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    body: TaskApprove,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> Task:
    task = await _get_task(session, task_id, project_id, current_user.id)

    if task.status != TaskStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Task is not awaiting approval (current: {task.status})",
        )

    target = TaskStatus.APPROVED if body.approved else TaskStatus.REJECTED
    if not task.can_transition_to(target):
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Cannot transition to {target}")

    task.status = target
    if body.comment:
        task.metadata_["approval_comment"] = body.comment
    return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_task(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> None:
    task = await _get_task(session, task_id, project_id, current_user.id)
    if not task.can_transition_to(TaskStatus.CANCELLED):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel task in status {task.status}",
        )
    task.status = TaskStatus.CANCELLED


async def _get_task(
    session: AsyncSession,
    task_id: uuid.UUID,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Task:
    await _assert_project_access(session, project_id, user_id)
    result = await session.execute(
        select(Task).where(
            Task.id == task_id,
            Task.project_id == project_id,
            Task.deleted_at.is_(None),
        )
    )
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Task not found")
    return task


async def _assert_project_access(
    session: AsyncSession, project_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    result = await session.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == user_id,
            Project.deleted_at.is_(None),
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Project not found")
