"""Repository-local task API backed by the deterministic v1 service layer."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from kodiak.orchestration.local_storage import LocalStateStore
from kodiak.orchestration.task_orchestrator import InvalidRepositoryError, TaskOrchestrator

router = APIRouter(prefix="/tasks", tags=["local tasks"])


class LocalTaskRunRequest(BaseModel):
    instruction: str = Field(min_length=1)
    path: Path = Path(".")
    dry_run: bool = False
    no_commit: bool = False


@router.post("/run")
async def run_local_task(body: LocalTaskRunRequest) -> dict[str, Any]:
    """Run the same bounded local workflow exposed by ``kodiak task run``."""
    try:
        result = await run_in_threadpool(
            TaskOrchestrator().run,
            body.instruction,
            body.path,
            dry_run=body.dry_run,
            no_commit=body.no_commit,
        )
    except (InvalidRepositoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@router.get("/{task_id}")
async def get_local_task(
    task_id: str,
    path: Path = Query(Path("."), description="Repository path containing .kodiak state."),
) -> dict[str, Any]:
    """Return a persisted repository-local task run."""
    try:
        record = await run_in_threadpool(_load_local_task, path, task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Task run not found")
    return record


def _load_local_task(path: Path, task_id: str) -> dict[str, Any] | None:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")
    return LocalStateStore(root).load_run(task_id)
