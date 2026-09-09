from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from kodiak.orchestration.local_storage import LocalStateStore

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryView(BaseModel):
    id: str
    status: str = "ok"


@router.get("")
async def list_memory() -> list[MemoryView]:
    return []


@router.get("/items/{item_id}")
async def get_memory(item_id: str) -> MemoryView:
    return MemoryView(id=item_id)


@router.get("/history")
async def task_history(
    limit: int = Query(10, ge=1, le=1000),
    path: Path = Query(Path("."), description="Repository path containing .kodiak state."),
) -> list[dict[str, object]]:
    """Return recent task history rooted at the server workspace."""
    try:
        return await run_in_threadpool(_task_history, path, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _task_history(path: Path, limit: int) -> list[dict[str, object]]:
    return LocalStateStore(path).history(limit)
