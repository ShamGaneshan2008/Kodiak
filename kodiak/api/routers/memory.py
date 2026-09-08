from pathlib import Path

from fastapi import APIRouter, Query
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
async def task_history(limit: int = Query(10, ge=1, le=1000)) -> list[dict[str, object]]:
    """Return recent task history rooted at the server workspace."""
    return LocalStateStore(Path.cwd()).history(limit)
