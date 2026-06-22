from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryView(BaseModel):
    id: str
    status: str = "ok"


@router.get("")
async def list_memory() -> list[MemoryView]:
    return []


@router.get("/{item_id}")
async def get_memory(item_id: str) -> MemoryView:
    return MemoryView(id=item_id)
