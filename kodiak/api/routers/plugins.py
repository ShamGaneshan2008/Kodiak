from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/plugins", tags=["plugins"])


class PluginView(BaseModel):
    id: str
    status: str = "ok"


@router.get("")
async def list_plugins() -> list[PluginView]:
    return []


@router.get("/{item_id}")
async def get_plugins(item_id: str) -> PluginView:
    return PluginView(id=item_id)
