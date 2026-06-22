from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/repositories", tags=["repositories"])


class RepositoryView(BaseModel):
    id: str
    status: str = "ok"


@router.get("")
async def list_repositories() -> list[RepositoryView]:
    return []


@router.get("/{item_id}")
async def get_repositories(item_id: str) -> RepositoryView:
    return RepositoryView(id=item_id)
