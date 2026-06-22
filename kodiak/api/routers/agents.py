from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentView(BaseModel):
    id: str
    status: str = "ok"


@router.get("")
async def list_agents() -> list[AgentView]:
    return []


@router.get("/{item_id}")
async def get_agents(item_id: str) -> AgentView:
    return AgentView(id=item_id)
