from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from kodiak.orchestration.local_agents import local_agent_descriptors

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentView(BaseModel):
    id: str
    status: str = "ok"
    role: str
    description: str
    backend: str
    capabilities: list[str] = Field(default_factory=list)


@router.get("")
async def list_agents() -> list[AgentView]:
    return [
        AgentView(
            id=agent.name,
            status=agent.status,
            role=agent.role,
            description=agent.description,
            backend=agent.backend,
        )
        for agent in local_agent_descriptors()
    ]


@router.get("/{item_id}")
async def get_agents(item_id: str) -> AgentView:
    agent = next((item for item in local_agent_descriptors() if item.name == item_id), None)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentView(
        id=agent.name,
        status=agent.status,
        role=agent.role,
        description=agent.description,
        backend=agent.backend,
    )
