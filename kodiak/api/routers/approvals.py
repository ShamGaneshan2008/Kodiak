from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalView(BaseModel):
    id: str
    status: str = "ok"


@router.get("")
async def list_approvals() -> list[ApprovalView]:
    return []


@router.get("/{item_id}")
async def get_approvals(item_id: str) -> ApprovalView:
    return ApprovalView(id=item_id)
