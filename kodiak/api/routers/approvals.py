from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kodiak.orchestration.approval_manager import ApprovalManager

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalView(BaseModel):
    id: str
    status: str = "ok"
    action: str = ""
    reason: str = ""


@router.get("")
async def list_approvals() -> list[ApprovalView]:
    return [
        ApprovalView(
            id=str(item.get("approval_id", "")),
            status=str(item.get("status", "pending")),
            action=str(item.get("action", "")),
            reason=str(item.get("reason", "")),
        )
        for item in ApprovalManager(Path.cwd()).list(status="pending")
    ]


@router.get("/{item_id}")
async def get_approvals(item_id: str) -> ApprovalView:
    item = next(
        (
            value
            for value in ApprovalManager(Path.cwd()).list(status=None)
            if value.get("approval_id") == item_id
        ),
        None,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return ApprovalView(
        id=item_id,
        status=str(item.get("status", "pending")),
        action=str(item.get("action", "")),
        reason=str(item.get("reason", "")),
    )


@router.post("/{approval_id}/approve")
async def approve_approval(approval_id: str) -> dict[str, object]:
    return _resolve(approval_id, "approved")


@router.post("/{approval_id}/reject")
async def reject_approval(approval_id: str) -> dict[str, object]:
    return _resolve(approval_id, "rejected")


def _resolve(approval_id: str, status: str) -> dict[str, object]:
    try:
        return ApprovalManager(Path.cwd()).resolve(approval_id, status)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
