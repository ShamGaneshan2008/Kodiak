from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from kodiak.orchestration.approval_manager import ApprovalManager

router = APIRouter(prefix="/approvals", tags=["approvals"])


class ApprovalView(BaseModel):
    approval_id: str
    task_id: str
    action: str
    risk_level: str
    reason: str
    status: str
    created_at: str
    updated_at: str


@router.get("")
async def list_approvals(
    path: Path = Query(Path("."), description="Repository path containing .kodiak state."),
) -> list[ApprovalView]:
    try:
        return [_approval_view(item) for item in ApprovalManager(path).list(status=None)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{item_id}")
async def get_approvals(
    item_id: str,
    path: Path = Query(Path("."), description="Repository path containing .kodiak state."),
) -> ApprovalView:
    try:
        return _approval_view(ApprovalManager(path).get(item_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{approval_id}/approve")
async def approve_approval(
    approval_id: str,
    path: Path = Query(Path("."), description="Repository path containing .kodiak state."),
) -> dict[str, object]:
    return _resolve(path, approval_id, "approved")


@router.post("/{approval_id}/reject")
async def reject_approval(
    approval_id: str,
    path: Path = Query(Path("."), description="Repository path containing .kodiak state."),
) -> dict[str, object]:
    return _resolve(path, approval_id, "rejected")


def _resolve(path: Path, approval_id: str, status: str) -> dict[str, object]:
    try:
        return ApprovalManager(path).resolve(approval_id, status)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _approval_view(item: dict[str, object]) -> ApprovalView:
    return ApprovalView(
        approval_id=str(item.get("approval_id", "")),
        task_id=str(item.get("task_id", "")),
        action=str(item.get("action", "")),
        risk_level=str(item.get("risk_level", "")),
        reason=str(item.get("reason", "")),
        status=str(item.get("status", "pending")),
        created_at=str(item.get("created_at", "")),
        updated_at=str(item.get("updated_at", "")),
    )
