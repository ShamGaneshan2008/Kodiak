import uuid
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ApprovalStatus(StrEnum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"


class ApprovalRequest(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    operation: str
    risk_level: str = "low"
    details: dict[str, str] = Field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING


class ApprovalGate:
    def __init__(self, auto_approve_low_risk: bool = True) -> None:
        self._auto_approve_low = auto_approve_low_risk
        self._protected_operations = {
            "delete_file",
            "overwrite_file",
            "git_push",
            "create_pr",
            "execute_shell",
            "database_modify",
        }
        self._high_risk_operations = {
            "delete_file",
            "git_push",
            "execute_shell",
            "database_modify",
        }

    async def request_approval(
        self, operation: str, details: dict[str, str] | None = None
    ) -> ApprovalRequest:
        request = ApprovalRequest(
            operation=operation,
            risk_level="high" if operation in self._high_risk_operations else "low",
            details=details or {},
        )

        if operation not in self._protected_operations:
            request.status = ApprovalStatus.APPROVED
            logger.info("operation_auto_approved_unprotected", operation=operation)
            return request

        if self._auto_approve_low and request.risk_level == "low":
            request.status = ApprovalStatus.APPROVED
            logger.info("operation_auto_approved_low_risk", operation=operation)
        else:
            logger.warning(
                "approval_required",
                operation=operation,
                risk=request.risk_level,
                request_id=str(request.id),
            )

        return request

    async def evaluate_operation(self, operation: str) -> bool:
        request = await self.request_approval(operation)
        return request.status == ApprovalStatus.APPROVED