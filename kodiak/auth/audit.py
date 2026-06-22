"""
kodiak/auth/audit.py

Append-only audit trail for every security-sensitive action in the auth
subsystem: logins, token issuance/revocation, API key lifecycle, scope/role
changes, and approval-gate decisions.

This is intentionally decoupled from application logging (config/logging.py).
Audit records are:
    - Written to Postgres (durable, queryable, joinable with org/user tables)
    - Mirrored to the event bus so monitoring/alerts.py can react in real time
      (e.g. alert on repeated failed logins or a sudden API-key creation spike)
    - Never deleted by normal application code -- only by an explicit,
      separately-permissioned retention job (see security/policy.py)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kodiak.db.models.audit_log import AuditLog  # ORM model: id, org_id, actor_id,
                                                  # actor_type, action, target_type,
                                                  # target_id, metadata, ip_address,
                                                  # user_agent, created_at
from kodiak.events.bus import publish
from kodiak.events.types import EventType


class ActorType(str, Enum):
    USER = "user"
    API_KEY = "api_key"
    SERVICE = "service"
    AGENT = "agent"  # an autonomous Kodiak agent acted on behalf of an org


class AuditAction(str, Enum):
    # Session lifecycle
    LOGIN_SUCCEEDED = "login.succeeded"
    LOGIN_FAILED = "login.failed"
    LOGOUT = "logout"
    TOKEN_REFRESHED = "token.refreshed"
    TOKEN_REVOKED = "token.revoked"

    # API keys
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_USED_INVALID = "api_key.used_invalid"

    # GitHub App / OAuth
    GITHUB_APP_INSTALLED = "github_app.installed"
    GITHUB_APP_UNINSTALLED = "github_app.uninstalled"
    OAUTH_LINKED = "oauth.linked"

    # RBAC
    ROLE_GRANTED = "role.granted"
    ROLE_REVOKED = "role.revoked"
    PERMISSION_DENIED = "permission.denied"

    # Approval workflow (human-in-the-loop gate before autonomous merges)
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_AUTO_ESCALATED = "approval.auto_escalated"


@dataclass
class AuditContext:
    """Request-scoped metadata attached to every audit entry for forensics."""

    org_id: str
    actor_id: str
    actor_type: ActorType
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


async def record(
    db: AsyncSession,
    *,
    action: AuditAction,
    context: AuditContext,
    target_type: str | None = None,
    target_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditLog:
    """
    Persist a single audit entry and emit a corresponding event.

    Callers should pass the same `db` session/transaction used for the
    underlying action (e.g. revoking an API key) so the audit row is
    committed atomically with the action it describes -- a revoked key
    with no audit trail is worse than no revocation feature at all.
    """
    entry = AuditLog(
        id=str(uuid.uuid4()),
        org_id=context.org_id,
        actor_id=context.actor_id,
        actor_type=context.actor_type.value,
        action=action.value,
        target_type=target_type,
        target_id=target_id,
        metadata={**context.metadata, **(metadata or {})},
        ip_address=context.ip_address,
        user_agent=context.user_agent,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    await db.flush()  # caller controls the final commit

    await publish(
        EventType.AUDIT_LOG_CREATED,
        payload={
            "audit_id": entry.id,
            "org_id": entry.org_id,
            "action": entry.action,
            "actor_id": entry.actor_id,
            "actor_type": entry.actor_type,
            "target_type": target_type,
            "target_id": target_id,
        },
    )

    return entry


async def query_audit_log(
    db: AsyncSession,
    *,
    org_id: str,
    actions: list[AuditAction] | None = None,
    actor_id: str | None = None,
    target_id: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AuditLog]:
    """Filterable read path for the dashboard's security/audit view."""
    stmt = select(AuditLog).where(AuditLog.org_id == org_id)

    if actions:
        stmt = stmt.where(AuditLog.action.in_([a.value for a in actions]))
    if actor_id:
        stmt = stmt.where(AuditLog.actor_id == actor_id)
    if target_id:
        stmt = stmt.where(AuditLog.target_id == target_id)
    if since:
        stmt = stmt.where(AuditLog.created_at >= since)

    stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def record_permission_denied(
    db: AsyncSession,
    *,
    context: AuditContext,
    required_scope: str,
    route: str,
) -> AuditLog:
    """Convenience wrapper used by permissions.py whenever `require_scope` rejects a request."""
    return await record(
        db,
        action=AuditAction.PERMISSION_DENIED,
        context=context,
        target_type="route",
        target_id=route,
        metadata={"required_scope": required_scope},
    )