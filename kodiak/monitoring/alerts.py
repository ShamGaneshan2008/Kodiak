from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(StrEnum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class Alert(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    message: str
    severity: AlertSeverity
    state: AlertState = AlertState.ACTIVE
    source: str
    fingerprint: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: datetime | None = None
    escalated_at: datetime | None = None


@runtime_checkable
class AlertRepository(Protocol):
    async def get_by_fingerprint(self, fingerprint: str) -> Alert | None: ...

    async def store(self, alert: Alert) -> Alert: ...

    async def get_by_id(self, alert_id: uuid.UUID) -> Alert | None: ...

    async def update(self, alert: Alert) -> Alert: ...

    async def list_active(self, source: str | None = None) -> list[Alert]: ...


class AlertManager:
    def __init__(
        self,
        repository: AlertRepository,
        dedup_window_secs: int = 300,
    ) -> None:
        self._repo = repository
        self._dedup_window_secs = dedup_window_secs

    @staticmethod
    def generate_fingerprint(
        name: str,
        source: str,
        context: dict[str, Any],
    ) -> str:
        normalized = json.dumps(
            {"name": name, "source": source, "ctx": context},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def create_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> Alert:
        ctx = context or {}
        fingerprint = self.generate_fingerprint(name, source, ctx)

        existing = await self._repo.get_by_fingerprint(fingerprint)
        now = datetime.now(timezone.utc)

        if (
            existing
            and existing.state == AlertState.ACTIVE
            and (now - existing.updated_at).total_seconds() < self._dedup_window_secs
        ):
            logger.debug(
                "alert_deduplicated",
                name=name,
                fingerprint=fingerprint,
            )
            existing.updated_at = now
            return await self._repo.update(existing)

        alert = Alert(
            name=name,
            message=message,
            severity=severity,
            source=source,
            fingerprint=fingerprint,
            context=ctx,
            created_at=now,
            updated_at=now,
        )

        logger.info(
            "alert_created",
            alert_id=str(alert.id),
            name=name,
            severity=severity.value,
        )

        return await self._repo.store(alert)

    async def acknowledge(self, alert_id: uuid.UUID) -> Alert | None:
        alert = await self._repo.get_by_id(alert_id)
        if not alert or alert.state != AlertState.ACTIVE:
            return None

        now = datetime.now(timezone.utc)
        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = now
        alert.updated_at = now

        logger.info("alert_acknowledged", alert_id=str(alert_id))
        return await self._repo.update(alert)

    async def escalate(self, alert_id: uuid.UUID) -> Alert | None:
        alert = await self._repo.get_by_id(alert_id)
        if not alert or alert.state not in (
            AlertState.ACTIVE,
            AlertState.ACKNOWLEDGED,
        ):
            return None

        now = datetime.now(timezone.utc)
        alert.state = AlertState.ESCALATED
        alert.escalated_at = now
        alert.updated_at = now

        logger.warning(
            "alert_escalated",
            alert_id=str(alert_id),
            name=alert.name,
        )
        return await self._repo.update(alert)

    async def resolve(self, alert_id: uuid.UUID) -> Alert | None:
        alert = await self._repo.get_by_id(alert_id)
        if not alert or alert.state == AlertState.RESOLVED:
            return None

        alert.state = AlertState.RESOLVED
        alert.updated_at = datetime.now(timezone.utc)

        logger.info("alert_resolved", alert_id=str(alert_id))
        return await self._repo.update(alert)

    async def get_active_alerts(self, source: str | None = None) -> list[Alert]:
        return await self._repo.list_active(source)