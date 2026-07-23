import asyncio
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    component_name: str
    status: HealthStatus
    response_time_ms: float
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class SystemHealthReport(BaseModel):
    overall_status: HealthStatus
    components: list[ComponentHealth]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class HealthCheck(Protocol):
    name: str

    async def check(self) -> ComponentHealth: ...


class HealthChecker:
    def __init__(self, default_timeout: float = 5.0) -> None:
        self._checks: dict[str, HealthCheck] = {}
        self._default_timeout = default_timeout

    def register_check(self, check: HealthCheck) -> None:
        self._checks[check.name] = check
        logger.debug("health_check_registered", name=check.name)

    async def run_check(
        self,
        name: str,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> ComponentHealth:
        check = self._checks.get(name)
        if check is None:
            return ComponentHealth(
                component_name=name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                details={"error": "Check not registered"},
            )

        timeout_secs = timeout if timeout is not None else self._default_timeout
        start_time = time.monotonic()

        try:
            return await asyncio.wait_for(check.check(), timeout=timeout_secs)
        except TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "health_check_timeout",
                name=name,
                timeout_secs=timeout_secs,
            )
            return ComponentHealth(
                component_name=name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=elapsed_ms,
                details={"error": "Timeout"},
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.exception("health_check_failed", name=name)
            return ComponentHealth(
                component_name=name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=elapsed_ms,
                details={"error": str(e)},
            )

    async def run_all_checks(self) -> list[ComponentHealth]:
        if not self._checks:
            return []
        tasks = [self.run_check(name) for name in self._checks]
        return list(await asyncio.gather(*tasks))

    async def get_system_health(self) -> SystemHealthReport:
        components = await self.run_all_checks()
        overall = self._compute_overall_status(components)
        return SystemHealthReport(overall_status=overall, components=components)

    async def get_unhealthy_components(self) -> list[ComponentHealth]:
        components = await self.run_all_checks()
        return [c for c in components if c.status != HealthStatus.HEALTHY]

    @staticmethod
    def _compute_overall_status(components: list[ComponentHealth]) -> HealthStatus:
        if not components:
            return HealthStatus.UNHEALTHY
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            return HealthStatus.UNHEALTHY
        if any(c.status == HealthStatus.DEGRADED for c in components):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
