import structlog
from pydantic import BaseModel, Field, model_validator

logger = structlog.get_logger(__name__)


class ResourceLimits(BaseModel):
    cpu_cores: float = Field(default=1.0, ge=0.1, le=64.0)
    memory_mb: int = Field(default=512, ge=64, le=32768)
    disk_mb: int = Field(default=1024, ge=100, le=102400)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)

    @model_validator(mode="after")
    def validate_cpu_memory_ratio(self) -> ResourceLimits:
        min_memory_per_core = 64
        required_memory = int(self.cpu_cores * min_memory_per_core)
        if self.memory_mb < required_memory:
            raise ValueError(
                f"Memory ({self.memory_mb}MB) too low for "
                f"{self.cpu_cores} CPU cores. Minimum: {required_memory}MB."
            )
        return self


class ResourceLimitManager:
    def __init__(self, global_limits: ResourceLimits | None = None) -> None:
        self._global_limits = global_limits or ResourceLimits()

    def validate_limits(self, requested: ResourceLimits) -> list[str]:
        violations: list[str] = []

        if requested.cpu_cores > self._global_limits.cpu_cores:
            violations.append(
                f"Requested CPU ({requested.cpu_cores}) exceeds limit "
                f"({self._global_limits.cpu_cores})"
            )
        if requested.memory_mb > self._global_limits.memory_mb:
            violations.append(
                f"Requested Memory ({requested.memory_mb}MB) exceeds limit "
                f"({self._global_limits.memory_mb}MB)"
            )
        if requested.disk_mb > self._global_limits.disk_mb:
            violations.append(
                f"Requested Disk ({requested.disk_mb}MB) exceeds limit "
                f"({self._global_limits.disk_mb}MB)"
            )
        if requested.timeout_seconds > self._global_limits.timeout_seconds:
            violations.append(
                f"Requested Timeout ({requested.timeout_seconds}s) exceeds limit "
                f"({self._global_limits.timeout_seconds}s)"
            )

        if violations:
            logger.warning("resource_limit_violations", violations=violations)

        return violations

    def apply_limits(self, limits: ResourceLimits) -> dict[str, str | int | float]:
        logger.info("applying_resource_limits", limits=limits.model_dump())
        return {
            "cpu_quota": int(limits.cpu_cores * 100000),
            "mem_limit": f"{limits.memory_mb}m",
            "disk_limit": f"{limits.disk_mb}m",
            "timeout": limits.timeout_seconds,
        }

    def check_usage(self, current_cpu: float, current_memory_mb: int) -> bool:
        within_cpu = current_cpu <= self._global_limits.cpu_cores
        within_mem = current_memory_mb <= self._global_limits.memory_mb

        if not within_cpu or not within_mem:
            logger.warning(
                "usage_exceeds_limits",
                current_cpu=current_cpu,
                current_memory_mb=current_memory_mb,
            )
            return False
        return True

    def exceeds_limits(self, requested: ResourceLimits) -> bool:
        return len(self.validate_limits(requested)) > 0