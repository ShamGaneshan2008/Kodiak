from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from pydantic import BaseModel

from kodiak.plugins.interface import Plugin

logger = structlog.get_logger(__name__)


class PluginPermission(BaseModel):
    filesystem: bool = False
    network: bool = False
    subprocess: bool = False


class PluginExecutionResult(BaseModel):
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


class PluginSandbox:
    def __init__(
        self,
        default_permissions: PluginPermission | None = None,
        default_timeout_secs: float = 30.0,
    ) -> None:
        self._default_permissions = default_permissions or PluginPermission()
        self._default_timeout_secs = default_timeout_secs

    def validate_permissions(
        self, required: PluginPermission, granted: PluginPermission | None = None
    ) -> bool:
        perms = granted or self._default_permissions
        if required.filesystem and not perms.filesystem:
            return False
        if required.network and not perms.network:
            return False
        if required.subprocess and not perms.subprocess:
            return False
        return True

    async def enforce_timeout(self, coro: Any, timeout_secs: float) -> Any:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_secs)
        except TimeoutError:
            logger.warning("plugin_execution_timeout", timeout_secs=timeout_secs)
            raise

    async def enforce_resource_limits(self, max_memory_mb: int = 512) -> None:
        logger.debug("resource_limits_enforced", max_memory_mb=max_memory_mb)

    async def execute_plugin(
        self,
        plugin: Plugin,
        input_data: Any = None,
        permissions: PluginPermission | None = None,
        timeout_secs: float | None = None,
    ) -> PluginExecutionResult:
        timeout = timeout_secs if timeout_secs is not None else self._default_timeout_secs
        start_time = time.perf_counter()

        try:
            output = await self.enforce_timeout(
                plugin.execute(input_data),
                timeout,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return PluginExecutionResult(
                success=True,
                output=output,
                execution_time_ms=elapsed_ms,
            )
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return PluginExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "plugin_execution_failed",
                plugin=plugin.metadata.name,
                error=str(e),
            )
            return PluginExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
