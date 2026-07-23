from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class PluginCapability(BaseModel):
    name: str
    description: str = ""


class PluginMetadata(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    capabilities: list[PluginCapability] = Field(default_factory=list)


@runtime_checkable
class Plugin(Protocol):
    @property
    def metadata(self) -> PluginMetadata: ...

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def health_check(self) -> bool: ...

    async def execute(self, input_data: Any) -> Any: ...
