from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Pagination:
    name: str = "pagination"

    async def run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"component": self.name, "payload": payload or {}}
