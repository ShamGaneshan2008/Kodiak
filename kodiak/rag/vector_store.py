from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VectorStore:
    name: str = "vector_store"

    async def run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"component": self.name, "payload": payload or {}}
