from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OutputParser:
    name: str = "output_parser"

    async def run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"component": self.name, "payload": payload or {}}
