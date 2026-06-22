from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CodeSymbol:
    name: str
    kind: str
    path: str
    line: int


class Parser:
    language = "text"

    def parse(self, path: Path, source: str) -> list[CodeSymbol]:
        return []
