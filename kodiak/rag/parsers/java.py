import re
from pathlib import Path

from kodiak.rag.parsers.base import CodeSymbol, Parser


class JavaParser(Parser):
    language = "java"
    pattern = re.compile(
        r"\s*(?:public|private|protected)?\s*(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
    )

    def parse(self, path: Path, source: str) -> list[CodeSymbol]:
        symbols: list[CodeSymbol] = []
        for index, line in enumerate(source.splitlines(), start=1):
            match = self.pattern.match(line)
            if match:
                symbols.append(
                    CodeSymbol(name=match.group(1), kind="symbol", path=str(path), line=index)
                )
        return symbols
