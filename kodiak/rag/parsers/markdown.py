import re
from pathlib import Path

from kodiak.rag.parsers.base import CodeSymbol, Parser


class MarkdownParser(Parser):
    language = "markdown"
    pattern = re.compile(r"\s*#+\s+(.+)")

    def parse(self, path: Path, source: str) -> list[CodeSymbol]:
        symbols: list[CodeSymbol] = []
        for index, line in enumerate(source.splitlines(), start=1):
            match = self.pattern.match(line)
            if match:
                symbols.append(
                    CodeSymbol(name=match.group(1), kind="symbol", path=str(path), line=index)
                )
        return symbols
