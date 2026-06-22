from pathlib import Path
import re

from kodiak.rag.parsers.base import CodeSymbol, Parser


class RustParser(Parser):
    language = "rust"
    pattern = re.compile(r"\s*(?:pub\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)")

    def parse(self, path: Path, source: str) -> list[CodeSymbol]:
        symbols: list[CodeSymbol] = []
        for index, line in enumerate(source.splitlines(), start=1):
            match = self.pattern.match(line)
            if match:
                symbols.append(CodeSymbol(name=match.group(1), kind="symbol", path=str(path), line=index))
        return symbols
