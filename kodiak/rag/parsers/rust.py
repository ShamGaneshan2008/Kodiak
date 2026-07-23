import re
from pathlib import Path

import structlog

from kodiak.rag.parsers.base import BaseParser, ParsedFile, ParsedSymbol, SourceChunk

logger = structlog.get_logger(__name__)


class RustParser(BaseParser):
    def supports(self, path: Path) -> bool:
        return path.suffix == ".rs"

    def parse(self, path: Path, content: str) -> ParsedFile:
        return ParsedFile(
            path=str(path),
            language="rust",
            symbols=self.extract_symbols(content),
            imports=self.extract_imports(content),
            chunks=self.extract_chunks(content, path),
        )

    def extract_symbols(self, content: str) -> list[ParsedSymbol]:
        symbols = []
        for match in re.finditer(r"(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", content):
            self._add_symbol(symbols, content, match, match.group(0).split()[1])
        for match in re.finditer(r"impl\s+(?:<[^>]+>\s+)?(\w+)", content):
            self._add_symbol(symbols, content, match, "impl")
        for match in re.finditer(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", content):
            self._add_symbol(symbols, content, match, "function")
        return symbols

    def _add_symbol(
        self, symbols: list[ParsedSymbol], content: str, match: re.Match[str], sym_type: str
    ) -> None:
        start = content[: match.start()].count("\n") + 1
        symbols.append(
            ParsedSymbol(
                name=match.group(1),
                symbol_type=sym_type,
                start_line=start,
                end_line=start,
            )
        )

    def extract_imports(self, content: str) -> list[str]:
        return re.findall(r"use\s+([^;]+);", content)

    def extract_chunks(self, content: str, path: Path) -> list[SourceChunk]:
        chunks = []
        lines = content.splitlines()
        pattern = re.compile(
            r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+\w+", re.MULTILINE
        )
        matches = list(pattern.finditer(content))
        for i, match in enumerate(matches):
            start_line = content[: match.start()].count("\n") + 1
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            chunk_content = content[match.start() : end_pos].strip()
            if chunk_content:
                end_line = content[:end_pos].count("\n")
                chunks.append(
                    SourceChunk(
                        content=chunk_content,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="block",
                    )
                )
        if not chunks and content.strip():
            chunks.append(SourceChunk(content=content, start_line=1, end_line=len(lines)))
        return chunks
