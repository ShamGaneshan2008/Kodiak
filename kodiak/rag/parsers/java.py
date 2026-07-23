import re
from pathlib import Path

import structlog

from kodiak.rag.parsers.base import BaseParser, ParsedFile, ParsedSymbol, SourceChunk

logger = structlog.get_logger(__name__)


class JavaParser(BaseParser):
    def supports(self, path: Path) -> bool:
        return path.suffix == ".java"

    def parse(self, path: Path, content: str) -> ParsedFile:
        return ParsedFile(
            path=str(path),
            language="java",
            symbols=self.extract_symbols(content),
            imports=self.extract_imports(content),
            chunks=self.extract_chunks(content, path),
        )

    def extract_symbols(self, content: str) -> list[ParsedSymbol]:
        symbols = []
        for match in re.finditer(
            r"(?:public|private|protected)?\s*(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)",
            content,
        ):
            self._add_symbol(symbols, content, match, "class")
        for match in re.finditer(r"(@\w+)", content):
            self._add_symbol(symbols, content, match, "annotation")
        for match in re.finditer(
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{",
            content,
        ):
            self._add_symbol(symbols, content, match, "method")
        for match in re.finditer(
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?([\w<>\[\]]+)\s+(\w+)\s*[;=]",
            content,
        ):
            start = content[: match.start()].count("\n") + 1
            symbols.append(
                ParsedSymbol(
                    name=match.group(2),
                    symbol_type="field",
                    start_line=start,
                    end_line=start,
                    metadata={"field_type": match.group(1)},
                )
            )
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
        return re.findall(r"import\s+(?:static\s+)?([\w.]+)", content)

    def extract_chunks(self, content: str, path: Path) -> list[SourceChunk]:
        chunks = []
        lines = content.splitlines()
        pattern = re.compile(
            r"^(?:public|private|protected)?\s*(?:abstract\s+)?(?:class|interface|enum)\s+\w+",
            re.MULTILINE,
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
                        chunk_type="class",
                    )
                )
        if not chunks and content.strip():
            chunks.append(SourceChunk(content=content, start_line=1, end_line=len(lines)))
        return chunks
