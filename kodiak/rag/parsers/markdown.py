import re
from pathlib import Path

import structlog

from kodiak.rag.parsers.base import BaseParser, ParsedFile, ParsedSymbol, SourceChunk

logger = structlog.get_logger(__name__)


class MarkdownParser(BaseParser):
    def supports(self, path: Path) -> bool:
        return path.suffix in {".md", ".mdx", ".markdown"}

    def parse(self, path: Path, content: str) -> ParsedFile:
        return ParsedFile(
            path=str(path), language="markdown",
            symbols=self.extract_symbols(content),
            imports=self.extract_imports(content),
            chunks=self.extract_chunks(content, path),
        )

    def extract_symbols(self, content: str) -> list[ParsedSymbol]:
        symbols = []
        lines = content.splitlines()
        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                symbols.append(ParsedSymbol(
                    name=match.group(2).strip(), symbol_type=f"h{level}",
                    start_line=i + 1, end_line=i + 1,
                ))
        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
            start = content[:match.start()].count('\n') + 1
            symbols.append(ParsedSymbol(
                name=match.group(1), symbol_type="link",
                start_line=start, end_line=start,
                metadata={"url": match.group(2)},
            ))
        return symbols

    def extract_imports(self, content: str) -> list[str]:
        urls = re.findall(r"!\[.*?\]\(([^)]+)\)", content)
        urls.extend(re.findall(r"(?<!\!)\[.*?\]\(([^)]+)\)", content))
        return urls

    def extract_chunks(self, content: str, path: Path) -> list[SourceChunk]:
        chunks = []
        lines = content.splitlines()
        heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        heading_matches = list(heading_pattern.finditer(content))

        for i, match in enumerate(heading_matches):
            start_pos = match.start()
            end_pos = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(content)
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n')
            chunk_content = content[start_pos:end_pos].strip()
            if chunk_content:
                chunks.append(SourceChunk(
                    content=chunk_content, start_line=start_line,
                    end_line=end_line, chunk_type="section",
                ))

        if not chunks and content.strip():
            chunks.append(SourceChunk(content=content, start_line=1, end_line=len(lines), chunk_type="document"))
        return chunks