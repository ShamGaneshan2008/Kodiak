import re
from pathlib import Path

import structlog

from kodiak.rag.parsers.base import BaseParser, ParsedFile, ParsedSymbol, SourceChunk

logger = structlog.get_logger(__name__)


class JavaScriptParser(BaseParser):
    def supports(self, path: Path) -> bool:
        return path.suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs"}

    def parse(self, path: Path, content: str) -> ParsedFile:
        lang = "javascript" if path.suffix in {".js", ".jsx", ".mjs"} else "typescript"
        return ParsedFile(
            path=str(path), language=lang,
            symbols=self.extract_symbols(content),
            imports=self.extract_imports(content),
            chunks=self.extract_chunks(content, path),
        )

    def extract_symbols(self, content: str) -> list[ParsedSymbol]:
        symbols: list[ParsedSymbol] = []
        patterns = [
            (r"(?:export\s+)?(?:default\s+)?(?:interface|enum)\s+(\w+)", lambda m: "interface" if "interface" in m.group(0) else "enum"),
            (r"(?:export\s+)?(?:default\s+)?class\s+(\w+)", lambda m: "class"),
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", lambda m: "function"),
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>", lambda m: "arrow_function"),
        ]
        for pattern, get_type in patterns:
            for match in re.finditer(pattern, content):
                start = content[:match.start()].count('\n') + 1
                symbols.append(ParsedSymbol(
                    name=match.group(1), symbol_type=get_type(match),
                    start_line=start, end_line=start,
                ))
        return symbols

    def extract_imports(self, content: str) -> list[str]:
        return re.findall(r"import\s+.*?from\s+['\"](.*?)['\"]", content)

    def extract_chunks(self, content: str, path: Path) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        lines = content.splitlines()
        pattern = re.compile(
            r"^(?:export\s+)?(?:default\s+)?(?:class|function|interface|enum|const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)\s+\w+",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(content))
        for i, match in enumerate(matches):
            start_line = content[:match.start()].count('\n') + 1
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            chunk_content = content[match.start():end_pos].strip()
            if chunk_content:
                end_line = content[:end_pos].count('\n')
                chunks.append(SourceChunk(
                    content=chunk_content, start_line=start_line,
                    end_line=end_line, chunk_type="block",
                ))
        if not chunks and content.strip():
            chunks.append(SourceChunk(content=content, start_line=1, end_line=len(lines)))
        return chunks