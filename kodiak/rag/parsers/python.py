import ast
from pathlib import Path

import structlog

from kodiak.rag.parsers.base import BaseParser, ParsedFile, ParsedSymbol, SourceChunk

logger = structlog.get_logger(__name__)


class PythonParser(BaseParser):
    def supports(self, path: Path) -> bool:
        return path.suffix == ".py"

    def parse(self, path: Path, content: str) -> ParsedFile:
        return ParsedFile(
            path=str(path),
            language="python",
            symbols=self.extract_symbols(content),
            imports=self.extract_imports(content),
            chunks=self.extract_chunks(content, path),
        )

    def extract_symbols(self, content: str) -> list[ParsedSymbol]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        symbols: list[ParsedSymbol] = []
        self._visit(tree, None, symbols)
        return symbols

    def _visit(self, node: ast.AST, parent: ast.AST | None, symbols: list[ParsedSymbol]) -> None:
        if isinstance(node, ast.ClassDef):
            decorators = [self._get_decorator_name(d) for d in node.decorator_list]
            docstring = ast.get_docstring(node)
            symbols.append(
                ParsedSymbol(
                    name=node.name,
                    symbol_type="class",
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    metadata={"decorators": ", ".join(decorators), "docstring": docstring or ""},
                )
            )
            for child in ast.iter_child_nodes(node):
                self._visit(child, node, symbols)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_method = isinstance(parent, ast.ClassDef)
            sym_type = (
                "method"
                if is_method
                else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
            )
            decorators = [self._get_decorator_name(d) for d in node.decorator_list]
            docstring = ast.get_docstring(node)
            symbols.append(
                ParsedSymbol(
                    name=node.name,
                    symbol_type=sym_type,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    metadata={"decorators": ", ".join(decorators), "docstring": docstring or ""},
                )
            )
            for child in ast.iter_child_nodes(node):
                self._visit(child, node, symbols)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(
                        ParsedSymbol(
                            name=target.id,
                            symbol_type="constant",
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                        )
                    )
        else:
            for child in ast.iter_child_nodes(node):
                self._visit(child, node, symbols)

    def _get_decorator_name(self, dec: ast.expr) -> str:
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Attribute):
            return f"{self._get_decorator_name(dec.value)}.{dec.attr}"
        return ""

    def extract_imports(self, content: str) -> list[str]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def extract_chunks(self, content: str, path: Path) -> list[SourceChunk]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return [
                SourceChunk(
                    content=content, start_line=1, end_line=max(1, len(content.splitlines()))
                )
            ]
        lines = content.splitlines()
        chunks: list[SourceChunk] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno or start + 1
                chunk_content = "\n".join(lines[start:end])
                chunks.append(
                    SourceChunk(
                        content=chunk_content,
                        start_line=node.lineno,
                        end_line=end,
                        chunk_type="class" if isinstance(node, ast.ClassDef) else "function",
                    )
                )
        if not chunks:
            chunks.append(SourceChunk(content=content, start_line=1, end_line=len(lines)))
        return chunks
