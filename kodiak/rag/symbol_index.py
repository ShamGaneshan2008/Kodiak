import ast
from pathlib import Path

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class Symbol(BaseModel):
    name: str
    symbol_type: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""


class SymbolIndex:
    def __init__(self) -> None:
        self._symbols: dict[str, list[Symbol]] = {}

    def add_symbol(self, symbol: Symbol) -> None:
        if symbol.name not in self._symbols:
            self._symbols[symbol.name] = []
        self._symbols[symbol.name].append(symbol)

    def get_symbol(self, name: str) -> list[Symbol]:
        return self._symbols.get(name, [])

    def search_symbols(self, prefix: str) -> list[Symbol]:
        results: list[Symbol] = []
        for name, symbols in self._symbols.items():
            if name.startswith(prefix):
                results.extend(symbols)
        return results

    def parse_file(self, file_path: Path) -> int:
        if not file_path.is_file() or file_path.suffix != ".py":
            return 0

        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        count = 0
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                sig = f"class {node.name}"
                bases = ", ".join(self._get_name(b) for b in node.bases)
                if bases:
                    sig += f"({bases})"
                self.add_symbol(
                    Symbol(
                        name=node.name,
                        symbol_type="class",
                        file_path=str(file_path),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=sig,
                    )
                )
                count += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [self._get_name(a) for a in node.args.args]
                sig = f"def {node.name}({', '.join(args)})"
                self.add_symbol(
                    Symbol(
                        name=node.name,
                        symbol_type="function",
                        file_path=str(file_path),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=sig,
                    )
                )
                count += 1

        logger.debug("symbols_parsed", path=str(file_path), count=count)
        return count

    def _get_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""
