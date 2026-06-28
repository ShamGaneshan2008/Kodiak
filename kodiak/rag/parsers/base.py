import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SourceChunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str
    start_line: int
    end_line: int
    chunk_type: str = "code"


class ParsedSymbol(BaseModel):
    name: str
    symbol_type: str
    signature: str = ""
    start_line: int = 0
    end_line: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedFile(BaseModel):
    path: str
    language: str
    symbols: list[ParsedSymbol] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    chunks: list[SourceChunk] = Field(default_factory=list)


class BaseParser(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    @abstractmethod
    def parse(self, path: Path, content: str) -> ParsedFile: ...

    @abstractmethod
    def extract_symbols(self, content: str) -> list[ParsedSymbol]: ...

    @abstractmethod
    def extract_imports(self, content: str) -> list[str]: ...

    @abstractmethod
    def extract_chunks(self, content: str, path: Path) -> list[SourceChunk]: ...