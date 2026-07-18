"""Parser module for the Repository Intelligence (RAG) subsystem.

This module is responsible for walking a repository, ignoring unwanted
directories, parsing supported file types, and producing ``ParsedDocument``
instances that can be fed into the rest of the RAG pipeline.

The implementation follows SOLID principles, uses pathlib for path handling,
structlog for logging, and provides extension points for custom language
parsers without modifying the core logic.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

# --------------------------------------------------------------------------- #
# Ignored directories (relative to repository root)
# --------------------------------------------------------------------------- #
_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
})

# --------------------------------------------------------------------------- #
# Supported file extensions and their associated languages
# --------------------------------------------------------------------------- #
_SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


# --------------------------------------------------------------------------- #
# Dataclass representing a parsed document
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Container for information about a single parsed file.

    Attributes
    ----------
    absolute_path: Path
        The full filesystem path to the file.
    relative_path: Path
        The path relative to the repository root.
    file_type: str
        The file type (e.g., "python", "markdown").
    language: str
        Human-readable language identifier.
    file_size: int
        Size of the file in bytes.
    last_modified: datetime.datetime
        Timestamp of the last modification (UTC).
    content: str
        The complete textual content of the file.
    metadata: dict[str, Any]
        Additional metadata (e.g., file extension, MIME type).
    """

    absolute_path: Path
    relative_path: Path
    file_type: str
    language: str
    file_size: int
    last_modified: datetime.datetime
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core parser class
# --------------------------------------------------------------------------- #
class RepositoryParser:
    """Parses all supported files within a repository.

    The class is designed to be easily extensible: new language parsers can be
    registered via :meth:`register_parser` without altering the core logic.

    Parameters
    ----------
    root_path: Path
        The root directory of the repository to be parsed.
    """

    def __init__(self, root_path: Path) -> None:
        if not root_path.is_dir():
            raise ValueError(f"Provided path '{root_path}' is not a directory.")
        self.root_path: Path = root_path.resolve()
        self._parser_registry: Dict[str, Callable[[Path, Path], ParsedDocument]] = {
            ext: self._default_text_parser for ext in _SUPPORTED_EXTENSIONS
        }
        self._logger = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    def parse_repository(self) -> List[ParsedDocument]:
        """Walk the repository and parse all supported files.

        Returns
        -------
        List[ParsedDocument]
            A list containing a ``ParsedDocument`` for each successfully parsed
            file. Files that cannot be parsed are silently skipped.
        """
        parsed_documents: List[ParsedDocument] = []
        for file_path in self._iter_repository_files():
            doc = self.parse_file(file_path)
            if doc is not None:
                parsed_documents.append(doc)
        self._logger.info("repository_parsed", total=len(parsed_documents), root=str(self.root_path))
        return parsed_documents

    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a single file and return its ``ParsedDocument``.

        If the file cannot be read or parsed, ``None`` is returned.

        Parameters
        ----------
        file_path: Path
            The absolute path to the file to parse.

        Returns
        -------
        Optional[ParsedDocument]
            The parsed document, or ``None`` if parsing failed.
        """
        try:
            # Skip if file is inside an ignored directory
            if any(ignored in file_path.parts for ignored in _IGNORED_DIRS):
                self._logger.debug("file_skipped_ignored_dir", file=str(file_path))
                return None

            # Skip binary files
            if self._is_binary(file_path):
                self._logger.debug("file_skipped_binary", file=str(file_path))
                return None

            # Ensure the file has a supported extension
            suffix = file_path.suffix.lower()
            if suffix not in _SUPPORTED_EXTENSIONS:
                self._logger.debug("file_type_not_supported", file=str(file_path), suffix=suffix)
                return None

            # Get the appropriate parser
            parser = self._parser_registry.get(suffix)
            if parser is None:
                self._logger.debug("no_parser_registered", file=str(file_path), suffix=suffix)
                return None

            # Parse the file
            return parser(file_path, self.root_path)

        except Exception as exc:
            self._logger.exception("file_parse_failed", file=str(file_path), error=str(exc))
            return None

    # ------------------------------------------------------------------- #
    # Extension points
    # ------------------------------------------------------------------- #
    def register_parser(
        self,
        extension: str,
        parser_func: Callable[[Path, Path], ParsedDocument],
    ) -> None:
        """Register a custom parser for a specific file extension.

        This method allows plugging in specialized parsing logic (e.g., for
        proprietary formats) without modifying the core ``RepositoryParser``.

        Parameters
        ----------
        extension: str
            The file extension (including the leading dot, e.g., ".toml").
        parser_func: Callable[[Path, Path], ParsedDocument]
            A function that accepts a ``Path`` (file path) and ``Path`` (root path)
            and returns a ``ParsedDocument``.
        """
        if not extension.startswith("."):
            raise ValueError("Extension must start with a dot, e.g., '.toml'")
        if extension not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Extension '{extension}' is not supported by default parsers.")
        self._parser_registry[extension] = parser_func
        self._logger.debug("parser_registered", extension=extension)

    # ------------------------------------------------------------------- #
    # Private helpers
    # ------------------------------------------------------------------- #
    def _iter_repository_files(self) -> List[Path]:
        """Yield all files under ``self.root_path`` while ignoring unwanted dirs.

        Directories matching any entry in ``_IGNORED_DIRS`` are skipped entirely.
        """
        all_files = list(self.root_path.rglob("*"))
        valid_files = [
            p
            for p in all_files
            if p.is_file()
            and not any(ignored in p.parts for ignored in _IGNORED_DIRS)
        ]
        self._logger.debug("repository_files_iterated", total=len(valid_files), root=str(self.root_path))
        return valid_files

    @staticmethod
    def _is_binary(file_path: Path) -> bool:
        """Very simple binary detection: presence of null bytes in the first 1024 bytes."""
        try:
            with file_path.open("rb") as f:
                chunk = f.read(1024)
                return b"\x00" in chunk
        except Exception:
            return False

    @staticmethod
    def _default_text_parser(file_path: Path, root_path: Path) -> ParsedDocument:
        """Default parser for plain text files.

        This parser simply reads the file as UTF-8 (or latin-1 fallback) and
        treats the whole content as the document's ``content`` field.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        stat = file_path.stat()
        suffix = file_path.suffix.lower()

        return ParsedDocument(
            absolute_path=file_path.resolve(),
            relative_path=file_path.relative_to(root_path),
            file_type=suffix[1:],  # strip leading dot
            language=_SUPPORTED_EXTENSIONS[suffix],
            file_size=stat.st_size,
            last_modified=datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
            content=content,
            metadata={
                "extension": suffix,
                "size_bytes": stat.st_size,
            },
        )
