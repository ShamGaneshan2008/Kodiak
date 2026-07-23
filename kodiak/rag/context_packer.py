import structlog
from pydantic import BaseModel

from kodiak.rag.vector_store import SearchResult as RetrievalResult

logger = structlog.get_logger(__name__)


class PackedContext(BaseModel):
    content: str
    token_count: int
    chunks_used: int
    discarded_chunks: int


class ContextPacker:
    def __init__(self, chars_per_token: float = 4.0) -> None:
        self._chars_per_token = chars_per_token

    def pack(
        self,
        chunks: list[RetrievalResult],
        max_tokens: int = 4000,
    ) -> PackedContext:
        max_chars = int(max_tokens * self._chars_per_token)
        unique_chunks = self._deduplicate(chunks)

        packed_parts: list[str] = []
        current_chars = 0
        discarded = 0

        for chunk in unique_chunks:
            file_path = chunk.metadata.get("file_path", "unknown")
            chunk_content = f"---\nFile: {file_path}\n```\n{chunk.content}\n```\n"
            chunk_len = len(chunk_content)

            if current_chars + chunk_len > max_chars:
                discarded += 1
                continue

            packed_parts.append(chunk_content)
            current_chars += chunk_len

        final_content = "\n".join(packed_parts)
        estimated_tokens = int(len(final_content) / self._chars_per_token)

        logger.info(
            "context_packed",
            chunks_requested=len(chunks),
            chunks_used=len(packed_parts),
            discarded=discarded,
            estimated_tokens=estimated_tokens,
        )

        return PackedContext(
            content=final_content,
            token_count=estimated_tokens,
            chunks_used=len(packed_parts),
            discarded_chunks=discarded,
        )

    def _deduplicate(self, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        seen_content: set[str] = set()
        unique: list[RetrievalResult] = []

        for chunk in chunks:
            if chunk.content not in seen_content:
                seen_content.add(chunk.content)
                unique.append(chunk)

        return unique
