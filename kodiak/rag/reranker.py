from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from kodiak.rag.vector_store import SearchResult as RetrievalResult

logger = structlog.get_logger(__name__)


class RankedResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class RankerProvider(Protocol):
    async def score(self, query: str, document: str) -> float: ...
    async def score_batch(self, query: str, documents: list[str]) -> list[float]: ...


class Reranker:
    def __init__(self, provider: RankerProvider, score_threshold: float = 0.0) -> None:
        self._provider = provider
        self._score_threshold = score_threshold

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RankedResult]:
        if not results:
            return []

        documents = [r.content for r in results]
        raw_scores = await self._provider.score_batch(query, documents)

        scored_results: list[RankedResult] = []
        for result, score in zip(results, raw_scores, strict=False):
            if score >= self._score_threshold:
                scored_results.append(
                    RankedResult(
                        id=str(result.chunk_id),
                        content=result.content,
                        score=score,
                        metadata=result.metadata,
                    )
                )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            "reranking_completed",
            input_count=len(results),
            output_count=len(scored_results),
            top_k=top_k,
        )
        return scored_results[:top_k]
