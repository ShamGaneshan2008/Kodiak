"""
kodiak/rag/embedder.py

Embedding generation with batching, caching, and retry logic.
Supports OpenAI, Anthropic-compatible, and local embedding models.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import structlog
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536
DEFAULT_BATCH_SIZE = 128
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    text: str
    embedding: list[float]
    model: str
    token_count: int
    cached: bool = False

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


# ---------------------------------------------------------------------------
# In-process LRU cache (replaced by Redis in production)
# ---------------------------------------------------------------------------


class EmbeddingCache:
    def __init__(self, max_size: int = 10_000) -> None:
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, text: str, model: str) -> str:
        raw = f"{model}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        k = self._key(text, model)
        val = self._cache.get(k)
        if val is not None:
            self._hits += 1
        else:
            self._misses += 1
        return val

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        if len(self._cache) >= self._max_size:
            # Evict oldest (first inserted)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[self._key(text, model)] = embedding

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """
    Async embedder with batching, in-process cache, and retry.

    Usage::

        embedder = Embedder(api_key="sk-...", model="text-embedding-3-small")
        results = await embedder.embed_many(["hello world", "foo bar"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_size: int = 10_000,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._cache = EmbeddingCache(max_size=cache_size)
        self._client = AsyncOpenAI(
            api_key=api_key or "not-set",
            base_url=base_url,
        )
        self._total_tokens = 0
        self._total_requests = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        results = await self.embed_many([text])
        return results[0]

    async def embed_many(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts with batching and cache lookup."""
        if not texts:
            return []

        results: list[EmbeddingResult | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Cache lookup pass
        for i, text in enumerate(texts):
            cached = self._cache.get(text, self.model)
            if cached is not None:
                results[i] = EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=self.model,
                    token_count=0,
                    cached=True,
                )
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch API calls for uncached
        if uncached_texts:
            api_results = await self._embed_batched(uncached_texts)
            for idx, result in zip(uncached_indices, api_results, strict=False):
                self._cache.set(result.text, self.model, result.embedding)
                results[idx] = result

        # Satisfy type checker – all slots must be filled
        return [r for r in results if r is not None]

    async def embed_chunks(self, chunks: list[Any]) -> list[tuple[Any, list[float]]]:
        """
        Convenience: embed Chunk objects, return (chunk, embedding) pairs.
        """
        texts = [c.content for c in chunks]
        results = await self.embed_many(texts)
        return [(chunk, result.embedding) for chunk, result in zip(chunks, results, strict=False)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _embed_batched(self, texts: list[str]) -> list[EmbeddingResult]:
        """Split into batches and call API concurrently."""
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        tasks = [self._call_api(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        flat: list[EmbeddingResult] = []
        for results in batch_results:
            flat.extend(results)
        return flat

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _call_api(self, texts: list[str]) -> list[EmbeddingResult]:
        start = time.perf_counter()
        log = logger.bind(model=self.model, batch_size=len(texts))

        try:
            response = await self._client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions if "3-" in self.model else None,  # type: ignore
            )
        except Exception as exc:
            log.error("embedding_api_error", error=str(exc))
            raise

        elapsed = time.perf_counter() - start
        total_tokens = response.usage.total_tokens if response.usage else 0
        self._total_tokens += total_tokens
        self._total_requests += 1

        log.debug(
            "embedding_api_success",
            elapsed_ms=round(elapsed * 1000),
            total_tokens=total_tokens,
        )

<<<<<<< HEAD
        results = []
        for item, text in zip(response.data, texts, strict=False):
=======
        results: list[EmbeddingResult] = []
        for item, text in zip(response.data, texts):
>>>>>>> narasimha/refactor/3-rag-type-hints
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=item.embedding,
                    model=self.model,
                    token_count=total_tokens // len(texts),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "cache": self._cache.stats,
        }

    async def health_check(self) -> bool:
        try:
            result = await self.embed("health check ping")
            return len(result.embedding) > 0
        except Exception as exc:
            logger.error("embedder_health_check_failed", error=str(exc))
            return False
