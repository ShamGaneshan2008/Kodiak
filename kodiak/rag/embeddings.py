"""
Provider-based embedding service for Kodiak V3 repository chunks.

This module consumes ``RepositoryChunk`` objects produced by
``kodiak.rag.chunking``. It does not parse source code, rescan repositories,
perform semantic search, retrieve documents, build prompts, or modify files.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import httpx
import structlog

from kodiak.rag.chunking import RepositoryChunk

logger = structlog.get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding provider identifiers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    VOYAGE = "voyage"
    GEMINI = "gemini"


DEFAULT_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.OLLAMA: "nomic-embed-text",
    EmbeddingProvider.VOYAGE: "voyage-code-3",
    EmbeddingProvider.GEMINI: "gemini-embedding-001",
}

DEFAULT_BASE_URLS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "https://api.openai.com/v1",
    EmbeddingProvider.OLLAMA: "http://localhost:11434",
    EmbeddingProvider.VOYAGE: "https://api.voyageai.com/v1",
    EmbeddingProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta",
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    batch_size: int = 64
    max_retries: int = 3
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 8.0
    timeout_seconds: float = 30.0
    max_concurrency: int = 4
    include_metadata_in_chunk_text: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        if isinstance(self.provider, str):
            object.__setattr__(self, "provider", EmbeddingProvider(self.provider))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.retry_min_seconds < 0:
            raise ValueError("retry_min_seconds must be zero or greater")
        if self.retry_max_seconds < self.retry_min_seconds:
            raise ValueError("retry_max_seconds must be greater than retry_min_seconds")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero when provided")

    @property
    def resolved_model(self) -> str:
        """Return the configured model or the provider default."""
        return self.model or DEFAULT_MODELS[self.provider]

    @property
    def resolved_base_url(self) -> str:
        """Return the configured base URL or the provider default."""
        return (self.base_url or DEFAULT_BASE_URLS[self.provider]).rstrip("/")


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding output for one input string."""

    text: str
    embedding: tuple[float, ...]
    provider: EmbeddingProvider
    model: str
    dimensions: int
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_hash(self) -> str:
        """Return a short stable hash for the embedded text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this result."""
        return {
            "text": self.text,
            "text_hash": self.text_hash,
            "embedding": list(self.embedding),
            "provider": self.provider.value,
            "model": self.model,
            "dimensions": self.dimensions,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ChunkEmbedding:
    """Embedding output paired with the repository chunk that produced it."""

    chunk: RepositoryChunk
    embedding: tuple[float, ...]
    provider: EmbeddingProvider
    model: str
    dimensions: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this chunk embedding."""
        return {
            "chunk": self.chunk.to_dict(),
            "embedding": list(self.embedding),
            "provider": self.provider.value,
            "model": self.model,
            "dimensions": self.dimensions,
            "metadata": self.metadata,
        }


class AsyncEmbeddingProvider(Protocol):
    """Minimal provider contract accepted by ``EmbeddingService``."""

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the provider identifier."""
        ...

    @property
    def model(self) -> str:
        """Return the model name used by the provider."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of input strings."""
        ...

    async def close(self) -> None:
        """Release provider-owned resources."""
        ...


class HTTPEmbeddingProvider(ABC):
    """Base class for HTTP embedding providers with retry handling."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Initialize an HTTP provider.

        Args:
            config: Provider configuration.
            client: Optional injected HTTP client for tests or shared transport.
        """
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._owns_client = client is None

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the configured provider identifier."""
        return self.config.provider

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self.config.resolved_model

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed ``texts`` with provider-specific HTTP calls and retries."""
        if not texts:
            return []

        response = await self._request_with_retries(texts)
        embeddings, token_count, metadata = self._parse_response(response)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"{self.provider.value} returned {len(embeddings)} embeddings for "
                f"{len(texts)} inputs"
            )

        return [
            EmbeddingResult(
                text=text,
                embedding=tuple(float(value) for value in embedding),
                provider=self.provider,
                model=self.model,
                dimensions=len(embedding),
                token_count=self._token_count_for_item(token_count, len(texts)),
                metadata=metadata,
            )
            for text, embedding in zip(texts, embeddings)
        ]

    async def close(self) -> None:
        """Close the provider HTTP client if this provider created it."""
        if self._owns_client:
            await self._client.aclose()

    async def _request_with_retries(self, texts: list[str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                start = time.perf_counter()
                response = await self._send_request(texts)
                response.raise_for_status()
                elapsed = time.perf_counter() - start
                logger.debug(
                    "embedding_provider_success",
                    provider=self.provider.value,
                    model=self.model,
                    batch_size=len(texts),
                    elapsed_ms=round(elapsed * 1000),
                )
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "embedding_provider_retry",
                    provider=self.provider.value,
                    model=self.model,
                    attempt=attempt,
                    max_retries=self.config.max_retries,
                    error=str(exc),
                )
                if attempt >= self.config.max_retries:
                    break
                await asyncio.sleep(self._retry_delay(attempt))

        raise RuntimeError(
            f"{self.provider.value} embedding request failed after "
            f"{self.config.max_retries} attempts"
        ) from last_error

    @abstractmethod
    async def _send_request(self, texts: list[str]) -> httpx.Response:
        """Send one provider-specific embedding request."""

    @abstractmethod
    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[list[float]], int | None, dict[str, Any]]:
        """Extract embeddings, optional token count, and metadata."""

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _retry_delay(self, attempt: int) -> float:
        delay = self.config.retry_min_seconds * (2 ** (attempt - 1))
        return min(delay, self.config.retry_max_seconds)

    @staticmethod
    def _token_count_for_item(total_tokens: int | None, item_count: int) -> int | None:
        if total_tokens is None or item_count <= 0:
            return None
        return total_tokens // item_count


class OpenAIEmbeddingProvider(HTTPEmbeddingProvider):
    """Embedding provider for OpenAI-compatible embeddings endpoints."""

    async def _send_request(self, texts: list[str]) -> httpx.Response:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.config.dimensions is not None:
            payload["dimensions"] = self.config.dimensions
        return await self._client.post(
            f"{self.config.resolved_base_url}/embeddings",
            headers=self._headers(),
            json=payload,
        )

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[list[float]], int | None, dict[str, Any]]:
        data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item["embedding"] for item in data]
        usage = response.get("usage") or {}
        metadata = {"usage": usage}
        return embeddings, usage.get("total_tokens"), metadata


class OllamaEmbeddingProvider(HTTPEmbeddingProvider):
    """Embedding provider for Ollama's local ``/api/embed`` endpoint."""

    async def _send_request(self, texts: list[str]) -> httpx.Response:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.config.dimensions is not None:
            payload["dimensions"] = self.config.dimensions
        return await self._client.post(
            f"{self.config.resolved_base_url}/api/embed",
            headers=self._headers(),
            json=payload,
        )

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[list[float]], int | None, dict[str, Any]]:
        embeddings = response.get("embeddings")
        if embeddings is None and "embedding" in response:
            embeddings = [response["embedding"]]
        metadata = {
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
        }
        return embeddings or [], response.get("prompt_eval_count"), metadata


class VoyageEmbeddingProvider(HTTPEmbeddingProvider):
    """Embedding provider for Voyage AI embeddings."""

    async def _send_request(self, texts: list[str]) -> httpx.Response:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.config.dimensions is not None:
            payload["output_dimension"] = self.config.dimensions
        return await self._client.post(
            f"{self.config.resolved_base_url}/embeddings",
            headers=self._headers(),
            json=payload,
        )

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[list[float]], int | None, dict[str, Any]]:
        data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item["embedding"] for item in data]
        usage = response.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        return embeddings, total_tokens, {"usage": usage}


class GeminiEmbeddingProvider(HTTPEmbeddingProvider):
    """Embedding provider for Gemini batch embedding requests."""

    async def _send_request(self, texts: list[str]) -> httpx.Response:
        model_path = f"models/{self.model}"
        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    **self._dimension_payload(),
                }
                for text in texts
            ]
        }
        params = {"key": self.config.api_key} if self.config.api_key else None
        return await self._client.post(
            f"{self.config.resolved_base_url}/{model_path}:batchEmbedContents",
            headers=self._headers(),
            params=params,
            json=payload,
        )

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[list[float]], int | None, dict[str, Any]]:
        embeddings = [
            item.get("values", [])
            for embedding in response.get("embeddings", [])
            for item in [embedding]
        ]
        return embeddings, None, {}

    def _dimension_payload(self) -> dict[str, int]:
        if self.config.dimensions is None:
            return {}
        return {"outputDimensionality": self.config.dimensions}


class EmbeddingService:
    """
    High-level embedding API for repository chunks and query text.

    A provider can be supplied directly for tests or custom integrations. When
    omitted, the service builds one from ``EmbeddingConfig.provider``.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        provider: AsyncEmbeddingProvider | None = None,
    ) -> None:
        """
        Initialize the embedding service.

        Args:
            config: Embedding behavior and provider configuration.
            provider: Optional injected provider implementation.
        """
        self.config = config or EmbeddingConfig()
        self.provider = provider or self._build_provider(self.config)
        self._semaphore = asyncio.Semaphore(max(self.config.max_concurrency, 1))

    async def embed_chunk(self, chunk: RepositoryChunk) -> ChunkEmbedding:
        """Embed one repository chunk."""
        results = await self.embed_chunks([chunk])
        return results[0]

    async def embed_chunks(
        self,
        chunks: list[RepositoryChunk] | tuple[RepositoryChunk, ...],
    ) -> list[ChunkEmbedding]:
        """Embed repository chunks while preserving input order."""
        if not chunks:
            return []

        texts = [self._chunk_text(chunk) for chunk in chunks]
        results = await self.batch_embed(texts)
        return [
            ChunkEmbedding(
                chunk=chunk,
                embedding=result.embedding,
                provider=result.provider,
                model=result.model,
                dimensions=result.dimensions,
                metadata={
                    "chunk_id": chunk.id,
                    "text_hash": result.text_hash,
                    "token_count": result.token_count,
                    **result.metadata,
                },
            )
            for chunk, result in zip(chunks, results)
        ]

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query string without building prompts or retrieving documents."""
        results = await self.batch_embed([query])
        return results[0]

    async def batch_embed(self, texts: list[str] | tuple[str, ...]) -> list[EmbeddingResult]:
        """Embed arbitrary texts with batching and bounded async concurrency."""
        if not texts:
            return []

        batches = [
            list(texts[index : index + self.config.batch_size])
            for index in range(0, len(texts), self.config.batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        return [result for batch_result in results for result in batch_result]

    async def close(self) -> None:
        """Release provider resources."""
        await self.provider.close()

    async def __aenter__(self) -> EmbeddingService:
        """Return this service for async context-manager usage."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close provider resources when leaving an async context."""
        await self.close()

    async def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        async with self._semaphore:
            return await self.provider.embed_texts(texts)

    def _chunk_text(self, chunk: RepositoryChunk) -> str:
        if not self.config.include_metadata_in_chunk_text:
            return chunk.source_code

        header = [
            f"module: {chunk.module_path}",
            f"symbol: {chunk.symbol_name}",
            f"type: {chunk.symbol_type.value}",
            f"lines: {chunk.start_line}-{chunk.end_line}",
        ]
        if chunk.parent_class:
            header.append(f"parent_class: {chunk.parent_class}")
        if chunk.docstring:
            header.append(f"docstring: {chunk.docstring}")
        if chunk.imports:
            header.append("imports:")
            header.extend(chunk.imports)

        return "\n".join([*header, "", chunk.source_code])

    @staticmethod
    def _build_provider(config: EmbeddingConfig) -> AsyncEmbeddingProvider:
        if config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingProvider(config)
        if config.provider == EmbeddingProvider.OLLAMA:
            return OllamaEmbeddingProvider(config)
        if config.provider == EmbeddingProvider.VOYAGE:
            return VoyageEmbeddingProvider(config)
        if config.provider == EmbeddingProvider.GEMINI:
            return GeminiEmbeddingProvider(config)
        raise ValueError(f"Unsupported embedding provider: {config.provider}")
