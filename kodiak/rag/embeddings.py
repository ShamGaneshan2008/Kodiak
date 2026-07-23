"""Embedding generation for the Kodiak Repository Intelligence subsystem.

This module defines the ``EmbeddingProvider`` abstraction, concrete provider
implementations (OpenAI, Ollama, Gemini), and the ``EmbeddingManager`` that
orchestrates provider selection, retries, fallback, dimension validation,
and caching.

This module has no dependency on ChromaDB or any other vector store. Its
sole responsibility is turning text into vectors; persistence and
similarity search belong to other `kodiak/rag/` modules (e.g. a future
``vector_store.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingManager",
    "EmbeddingResult",
    "RetryConfig",
    "EmbeddingCache",
    "NullEmbeddingCache",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "EmbeddingError",
    "ProviderConfigurationError",
    "ProviderRequestError",
    "EmbeddingDimensionMismatchError",
    "AllProvidersExhaustedError",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Base exception for all embedding-related failures."""


class ProviderConfigurationError(EmbeddingError):
    """Raised when a provider or the manager is misconfigured."""


class ProviderRequestError(EmbeddingError):
    """Raised when a provider's underlying request fails.

    Attributes:
        retryable: Whether the manager's retry logic should retry this
            failure. Transient issues (timeouts, 5xx, rate limits) should
            set this to ``True``; malformed responses or auth failures
            should set it to ``False``.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when a provider returns a vector of an unexpected dimension."""

    def __init__(self, *, expected: int, actual: int, provider: str) -> None:
        super().__init__(f"{provider} returned a {actual}-dimensional vector, expected {expected}")
        self.expected = expected
        self.actual = actual
        self.provider = provider


class AllProvidersExhaustedError(EmbeddingError):
    """Raised when every eligible provider failed for a given request."""


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """A single embedding result.

    Attributes:
        vector: The embedding vector.
        model: The model identifier that produced the vector.
        provider: The provider name that produced the vector.
    """

    vector: list[float]
    model: str
    provider: str


@dataclass(slots=True)
class RetryConfig:
    """Retry policy for transient provider failures.

    Attributes:
        max_attempts: Total attempts per provider, including the first.
        base_delay_seconds: Initial backoff delay.
        max_delay_seconds: Ceiling applied to exponential backoff.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0


@runtime_checkable
class EmbeddingCache(Protocol):
    """Structural type for a caching backend used by ``EmbeddingManager``.

    Implementations might be backed by Redis, an in-memory dict, or a
    database table. ``EmbeddingManager`` depends only on this narrow
    protocol, never on a concrete caching technology.
    """

    async def get(self, key: str) -> list[float] | None:
        """Return a cached vector for ``key``, or ``None`` on a miss."""
        ...

    async def set(self, key: str, vector: list[float]) -> None:
        """Store ``vector`` under ``key``."""
        ...


class NullEmbeddingCache:
    """No-op cache used when no caching backend is configured."""

    async def get(self, key: str) -> list[float] | None:
        """Always return ``None``, indicating a cache miss."""
        return None

    async def set(self, key: str, vector: list[float]) -> None:
        """Discard the value; this cache stores nothing."""
        return None


@runtime_checkable
class AsyncHTTPResponse(Protocol):
    """Structural type for the HTTP response object providers receive."""

    status_code: int

    def json(self) -> dict[str, Any]:
        """Parse and return the response body as JSON."""
        ...

    def raise_for_status(self) -> None:
        """Raise an exception if the response indicates an HTTP error."""
        ...


@runtime_checkable
class AsyncHTTPClient(Protocol):
    """Structural type for the async HTTP client injected into providers.

    Matches the subset of ``httpx.AsyncClient`` that providers need, so
    providers can be exercised in tests with a fake client instead of a
    hard dependency on any specific HTTP library.
    """

    async def post(self, url: str, **kwargs: object) -> AsyncHTTPResponse:
        """Issue an async POST request."""
        ...


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base class for all embedding providers.

    Concrete providers wrap a specific embedding backend behind a uniform
    async interface so the rest of Kodiak never depends on a particular
    vendor's SDK or wire format.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single piece of text.

        Args:
            text: The text to embed.

        Returns:
            The resulting embedding.

        Raises:
            ProviderRequestError: If the underlying request fails.
        """

    @abstractmethod
    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed, in order.

        Returns:
            Embeddings in the same order as ``texts``.

        Raises:
            ProviderRequestError: If the underlying request fails.
        """

    @abstractmethod
    def dimensions(self) -> int:
        """Return the vector dimensionality this provider produces."""

    @abstractmethod
    def provider_name(self) -> str:
        """Return a short, stable identifier for this provider."""


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI embeddings API."""

    _DEFAULT_ENDPOINT = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        endpoint: str = _DEFAULT_ENDPOINT,
        request_timeout: float = 30.0,
    ) -> None:
        """Initialize the provider.

        Args:
            http_client: Injected async HTTP client used for all requests.
            api_key: OpenAI API key.
            model: Embedding model identifier.
            dimensions: Expected output vector dimensionality for ``model``.
            endpoint: Override for the embeddings endpoint (useful for
                Azure OpenAI deployments or internal proxies).
            request_timeout: Per-request timeout, in seconds.

        Raises:
            ProviderConfigurationError: If ``api_key`` is empty.
        """
        if not api_key:
            raise ProviderConfigurationError("OpenAI provider requires an api_key")
        self._client = http_client
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._endpoint = endpoint
        self._timeout = request_timeout
        self._log = logger.bind(provider="openai", model=model)

    async def embed_text(self, text: str) -> EmbeddingResult:
        """See ``EmbeddingProvider.embed_text``."""
        (result,) = await self.embed_batch([text])
        return result

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """See ``EmbeddingProvider.embed_batch``.

        OpenAI's endpoint natively accepts a list of inputs, so batches are
        sent as a single request.
        """
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "input": list(texts)}

        try:
            response = await self._client.post(
                self._endpoint, json=payload, headers=headers, timeout=self._timeout
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - normalized into ProviderRequestError
            self._log.warning("embedding_request_failed", error=str(exc))
            raise ProviderRequestError(f"OpenAI embedding request failed: {exc}") from exc

        body = response.json()
        items = sorted(body.get("data", []), key=lambda item: item["index"])
        return [
            EmbeddingResult(
                vector=item["embedding"], model=self._model, provider=self.provider_name()
            )
            for item in items
        ]

    def dimensions(self) -> int:
        """See ``EmbeddingProvider.dimensions``."""
        return self._dimensions

    def provider_name(self) -> str:
        """See ``EmbeddingProvider.provider_name``."""
        return "openai"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by a local or remote Ollama server.

    Ollama's embeddings endpoint accepts one prompt per request, so
    ``embed_batch`` fans requests out concurrently, bounded by
    ``max_concurrency``, rather than relying on server-side batching.
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        *,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        base_url: str = "http://localhost:11434",
        max_concurrency: int = 8,
        request_timeout: float = 60.0,
    ) -> None:
        """Initialize the provider.

        Args:
            http_client: Injected async HTTP client used for all requests.
            model: Embedding model identifier as known to the Ollama server.
            dimensions: Expected output vector dimensionality for ``model``.
            base_url: Base URL of the Ollama server.
            max_concurrency: Maximum number of concurrent in-flight requests
                when embedding a batch.
            request_timeout: Per-request timeout, in seconds.
        """
        self._client = http_client
        self._model = model
        self._dimensions = dimensions
        self._endpoint = f"{base_url.rstrip('/')}/api/embeddings"
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout = request_timeout
        self._log = logger.bind(provider="ollama", model=model)

    async def embed_text(self, text: str) -> EmbeddingResult:
        """See ``EmbeddingProvider.embed_text``."""
        async with self._semaphore:
            payload = {"model": self._model, "prompt": text}
            try:
                response = await self._client.post(
                    self._endpoint, json=payload, timeout=self._timeout
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("embedding_request_failed", error=str(exc))
                raise ProviderRequestError(f"Ollama embedding request failed: {exc}") from exc

            body = response.json()
            vector = body.get("embedding")
            if vector is None:
                raise ProviderRequestError(
                    "Ollama response missing 'embedding' field", retryable=False
                )
            return EmbeddingResult(vector=vector, model=self._model, provider=self.provider_name())

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """See ``EmbeddingProvider.embed_batch``."""
        if not texts:
            return []
        return list(await asyncio.gather(*(self.embed_text(text) for text in texts)))

    def dimensions(self) -> int:
        """See ``EmbeddingProvider.dimensions``."""
        return self._dimensions

    def provider_name(self) -> str:
        """See ``EmbeddingProvider.provider_name``."""
        return "ollama"


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Google Gemini embeddings API."""

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        *,
        api_key: str,
        model: str = "text-embedding-004",
        dimensions: int = 768,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        request_timeout: float = 30.0,
    ) -> None:
        """Initialize the provider.

        Args:
            http_client: Injected async HTTP client used for all requests.
            api_key: Google Generative AI API key.
            model: Embedding model identifier.
            dimensions: Expected output vector dimensionality for ``model``.
            base_url: Base URL of the Generative Language API.
            request_timeout: Per-request timeout, in seconds.

        Raises:
            ProviderConfigurationError: If ``api_key`` is empty.
        """
        if not api_key:
            raise ProviderConfigurationError("Gemini provider requires an api_key")
        self._client = http_client
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout
        self._log = logger.bind(provider="gemini", model=model)

    async def embed_text(self, text: str) -> EmbeddingResult:
        """See ``EmbeddingProvider.embed_text``."""
        url = f"{self._base_url}/models/{self._model}:embedContent?key={self._api_key}"
        payload = {"content": {"parts": [{"text": text}]}}

        try:
            response = await self._client.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("embedding_request_failed", error=str(exc))
            raise ProviderRequestError(f"Gemini embedding request failed: {exc}") from exc

        body = response.json()
        vector = body.get("embedding", {}).get("values")
        if vector is None:
            raise ProviderRequestError("Gemini response missing embedding values", retryable=False)
        return EmbeddingResult(vector=vector, model=self._model, provider=self.provider_name())

    async def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """See ``EmbeddingProvider.embed_batch``.

        Uses Gemini's ``batchEmbedContents`` endpoint so a batch costs a
        single request regardless of size.
        """
        if not texts:
            return []

        url = f"{self._base_url}/models/{self._model}:batchEmbedContents?key={self._api_key}"
        requests_payload = [
            {"model": f"models/{self._model}", "content": {"parts": [{"text": text}]}}
            for text in texts
        ]

        try:
            response = await self._client.post(
                url, json={"requests": requests_payload}, timeout=self._timeout
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("embedding_batch_request_failed", error=str(exc))
            raise ProviderRequestError(f"Gemini batch embedding request failed: {exc}") from exc

        body = response.json()
        embeddings = body.get("embeddings", [])
        return [
            EmbeddingResult(vector=item["values"], model=self._model, provider=self.provider_name())
            for item in embeddings
        ]

    def dimensions(self) -> int:
        """See ``EmbeddingProvider.dimensions``."""
        return self._dimensions

    def provider_name(self) -> str:
        """See ``EmbeddingProvider.provider_name``."""
        return "gemini"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EmbeddingManager:
    """Coordinates embedding generation across one or more providers.

    This is the single entry point the rest of Kodiak should use to obtain
    embeddings. It handles provider selection and switching, retrying
    transient failures, falling back to secondary providers, validating
    returned vector dimensions, and an optional caching layer, while
    remaining completely decoupled from any vector store.

    Args:
        providers: Mapping of provider name to provider instance. The
            first entry (insertion order) is the default/primary provider.
        cache: Optional cache backend. Defaults to a no-op cache.
        retry_config: Optional retry policy. Defaults are conservative and
            suitable for production use.
        cache_key_fn: Optional override for deriving a cache key from a
            ``(provider_name, text)`` pair.

    Raises:
        ProviderConfigurationError: If ``providers`` is empty.
    """

    def __init__(
        self,
        providers: dict[str, EmbeddingProvider],
        *,
        cache: EmbeddingCache | None = None,
        retry_config: RetryConfig | None = None,
        cache_key_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        if not providers:
            raise ProviderConfigurationError("EmbeddingManager requires at least one provider")
        self._providers = dict(providers)
        self._active_provider_name = next(iter(self._providers))
        self._cache = cache or NullEmbeddingCache()
        self._retry_config = retry_config or RetryConfig()
        self._cache_key_fn = cache_key_fn or self._default_cache_key
        self._log = logger.bind(component="embedding_manager")

    @staticmethod
    def _default_cache_key(provider_name: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{provider_name}:{digest}"

    @property
    def active_provider(self) -> EmbeddingProvider:
        """Return the currently active provider instance."""
        return self._providers[self._active_provider_name]

    def use_provider(self, name: str) -> None:
        """Switch the active (default) provider.

        Args:
            name: Registered provider name to make active.

        Raises:
            ProviderConfigurationError: If ``name`` is not registered.
        """
        if name not in self._providers:
            raise ProviderConfigurationError(
                f"Unknown provider '{name}'. Registered providers: {sorted(self._providers)}"
            )
        self._active_provider_name = name
        self._log.info("provider_switched", provider=name)

    def register_provider(self, name: str, provider: EmbeddingProvider) -> None:
        """Register or replace a provider without disturbing the active one.

        This is the extension point for adding future providers at
        runtime, without needing to touch ``EmbeddingManager`` itself.

        Args:
            name: Name to register the provider under.
            provider: The provider instance.
        """
        self._providers[name] = provider
        self._log.info("provider_registered", provider=name)

    async def embed(self, text: str, *, provider: str | None = None) -> EmbeddingResult:
        """Embed a single piece of text, applying cache, retry, and fallback.

        Args:
            text: Text to embed.
            provider: Optional provider name override for this call only.

        Returns:
            The embedding result.

        Raises:
            AllProvidersExhaustedError: If every eligible provider failed.
        """
        (result,) = await self.embed_batch([text], provider=provider)
        return result

    async def embed_batch(
        self, texts: Sequence[str], *, provider: str | None = None
    ) -> list[EmbeddingResult]:
        """Embed a batch of texts, applying cache, retry, and fallback.

        Cache hits are served immediately without contacting any provider.
        Only cache misses are sent out, and only that subset is retried or
        failed over as a unit, so a single bad request never discards
        already-cached results.

        Args:
            texts: Texts to embed, in order.
            provider: Optional provider name override for this call only.

        Returns:
            Embeddings in the same order as ``texts``.

        Raises:
            AllProvidersExhaustedError: If every eligible provider failed
                for the uncached texts.
        """
        if not texts:
            return []

        target_name = provider or self._active_provider_name
        if target_name not in self._providers:
            raise ProviderConfigurationError(f"Unknown provider '{target_name}'")

        cache_keys = [self._cache_key_fn(target_name, text) for text in texts]
        results: list[EmbeddingResult | None] = [None] * len(texts)
        miss_indices: list[int] = []

        for idx, key in enumerate(cache_keys):
            cached_vector = await self._cache.get(key)
            if cached_vector is not None:
                results[idx] = EmbeddingResult(
                    vector=cached_vector, model="cached", provider=target_name
                )
            else:
                miss_indices.append(idx)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            fresh_results = await self._embed_with_fallback(miss_texts, preferred=target_name)
            for i, result in zip(miss_indices, fresh_results, strict=True):
                results[i] = result
                await self._cache.set(cache_keys[i], result.vector)

        return [result for result in results if result is not None]

    async def _embed_with_fallback(
        self, texts: Sequence[str], *, preferred: str
    ) -> list[EmbeddingResult]:
        """Try ``preferred``, then fall back through remaining providers in order."""
        ordered_names = [preferred, *(name for name in self._providers if name != preferred)]
        last_error: Exception | None = None

        for name in ordered_names:
            provider_instance = self._providers[name]
            try:
                batch_results = await self._embed_batch_with_retry(provider_instance, texts)
                self._validate_dimensions(batch_results, provider_instance)
                return batch_results
            except EmbeddingError as exc:
                self._log.warning("provider_failed_trying_next", provider=name, error=str(exc))
                last_error = exc
                continue

        raise AllProvidersExhaustedError(
            f"All embedding providers failed for {len(texts)} text(s)"
        ) from last_error

    async def _embed_batch_with_retry(
        self, provider_instance: EmbeddingProvider, texts: Sequence[str]
    ) -> list[EmbeddingResult]:
        """Call ``provider_instance.embed_batch`` with exponential-backoff retry."""
        attempt = 0
        delay = self._retry_config.base_delay_seconds
        last_error: ProviderRequestError | None = None

        while attempt < self._retry_config.max_attempts:
            attempt += 1
            try:
                return await provider_instance.embed_batch(texts)
            except ProviderRequestError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self._retry_config.max_attempts:
                    break
                self._log.warning(
                    "retrying_embedding_request",
                    provider=provider_instance.provider_name(),
                    attempt=attempt,
                    delay=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_config.max_delay_seconds)

        assert last_error is not None  # loop always sets this before exiting
        raise last_error

    @staticmethod
    def _validate_dimensions(
        results: list[EmbeddingResult], provider_instance: EmbeddingProvider
    ) -> None:
        """Ensure every returned vector matches the provider's declared dimensionality."""
        expected = provider_instance.dimensions()
        for result in results:
            actual = len(result.vector)
            if actual != expected:
                raise EmbeddingDimensionMismatchError(
                    expected=expected,
                    actual=actual,
                    provider=provider_instance.provider_name(),
                )
