from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import structlog

from kodiak.config.feature_flags import get_feature_flags
from kodiak.config.metrics import (
    llm_request_duration_seconds,
    llm_requests_total,
    llm_tokens_used_total,
)
from kodiak.config.settings import LLMProvider, get_settings
from kodiak.llm.cost_optimizer import CostOptimizer
from kodiak.llm.fallback import FallbackChain
from kodiak.llm.providers.anthropic import AnthropicProvider
from kodiak.llm.providers.base import LLMMessage, ModelTier
from kodiak.llm.providers.openai import OpenAIProvider

logger = structlog.get_logger(__name__)


def _resolve_tier(preference: str) -> ModelTier:
    match preference.lower():
        case "fast":
            return ModelTier.FAST
        case "strong":
            return ModelTier.STRONG
        case _:
            return ModelTier.DEFAULT


def _coerce_messages(
    messages: list[dict[str, str]] | list[LLMMessage],
) -> list[LLMMessage]:
    if not messages:
        return []
    if isinstance(messages[0], LLMMessage):
        return list(messages)  # type: ignore[arg-type]
    return [LLMMessage(role=m["role"], content=m["content"]) for m in messages]  # type: ignore[index]


class LLMRouter:
    def __init__(self, cost_optimizer: CostOptimizer | None = None) -> None:
        self._settings = get_settings()
        self._flags = get_feature_flags()
        self._cost_optimizer = cost_optimizer or CostOptimizer()

        anthropic = AnthropicProvider()
        openai = OpenAIProvider()

        providers = (
            [anthropic, openai]
            if self._settings.primary_llm_provider == LLMProvider.ANTHROPIC
            else [openai, anthropic]
        )
        fallback_enabled = self._flags.is_enabled("llm.fallback.enabled")
        self._chain = FallbackChain(
            providers=providers if fallback_enabled else providers[:1],
            max_retries_per_provider=self._settings.llm_max_retries,
        )

    async def complete(
        self,
        messages: list[dict[str, str]] | list[LLMMessage],
        system: str | None = None,
        model_preference: str = "default",
        max_tokens: int | None = None,
        temperature: float = 0.2,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tier = _resolve_tier(model_preference)

        if task_id and self._flags.is_enabled("llm.cost_optimizer.enabled"):
            tier = await self._cost_optimizer.recommend_tier(task_id, tier)

        llm_messages = _coerce_messages(messages)
        effective_max_tokens = max_tokens or self._settings.llm_max_tokens

        start = time.monotonic()
        try:
            response = await self._chain.complete(
                messages=llm_messages,
                tier=tier,
                system=system,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception:
            llm_requests_total.labels(provider="unknown", model="unknown", status="error").inc()
            raise

        elapsed = time.monotonic() - start
        llm_requests_total.labels(
            provider=response.provider, model=response.model, status="success"
        ).inc()
        llm_request_duration_seconds.labels(
            provider=response.provider, model=response.model
        ).observe(elapsed)
        llm_tokens_used_total.labels(
            provider=response.provider, model=response.model, token_type="input"
        ).inc(response.input_tokens)
        llm_tokens_used_total.labels(
            provider=response.provider, model=response.model, token_type="output"
        ).inc(response.output_tokens)

        if task_id:
            await self._cost_optimizer.record_usage(
                task_id=task_id,
                model=response.model,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage,
            "stop_reason": response.stop_reason,
        }


@lru_cache(maxsize=1)
def get_llm_router() -> LLMRouter:
    return LLMRouter()