from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from kodiak.llm.router import LLMRouter, _coerce_messages, _resolve_tier, get_llm_router

logger = structlog.get_logger(__name__)


class LLMClient:
    """Facade injected into every agent. Never talks to providers directly."""

    def __init__(
        self,
        router: LLMRouter | None = None,
        task_id: str | None = None,
    ) -> None:
        self._router = router or get_llm_router()
        self._task_id = task_id

    def with_task(self, task_id: str) -> LLMClient:
        return LLMClient(router=self._router, task_id=task_id)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model_preference: str = "default",
        max_tokens: int | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._router.complete(
            messages=messages,
            system=system,
            model_preference=model_preference,
            max_tokens=max_tokens,
            temperature=temperature,
            task_id=self._task_id,
            **kwargs,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model_preference: str = "default",
        max_tokens: int | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> AsyncIterator[str]:

        tier = _resolve_tier(model_preference)
        llm_messages = _coerce_messages(messages)
        provider = self._router._chain._providers[0]
        model = provider.model_for_tier(tier)

        stream_ctx = await provider.stream(
            messages=llm_messages,
            model=model,
            system=system,
            max_tokens=max_tokens or 4096,
            temperature=temperature,
            **kwargs,
        )
        async with stream_ctx as stream:
            async for text in stream.text_stream:
                yield text


def get_llm_client(task_id: str | None = None) -> LLMClient:
    return LLMClient(router=get_llm_router(), task_id=task_id)
