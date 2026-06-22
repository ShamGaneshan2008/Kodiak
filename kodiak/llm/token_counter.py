from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_CHARS_PER_TOKEN_FALLBACK = 4

_MODEL_PRICING: dict[str, dict[str, float]] = {
    # per million tokens
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.8, "output": 4.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


@lru_cache(maxsize=8)
def _get_encoding(model: str) -> Any:
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        enc = _get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // _CHARS_PER_TOKEN_FALLBACK


def count_messages_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    system: str | None = None,
) -> int:
    total = 0
    if system:
        total += count_tokens(system, model) + 4
    for msg in messages:
        total += count_tokens(msg.get("content", ""), model) + 4
    total += 2
    return total


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return round(
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"],
        6,
    )