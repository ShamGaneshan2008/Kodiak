from __future__ import annotations

import json
from typing import Any

import structlog

from kodiak.llm.client import LLMClient

logger = structlog.get_logger(__name__)

_ENFORCEMENT_SUFFIX = (
    "\n\nCRITICAL: Your response must be valid JSON only. "
    "No prose before or after. No markdown fences. Just the raw JSON object."
)
_RETRY_PROMPT = (
    "Your previous response was not valid JSON or was missing required keys.\n"
    "Previous response: {previous}\nError: {error}\n\n"
    "Try again. Output ONLY the JSON object."
)


class StructuredOutputParser:
    def __init__(self, client: LLMClient, max_retries: int = 2) -> None:
        self._client = client
        self._max_retries = max_retries

    async def parse(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        required_keys: list[str] | None = None,
        model_preference: str = "default",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        augmented_system = (system or "") + _ENFORCEMENT_SUFFIX
        history = list(messages)
        last_raw = ""
        last_error = ""

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                history += [
                    {"role": "assistant", "content": last_raw},
                    {
                        "role": "user",
                        "content": _RETRY_PROMPT.format(previous=last_raw[:500], error=last_error),
                    },
                ]

            response = await self._client.complete(
                messages=history,
                system=augmented_system,
                model_preference=model_preference,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            last_raw = response.get("content", "")
            result, last_error = _try_parse(last_raw, required_keys)

            if result is not None:
                if attempt > 0:
                    logger.debug("structured_output.recovered", attempt=attempt)
                return result

            logger.warning(
                "structured_output.parse_failed",
                attempt=attempt,
                error=last_error,
            )

        raise ValueError(
            f"Failed to extract valid JSON after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )


def _try_parse(raw: str, required_keys: list[str] | None) -> tuple[dict[str, Any] | None, str]:
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("\n", 1)
        if len(parts) > 1:
            clean = parts[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    if not isinstance(data, dict):
        return None, f"Expected object, got {type(data).__name__}"
    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            return None, f"Missing required keys: {missing}"
    return data, ""
