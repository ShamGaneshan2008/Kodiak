from __future__ import annotations

import json
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)


class RetrievalAgent(BaseAgent):
    """Rank and filter repository chunks for a query."""

    role = AgentRole.RETRIEVAL
    capabilities = frozenset({"retrieval", "repository_context", "information_retrieval"})

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        query = str(input_.context.get("query", input_.instruction))
        raw_chunks = input_.context.get("raw_chunks", "[]")
        user_message = (
            f"Rank and filter these code chunks for the query: {query}\n\n"
            f"Chunks:\n{raw_chunks}\n\n"
            "Return a JSON array of objects with 'content', 'source', and 'score' (0.0-1.0)."
        )
        response = await self._llm.complete(
            system="You are a retrieval agent inside Kodiak. Output JSON only.",
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
        )
        raw = response.get("content", "")
        try:
            chunks = json.loads(raw)
            if not isinstance(chunks, list):
                raise ValueError("Expected a JSON array.")
            return self._make_output(
                input_,
                {"chunks": chunks},
                token_usage=response.get("usage", {}),
            )
        except Exception as exc:
            return self._make_error(input_, f"Retrieval parse failed: {exc}")
