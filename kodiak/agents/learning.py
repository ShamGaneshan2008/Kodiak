from __future__ import annotations

import json
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)


class LearningAgent(BaseAgent):
    """Extract reusable patterns from completed executions."""

    role = AgentRole.LEARNING
    capabilities = frozenset({"learning", "pattern_extraction"})

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        reward = input_.context.get("reward", 0.0)
        trace = str(input_.context.get("execution_trace", ""))
        user_message = (
            f"Extract reusable coding patterns from this execution.\n"
            f"Task: {input_.instruction}\nReward: {reward}\nTrace:\n{trace}\n\n"
            "Output a JSON array of patterns with 'name', 'description', 'reward'."
        )
        response = await self._llm.complete(
            system="You are a learning agent inside Kodiak. Output JSON only.",
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
        )
        raw = response.get("content", "")
        try:
            patterns = json.loads(raw)
            if not isinstance(patterns, list):
                raise ValueError("Expected a JSON array.")
            return self._make_output(
                input_,
                {"patterns": patterns},
                token_usage=response.get("usage", {}),
            )
        except Exception as exc:
            return self._make_error(input_, f"Learning parse failed: {exc}")
