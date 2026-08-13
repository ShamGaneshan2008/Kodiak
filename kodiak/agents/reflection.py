from __future__ import annotations

from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)


class ReflectionAgent(BaseAgent):
    """Analyze execution traces and extract lessons learned."""

    role = AgentRole.REFLECTION
    capabilities = frozenset({"reflection", "self_improvement"})

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        trace = str(input_.context.get("execution_trace", ""))
        outcome = str(input_.context.get("outcome", "unknown"))
        user_message = (
            f"Reflect on this task execution.\nTask: {input_.instruction}\n"
            f"Outcome: {outcome}\nTrace:\n{trace}\n\n"
            "List lessons learned, each starting with '- '."
        )
        response = await self._llm.complete(
            system="You are a reflection agent inside Kodiak.",
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
        )
        raw = response.get("content", "")
        lessons = [
            line.strip().lstrip("- ")
            for line in raw.splitlines()
            if line.strip().startswith("- ")
        ]
        return self._make_output(
            input_,
            {"reflection": raw, "lessons_learned": lessons},
            token_usage=response.get("usage", {}),
        )
