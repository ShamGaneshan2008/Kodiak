from __future__ import annotations

from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)


class MemoryAgent(BaseAgent):
    """Query semantic and episodic memory stores."""

    role = AgentRole.MEMORY
    capabilities = frozenset({"memory", "memory_retrieval"})

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        mem_type = str(input_.context.get("memory_type", "semantic"))
        query = str(input_.context.get("query", input_.instruction))
        existing = str(input_.context.get("existing_memories", ""))
        existing_section = f"Existing Memories:\n{existing}\n\n" if existing else ""
        user_message = (
            f"{existing_section}Query {mem_type} memory for: {query}\n\n"
            "Formulate a precise response based on the memories."
        )
        response = await self._llm.complete(
            system="You are a memory agent inside Kodiak.",
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
        )
        return self._make_output(
            input_,
            {
                "memory_type": mem_type,
                "response": response.get("content", ""),
            },
            token_usage=response.get("usage", {}),
        )
