import structlog
from pydantic import BaseModel

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class MemoryOutput(AgentOutput):
    result: str = ""
    memory_type: str = ""


class MemoryAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="memory_agent", description="Interfaces with memory systems")

    async def execute(self, input_data: AgentInput) -> MemoryOutput:
        self._logger.info("accessing_memory", task=input_data.task)
        mem_type = input_data.context.get("memory_type", "semantic")
        query = input_data.context.get("query", input_data.task)

        prompt = self._build_prompt(query, mem_type, input_data.context.get("existing_memories", ""))
        result = await self._run_with_timing(prompt)

        self._logger.info("memory_access_complete", memory_type=mem_type)
        return MemoryOutput(success=True, result=result, memory_type=mem_type)

    def _build_prompt(self, query: str, mem_type: str, existing: str) -> str:
        existing_section = f"Existing Memories:\n{existing}\n\n" if existing else ""
        return (
            f"{existing_section}Query {mem_type} memory for: {query}\n\n"
            "Formulate a precise response based on the memories."
        )