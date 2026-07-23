import json

import structlog
from pydantic import BaseModel, Field

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class Chunk(BaseModel):
    content: str
    source: str
    score: float


class RetrievalOutput(AgentOutput):
    result: list[Chunk] = Field(default_factory=list)


class RetrievalAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="retrieval", description="Performs RAG and context retrieval")

    async def execute(self, input_data: AgentInput) -> RetrievalOutput:
        self._logger.info("retrieving_context", task=input_data.task)
        query = input_data.context.get("query", input_data.task)
        raw_chunks = input_data.context.get("raw_chunks", "[]")

        prompt = (
            f"Rank and filter these code chunks for the query: {query}\n\nChunks:\n{raw_chunks}\n\n"
            "Return a JSON array of objects with 'content', 'source', and 'score' (0.0-1.0)."
        )
        raw = await self._run_with_timing(prompt)

        try:
            chunks = [Chunk(**c) for c in json.loads(raw)]
            return RetrievalOutput(success=True, result=chunks)
        except Exception as e:
            self._logger.error("retrieval_parse_failed", error=str(e))
            return RetrievalOutput(success=False, error=str(e))
