import json

import structlog
from pydantic import BaseModel, Field

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class Pattern(BaseModel):
    name: str
    description: str
    reward: float = 0.0


class LearningOutput(AgentOutput):
    result: list[Pattern] = Field(default_factory=list)


class LearningAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="learning", description="Extracts patterns and updates learning")

    async def execute(self, input_data: AgentInput) -> LearningOutput:
        self._logger.info("learning_from_execution", task=input_data.task)
        reward = input_data.context.get("reward", 0.0)
        trace = input_data.context.get("execution_trace", "")

        prompt = (
            f"Extract reusable coding patterns from this execution.\n"
            f"Task: {input_data.task}\nReward: {reward}\nTrace:\n{trace}\n\n"
            "Output a JSON array of patterns with 'name', 'description', 'reward'."
        )
        raw = await self._run_with_timing(prompt)

        try:
            patterns = [Pattern(**p) for p in json.loads(raw)]
            return LearningOutput(success=True, result=patterns)
        except Exception as e:
            self._logger.error("learning_parse_failed", error=str(e))
            return LearningOutput(success=False, error=str(e))