import json

import structlog
from pydantic import Field

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class EvaluationOutput(AgentOutput):
    result: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(
            llm, name="evaluation", description="Evaluates output quality and confidence"
        )

    async def execute(self, input_data: AgentInput) -> EvaluationOutput:
        self._logger.info("evaluating_output", task=input_data.task)
        output = input_data.context.get("output", "")
        goal = input_data.context.get("goal", input_data.task)

        prompt = (
            f"Evaluate this output against the goal.\nGoal: {goal}\nOutput: {output}\n\n"
            'Respond in JSON: {"analysis": "...", "score": 0.0-1.0, "confidence": 0.0-1.0}'
        )
        raw = await self._run_with_timing(prompt)

        try:
            data = json.loads(raw)
            return EvaluationOutput(
                success=True,
                result=data.get("analysis", ""),
                score=float(data.get("score", 0.0)),
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception as e:
            self._logger.error("evaluation_parse_failed", error=str(e))
            return EvaluationOutput(success=False, error=str(e))
