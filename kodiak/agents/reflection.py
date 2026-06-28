import structlog
from pydantic import BaseModel, Field

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class ReflectionOutput(AgentOutput):
    result: str = ""
    lessons_learned: list[str] = Field(default_factory=list)


class ReflectionAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="reflection", description="Analyzes execution for self-improvement")

    async def execute(self, input_data: AgentInput) -> ReflectionOutput:
        self._logger.info("reflecting_on_execution", task=input_data.task)
        trace = input_data.context.get("execution_trace", "")
        outcome = input_data.context.get("outcome", "unknown")

        prompt = self._build_prompt(input_data.task, trace, outcome)
        raw = await self._run_with_timing(prompt)

        lessons = [l.strip().lstrip("- ") for l in raw.split("\n") if l.strip().startswith("- ")]
        self._logger.info("reflection_complete", lessons=len(lessons))
        return ReflectionOutput(success=True, result=raw, lessons_learned=lessons)

    def _build_prompt(self, task: str, trace: str, outcome: str) -> str:
        return (
            f"Reflect on this task execution.\nTask: {task}\nOutcome: {outcome}\nTrace:\n{trace}\n\n"
            "List lessons learned, each starting with '- '."
        )