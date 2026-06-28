import structlog

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class ResearchOutput(AgentOutput):
    result: str = ""


class ResearcherAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="researcher", description="Retrieves external knowledge")

    async def execute(self, input_data: AgentInput) -> ResearchOutput:
        self._logger.info("researching_topic", task=input_data.task)
        context = input_data.context.get("existing_context", "")
        prompt = self._build_prompt(input_data.task, context)
        findings = await self._run_with_timing(prompt)
        self._logger.info("research_completed", chars=len(findings))
        return ResearchOutput(success=True, result=findings)

    def _build_prompt(self, query: str, context: str) -> str:
        ctx_section = f"Existing Context:\n{context}\n\n" if context else ""
        return (
            f"{ctx_section}Research the following topic or question:\n{query}\n\n"
            "Provide a comprehensive summary of findings, APIs, or best practices."
        )