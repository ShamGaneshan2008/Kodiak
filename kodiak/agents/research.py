from kodiak.agents.base import AgentContext, AgentResult, BaseAgent


class ResearchAgent(BaseAgent):
    name = "research"

    async def run(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status="completed",
            output={
                "task_id": context.task_id,
                "repository": context.repository,
                "message": "Collect external or repository context.",
            },
        )
