from kodiak.agents.base import AgentContext, AgentResult, BaseAgent


class TesterAgent(BaseAgent):
    name = "tester"

    async def run(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status="completed",
            output={
                "task_id": context.task_id,
                "repository": context.repository,
                "message": "Run or propose tests for the task.",
            },
        )
