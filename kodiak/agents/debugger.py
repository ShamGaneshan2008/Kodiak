import structlog

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class DebugOutput(AgentOutput):
    result: str = ""
    root_cause: str = ""
    suggested_fix: str = ""


class DebuggerAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="debugger", description="Analyzes failures and bugs")

    async def execute(self, input_data: AgentInput) -> DebugOutput:
        self._logger.info("debugging_failure", task=input_data.task)
        error = input_data.context.get("error", "")
        code = input_data.context.get("code", "")
        prompt = self._build_prompt(code, error, input_data.task)
        raw = await self._run_with_timing(prompt)
        parts = [p.strip() for p in raw.split("###") if p.strip()]
        root_cause = parts[1] if len(parts) > 1 else raw
        fix = parts[2] if len(parts) > 2 else ""
        self._logger.info("debugging_complete", has_root_cause=bool(root_cause))
        return DebugOutput(success=True, result=raw, root_cause=root_cause, suggested_fix=fix)

    def _build_prompt(self, code: str, error: str, task: str) -> str:
        return (
            f"Analyze this error for the task: {task}\n\nCode:\n{code}\n\nError:\n{error}\n\n"
            "Respond with exactly three sections separated by '###':\n"
            "###ANALYSIS\n###ROOT_CAUSE\n###SUGGESTED_FIX"
        )
