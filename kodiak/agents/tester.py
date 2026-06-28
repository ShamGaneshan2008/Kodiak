import structlog
from pydantic import BaseModel

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent, LLMClient

logger = structlog.get_logger(__name__)


class TestOutput(AgentOutput):
    result: str = ""
    test_file_path: str = ""


class TesterAgent(BaseAgent):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, name="tester", description="Generates and runs tests")

    async def execute(self, input_data: AgentInput) -> TestOutput:
        self._logger.info("generating_tests", task=input_data.task)
        code = input_data.context.get("code", "")
        language = input_data.context.get("language", "python")
        prompt = self._build_prompt(code, input_data.task, language)
        test_code = await self._run_with_timing(prompt)
        path = f"test_{input_data.context.get('filename', 'generated')}.{language}"
        self._logger.info("tests_generated", path=path)
        return TestOutput(success=True, result=test_code, test_file_path=path)

    def _build_prompt(self, code: str, task: str, language: str) -> str:
        return (
            f"Write {language} tests for the following code related to this task: {task}\n\n"
            f"Code:\n{code}\n\nReturn ONLY the raw test code."
        )