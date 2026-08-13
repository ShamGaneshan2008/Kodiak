from __future__ import annotations

from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a debugging agent inside Kodiak. Analyze failures and propose minimal fixes.
Respond with exactly three sections separated by '###':
###ANALYSIS
###ROOT_CAUSE
###SUGGESTED_FIX
"""


class DebuggerAgent(BaseAgent):
    """Analyze failures and suggest fixes."""

    role = AgentRole.DEBUGGER
    capabilities = frozenset({"debugging", "root_cause_analysis"})

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        error = str(input_.context.get("error", ""))
        code = str(input_.context.get("code", ""))
        user_message = (
            f"Analyze this error for the task: {input_.instruction}\n\n"
            f"Code:\n{code}\n\nError:\n{error}"
        )
        response = await self._llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
        )
        raw = response.get("content", "")
        parts = [part.strip() for part in raw.split("###") if part.strip()]
        return self._make_output(
            input_,
            {
                "analysis": parts[0] if parts else raw,
                "root_cause": parts[1] if len(parts) > 1 else "",
                "suggested_fix": parts[2] if len(parts) > 2 else "",
            },
            token_usage=response.get("usage", {}),
        )
