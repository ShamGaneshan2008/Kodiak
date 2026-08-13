from __future__ import annotations

import json
from typing import Any

import structlog

from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent

logger = structlog.get_logger(__name__)


class EvaluationAgent(BaseAgent):
    """Evaluate generated output quality and confidence."""

    role = AgentRole.EVALUATION
    capabilities = frozenset({"evaluation", "quality_assessment"})

    def __init__(self, llm_client: Any) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        output = str(input_.context.get("output", ""))
        goal = str(input_.context.get("goal", input_.instruction))
        user_message = (
            f"Evaluate this output against the goal.\nGoal: {goal}\nOutput: {output}\n\n"
            'Respond in JSON: {"analysis": "...", "score": 0.0-1.0, "confidence": 0.0-1.0}'
        )
        response = await self._llm.complete(
            system="You are an evaluation agent inside Kodiak. Output JSON only.",
            messages=[{"role": "user", "content": user_message}],
            model_preference="default",
        )
        raw = response.get("content", "")
        try:
            data = json.loads(raw)
            return self._make_output(
                input_,
                {
                    "analysis": data.get("analysis", ""),
                    "score": float(data.get("score", 0.0)),
                    "confidence": float(data.get("confidence", 0.0)),
                },
                token_usage=response.get("usage", {}),
            )
        except Exception as exc:
            return self._make_error(input_, f"Evaluation parse failed: {exc}")
