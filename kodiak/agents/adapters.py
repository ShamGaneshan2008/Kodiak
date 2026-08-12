"""Adapters bridging discovered BaseAgent instances to Kodiak registry and manager contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kodiak.agents.base import AgentInput, BaseAgent


@dataclass(slots=True)
class DiscoveredAgentHandle:
    """Wraps a BaseAgent so it satisfies AgentRegistry's AgentProtocol."""

    agent_id: str
    _agent: BaseAgent

    @property
    def agent(self) -> BaseAgent:
        return self._agent


class ManagerAgentAdapter:
    """Adapts a discovered agent for AgentManager's Agent protocol."""

    def __init__(
        self,
        handle: DiscoveredAgentHandle,
        *,
        capabilities: frozenset[str],
    ) -> None:
        self.name = handle.agent_id
        self.capabilities = capabilities
        self._handle = handle

    async def execute(self, task: Any) -> Any:
        instruction = getattr(task, "task_type", "") or getattr(task, "instruction", "")
        project_id = getattr(task, "project_id", "default")
        context = getattr(task, "context", None) or {}
        agent_input = AgentInput(
            task_id=str(getattr(task, "task_id", "")),
            project_id=str(project_id),
            instruction=str(instruction),
            context=dict(context) if isinstance(context, dict) else {},
        )
        output = await self._handle.agent.run(agent_input)
        return output.result if output.success else output

    async def health_check(self) -> bool:
        return True


__all__ = ["DiscoveredAgentHandle", "ManagerAgentAdapter"]
