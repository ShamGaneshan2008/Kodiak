"""Adapters bridging discovered BaseAgent instances to Kodiak registry and manager contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kodiak.agents.base import AgentInput, AgentOutput, BaseAgent


@dataclass(slots=True)
class DiscoveredAgentHandle:
    """Wraps a BaseAgent so it satisfies AgentRegistry's AgentProtocol."""

    agent_id: str
    _agent: BaseAgent

    @property
    def agent(self) -> BaseAgent:
        return self._agent


class BaseAgentAdapter:
    """Adapts a BaseAgent for AgentManager's Agent protocol."""

    def __init__(self, agent: BaseAgent) -> None:
        self._agent = agent
        self.role = agent.role
        self.agent_id = agent.agent_id
        self.name = agent.agent_id
        self.capabilities = agent.resolved_capabilities()

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
        output = await self._agent.run(agent_input)
        if isinstance(output, AgentOutput):
            return output.result if output.success else output
        return output

    async def health_check(self) -> bool:
        return await self._agent.health_check()


class ManagerAgentAdapter(BaseAgentAdapter):
    """Backward-compatible alias for registry-discovered agent handles."""

    def __init__(
        self,
        handle: DiscoveredAgentHandle,
        *,
        capabilities: frozenset[str] | None = None,
    ) -> None:
        super().__init__(handle.agent)
        self.agent_id = handle.agent_id
        self.name = handle.agent_id
        if capabilities is not None:
            self.capabilities = capabilities


__all__ = [
    "BaseAgentAdapter",
    "DiscoveredAgentHandle",
    "ManagerAgentAdapter",
]
