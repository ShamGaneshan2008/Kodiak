# kodiak/orchestration/planning/matcher.py
"""Agent capability and tool matching component for planning tasks."""

from __future__ import annotations

from typing import Any

import structlog

from kodiak.orchestration.state import AgentRole

logger = structlog.get_logger(__name__)

__all__ = ["AgentCapabilityMatcher"]

# Canonical mapping from task types to required capability strings and default agent roles
_TASK_TYPE_TO_CAPABILITIES: dict[str, tuple[list[str], AgentRole]] = {
    "inspection": (["repository_context", "file_reading"], AgentRole.RESEARCHER),
    "research": (["repository_search", "symbol_index"], AgentRole.RESEARCHER),
    "implementation": (["code_generation", "file_editing"], AgentRole.CODER),
    "test": (["test_generation", "test_execution"], AgentRole.TESTER),
    "documentation": (["documentation_edit", "code_generation"], AgentRole.CODER),
    "review": (["code_review", "static_analysis"], AgentRole.REVIEWER),
    "debugging": (["failure_analysis", "root_cause_analysis"], AgentRole.DEBUGGER),
}


class AgentCapabilityMatcher:
    """Matches task requirements against registered agent capabilities and tools."""

    def __init__(self, agent_manager: Any | None = None) -> None:
        """Initialize AgentCapabilityMatcher.

        Args:
            agent_manager: Optional AgentManager instance for dynamic capability inspection.
        """
        self.agent_manager = agent_manager

    def determine_capabilities(
        self, task_type: str, tools: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """Determine required capability strings for a given task type and tools list.

        Args:
            task_type: Task type label.
            tools: List of tool specification dicts.

        Returns:
            List of required capability strings.
        """
        caps: set[str] = set()
        default_caps, _ = _TASK_TYPE_TO_CAPABILITIES.get(
            task_type.lower(), (["code_generation"], AgentRole.CODER)
        )
        caps.update(default_caps)

        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("required_capability"):
                    caps.add(str(tool["required_capability"]))

        return sorted(caps)

    def match_agent_role(self, task_type: str) -> str:
        """Map a task type label to a canonical AgentRole string.

        Args:
            task_type: Task type label.

        Returns:
            AgentRole string value.
        """
        _, role = _TASK_TYPE_TO_CAPABILITIES.get(task_type.lower(), ([], AgentRole.CODER))
        return role.value

    def match_best_agent(
        self,
        task_type: str,
        required_capabilities: list[str] | None = None,
    ) -> str:
        """Find the optimal agent name for a task using AgentManager if available.

        Args:
            task_type: Task type string.
            required_capabilities: Explicit capability list.

        Returns:
            Agent name or role string.
        """
        caps = required_capabilities or self.determine_capabilities(task_type)
        role_str = self.match_agent_role(task_type)

        if self.agent_manager is not None and hasattr(self.agent_manager, "get_agent"):
            # Check if agent manager has an agent matching the capability or role
            try:
                for agent_name in [
                    "coder",
                    "reviewer",
                    "tester",
                    "researcher",
                    "debugger",
                    "planner",
                ]:
                    agent = self.agent_manager.get_agent(agent_name)
                    if agent and hasattr(agent, "capabilities"):
                        if set(caps).issubset(getattr(agent, "capabilities", set())):
                            return str(agent_name)
            except Exception:
                pass

        return role_str
