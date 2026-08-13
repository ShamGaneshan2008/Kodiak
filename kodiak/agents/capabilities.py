"""Canonical capability definitions for Kodiak specialized agents."""

from __future__ import annotations

from kodiak.agents.base import AgentRole

ROLE_CAPABILITIES: dict[AgentRole, frozenset[str]] = {
    AgentRole.PLANNER: frozenset({"planning", "task_decomposition"}),
    AgentRole.REPOSITORY: frozenset({"repository_analysis", "repository_context"}),
    AgentRole.RETRIEVAL: frozenset({"retrieval", "repository_context", "information_retrieval"}),
    AgentRole.RESEARCH: frozenset({"research", "repository_context", "information_retrieval"}),
    AgentRole.ARCHITECT: frozenset({"architecture", "planning"}),
    AgentRole.CODER: frozenset({"write_code", "refactor", "code_generation", "code_modification"}),
    AgentRole.REVIEWER: frozenset({"code_review", "static_analysis"}),
    AgentRole.TESTER: frozenset({"run_tests", "write_tests", "test_execution", "test_analysis"}),
    AgentRole.DEBUGGER: frozenset({"debugging", "root_cause_analysis"}),
    AgentRole.REFLECTION: frozenset({"reflection", "self_improvement"}),
    AgentRole.GIT: frozenset({"git_operations", "pull_request_creation"}),
    AgentRole.MEMORY: frozenset({"memory", "memory_retrieval"}),
    AgentRole.LEARNING: frozenset({"learning", "pattern_extraction"}),
    AgentRole.EVALUATION: frozenset({"evaluation", "quality_assessment"}),
}


def default_capabilities_for_role(role: AgentRole | None) -> frozenset[str]:
    """Return default capabilities for an agent role."""
    if role is None:
        return frozenset()
    return ROLE_CAPABILITIES.get(role, frozenset({role.value}))


__all__ = ["ROLE_CAPABILITIES", "default_capabilities_for_role"]
