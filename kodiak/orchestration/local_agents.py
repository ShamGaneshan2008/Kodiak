"""Shared inventory of concrete Kodiak agent implementations."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    name: str
    role: str
    status: str
    description: str
    backend: str
    module: str


_AGENTS = (
    AgentDescriptor(
        "PlannerAgent",
        "planner",
        "available",
        "Builds structured plans; the advanced implementation needs an LLM client.",
        "LLM-backed",
        "kodiak.agents.planner",
    ),
    AgentDescriptor(
        "RepositoryAgent",
        "repository analysis",
        "available",
        "Scans repository structure and project signals locally.",
        "deterministic",
        "kodiak.agents.repository",
    ),
    AgentDescriptor(
        "CoderAgent",
        "code editing",
        "available",
        "Builds and validates patches; the advanced implementation needs an LLM client.",
        "LLM-backed",
        "kodiak.agents.coder",
    ),
    AgentDescriptor(
        "TesterAgent",
        "testing",
        "available",
        "Discovers and executes tests; optional test generation is LLM-backed.",
        "deterministic",
        "kodiak.agents.tester",
    ),
    AgentDescriptor(
        "ReviewerAgent",
        "review",
        "available",
        "Reviews patches with deterministic checks and optional LLM analysis.",
        "LLM-backed",
        "kodiak.agents.reviewer",
    ),
    AgentDescriptor(
        "GitAgent",
        "git",
        "available",
        "Summarizes local changes and prepares commit metadata.",
        "deterministic",
        "kodiak.agents.git",
    ),
    AgentDescriptor(
        "MemoryAgent",
        "memory",
        "available",
        "Queries advanced memory stores through an injected LLM client.",
        "LLM-backed",
        "kodiak.agents.memory_agent",
    ),
)


def local_agent_descriptors() -> list[AgentDescriptor]:
    """Return descriptors backed by importable implementation modules."""
    return [agent for agent in _AGENTS if importlib.util.find_spec(agent.module) is not None]
