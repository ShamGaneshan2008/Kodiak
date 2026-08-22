"""Tests for automatic agent discovery, registry integration, and selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from kodiak.agents.adapters import DiscoveredAgentHandle, ManagerAgentAdapter
from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent
from kodiak.agents.discovery import (
    AgentDiscovery,
    DiscoveryRejectReason,
    DiscoveryResult,
)
from kodiak.agents.manager import AgentManager
from kodiak.agents.registry import AgentAlreadyRegisteredError, AgentRegistry

# ---------------------------------------------------------------------------
# Test-only agent implementations (loaded via AgentDiscovery.modules=)
# ---------------------------------------------------------------------------


class _ValidAlphaAgent(BaseAgent):
    role = AgentRole.PLANNER
    capabilities = frozenset({"planning", "alpha"})

    async def _run(self, input_: AgentInput) -> AgentOutput:
        return self._make_output(input_, {"agent": "alpha"})


class _ValidBetaAgent(BaseAgent):
    role = AgentRole.CODER
    capabilities = frozenset({"write_code", "beta"})

    async def _run(self, input_: AgentInput) -> AgentOutput:
        return self._make_output(input_, {"agent": "beta"})


class _AbstractAgent(BaseAgent, ABC):
    role = AgentRole.REVIEWER

    @abstractmethod
    async def _run(self, input_: AgentInput) -> AgentOutput:
        raise NotImplementedError


class _MissingRoleAgent(BaseAgent):
    async def _run(self, input_: AgentInput) -> AgentOutput:
        return self._make_output(input_, {})


class _ZDuplicatePlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    async def _run(self, input_: AgentInput) -> AgentOutput:
        return self._make_output(input_, {"agent": "duplicate"})


class _NeedsDependencyAgent(BaseAgent):
    role = AgentRole.TESTER

    def __init__(self, llm_client: object) -> None:
        super().__init__()
        self._llm = llm_client

    async def _run(self, input_: AgentInput) -> AgentOutput:
        return self._make_output(input_, {"has_llm": self._llm is not None})


class NotAnAgent:
    """Unrelated class that must be ignored."""


TEST_MODULE = __name__


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture
def discovery(registry: AgentRegistry) -> AgentDiscovery:
    return AgentDiscovery(registry, modules=[TEST_MODULE])


@pytest.mark.asyncio
async def test_valid_agent_discovery(discovery: AgentDiscovery, registry: AgentRegistry) -> None:
    result = await discovery.discover_and_register()
    assert "planner" in result.registered
    handle = await registry.get("planner")
    assert isinstance(handle, DiscoveredAgentHandle)
    assert handle.agent_id == "planner"


@pytest.mark.asyncio
async def test_multiple_valid_agents(discovery: AgentDiscovery) -> None:
    result = await discovery.discover_and_register()
    assert "planner" in result.registered
    assert "coder" in result.registered
    assert result.registered == tuple(sorted(result.registered))


@pytest.mark.asyncio
async def test_invalid_candidate_rejection(discovery: AgentDiscovery) -> None:
    result = await discovery.discover_and_register()
    assert "NotAnAgent" not in result.registered
    assert all("NotAnAgent" not in agent_id for agent_id in result.registered)


@pytest.mark.asyncio
async def test_abstract_candidate_rejection(discovery: AgentDiscovery) -> None:
    result = await discovery.discover_and_register()
    abstract = [r for r in result.rejections if r.reason is DiscoveryRejectReason.ABSTRACT]
    assert any("_AbstractAgent" in r.qualname for r in abstract)


@pytest.mark.asyncio
async def test_missing_metadata_handling(discovery: AgentDiscovery) -> None:
    result = await discovery.discover_and_register()
    missing = [r for r in result.rejections if r.reason is DiscoveryRejectReason.MISSING_ROLE]
    assert any("_MissingRoleAgent" in r.qualname for r in missing)


@pytest.mark.asyncio
async def test_duplicate_agent_handling(discovery: AgentDiscovery) -> None:
    result = await discovery.discover_and_register()
    duplicates = [r for r in result.rejections if r.reason is DiscoveryRejectReason.DUPLICATE]
    assert any("_ZDuplicatePlannerAgent" in r.qualname for r in duplicates)
    assert result.registered.count("planner") == 1


@pytest.mark.asyncio
async def test_import_failure_handling(registry: AgentRegistry) -> None:
    discovery = AgentDiscovery(
        registry,
        modules=["kodiak.agents.this_module_does_not_exist_xyz"],
    )
    result = await discovery.discover_and_register()
    assert result.registered == ()
    assert len(result.import_errors) == 1
    assert "this_module_does_not_exist_xyz" in result.import_errors[0].module


@pytest.mark.asyncio
async def test_empty_discovery(registry: AgentRegistry) -> None:
    discovery = AgentDiscovery(registry, modules=[])
    result = await discovery.discover_and_register()
    assert result == DiscoveryResult()


@pytest.mark.asyncio
async def test_deterministic_ordering(discovery: AgentDiscovery, registry: AgentRegistry) -> None:
    first = await discovery.discover_and_register()
    await registry.clear()
    second = await discovery.discover_and_register()
    assert first.registered == second.registered


@pytest.mark.asyncio
async def test_successful_registration_into_registry(
    discovery: AgentDiscovery,
    registry: AgentRegistry,
) -> None:
    await discovery.discover_and_register()
    metadata = await registry.list_agents()
    ids = [item.agent_id for item in metadata]
    assert "planner" in ids
    assert "coder" in ids


@pytest.mark.asyncio
async def test_agent_manager_can_access_discovered_agents(
    discovery: AgentDiscovery,
    registry: AgentRegistry,
) -> None:
    await discovery.discover_and_register()
    manager = AgentManager()
    registered = await discovery.register_with_manager(manager)
    assert "planner" in registered
    assert "coder" in registered

    class _Task:
        task_id = "t-1"
        task_type = "plan something"
        required_capabilities = frozenset({"planning"})
        priority = __import__(
            "kodiak.db.models.task", fromlist=["TaskPriority"]
        ).TaskPriority.MEDIUM
        attempt = 1
        allow_fallback = True
        health_check_required = False

    agent = await manager.select_agent(_Task())
    assert agent.name == "planner"


@pytest.mark.asyncio
async def test_agent_selector_can_select_discovered_agent(
    discovery: AgentDiscovery,
    registry: AgentRegistry,
) -> None:
    await discovery.discover_and_register()
    manager = AgentManager(registry=registry)
    await discovery.register_with_manager(manager)

    class _Task:
        task_id = "t-1"
        task_type = "implement"
        required_capabilities = frozenset({"write_code"})
        priority = __import__(
            "kodiak.db.models.task", fromlist=["TaskPriority"]
        ).TaskPriority.MEDIUM

    agent = await manager.select_agent(_Task())
    assert agent.name == "coder"
    handle = await registry.get("coder")
    assert isinstance(handle, DiscoveredAgentHandle)


@pytest.mark.asyncio
async def test_missing_dependency_rejection(registry: AgentRegistry) -> None:
    discovery = AgentDiscovery(registry, modules=[TEST_MODULE], dependencies={})
    result = await discovery.discover_and_register()
    missing = [
        r for r in result.rejections if r.reason is DiscoveryRejectReason.MISSING_DEPENDENCIES
    ]
    assert any("_NeedsDependencyAgent" in r.qualname for r in missing)


@pytest.mark.asyncio
async def test_dependency_provided_registers_agent(registry: AgentRegistry) -> None:
    discovery = AgentDiscovery(
        registry,
        modules=[TEST_MODULE],
        dependencies={"llm_client": object()},
    )
    result = await discovery.discover_and_register()
    assert "tester" in result.registered


@pytest.mark.asyncio
async def test_manager_agent_adapter_executes(
    discovery: AgentDiscovery,
    registry: AgentRegistry,
) -> None:
    await discovery.discover_and_register()
    handle = await registry.get("planner")
    adapter = ManagerAgentAdapter(handle, capabilities=frozenset({"planning"}))

    class _Task:
        task_id = "task-99"
        task_type = "create plan"

    output = await adapter.execute(_Task())
    assert output == {"agent": "alpha"}


@pytest.mark.asyncio
async def test_duplicate_registry_registration(registry: AgentRegistry) -> None:
    discovery = AgentDiscovery(registry, modules=[TEST_MODULE])

    async def _register_planner() -> None:
        await registry.register(
            "planner",
            factory=lambda: DiscoveredAgentHandle(agent_id="planner", _agent=_ValidAlphaAgent()),
            capabilities=("planning",),
        )

    await _register_planner()
    result = await discovery.discover_and_register(replace=False)
    duplicates = [r for r in result.rejections if r.reason is DiscoveryRejectReason.DUPLICATE]
    assert any("planner" in r.detail for r in duplicates)

    with pytest.raises(AgentAlreadyRegisteredError):
        await _register_planner()
