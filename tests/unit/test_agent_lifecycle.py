"""Tests for agent lifecycle management."""

from __future__ import annotations

import asyncio

import pytest

from kodiak.agents.adapters import DiscoveredAgentHandle, ManagerAgentAdapter
from kodiak.agents.base import AgentInput, AgentOutput, AgentRole, BaseAgent
from kodiak.agents.discovery import AgentDiscovery
from kodiak.agents.lifecycle import (
    AgentLifecycleHealth,
    AgentLifecycleManager,
    AgentLifecycleState,
    InvalidLifecycleTransitionError,
    LifecycleOperationConflictError,
    LifecycleOperationError,
)
from kodiak.agents.manager import AgentManager
from kodiak.agents.registry import AgentRegistry
from kodiak.agents.selector import AgentSelector


class _LifecycleAgent(BaseAgent):
    role = AgentRole.PLANNER
    capabilities = frozenset({"planning"})

    def __init__(self) -> None:
        super().__init__()
        self.initialize_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.shutdown_calls = 0
        self.fail_initialize = False
        self.fail_start = False
        self.fail_health = False
        self._running = False

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self.fail_initialize:
            raise RuntimeError("init failed")

    async def start(self) -> None:
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("start failed")
        self._running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._running = False

    async def health_check(self) -> bool:
        return not self.fail_health

    async def _run(self, input_: AgentInput) -> AgentOutput:
        return self._make_output(input_, {"ok": True})




TEST_MODULE = __name__


async def _register_agent(registry: AgentRegistry, agent_id: str, agent: BaseAgent) -> None:
    await registry.register(
        agent_id,
        instance=DiscoveredAgentHandle(agent_id=agent_id, _agent=agent),
        capabilities=tuple(getattr(agent, "capabilities", (agent_id,))),
    )


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture
def lifecycle(registry: AgentRegistry) -> AgentLifecycleManager:
    return AgentLifecycleManager(registry)


@pytest.mark.asyncio
async def test_initial_state(lifecycle: AgentLifecycleManager, registry: AgentRegistry) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    assert lifecycle.get_state("planner") is AgentLifecycleState.DISCOVERED


@pytest.mark.asyncio
async def test_successful_initialization(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    state = await lifecycle.initialize("planner")
    assert state is AgentLifecycleState.READY
    assert agent.initialize_calls == 1


@pytest.mark.asyncio
async def test_initialization_failure(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    agent.fail_initialize = True
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    with pytest.raises(LifecycleOperationError):
        await lifecycle.initialize("planner")
    assert lifecycle.get_state("planner") is AgentLifecycleState.INITIALIZATION_FAILED


@pytest.mark.asyncio
async def test_successful_start(lifecycle: AgentLifecycleManager, registry: AgentRegistry) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    state = await lifecycle.start("planner")
    assert state is AgentLifecycleState.RUNNING
    assert agent.start_calls == 1


@pytest.mark.asyncio
async def test_successful_stop(lifecycle: AgentLifecycleManager, registry: AgentRegistry) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    await lifecycle.start("planner")
    state = await lifecycle.stop("planner")
    assert state is AgentLifecycleState.READY
    assert agent.stop_calls == 1


@pytest.mark.asyncio
async def test_successful_shutdown(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    state = await lifecycle.shutdown("planner")
    assert state is AgentLifecycleState.STOPPED
    assert agent.shutdown_calls == 1


@pytest.mark.asyncio
async def test_restart(lifecycle: AgentLifecycleManager, registry: AgentRegistry) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    await lifecycle.start("planner")
    state = await lifecycle.restart("planner")
    assert state is AgentLifecycleState.RUNNING
    assert agent.start_calls >= 2


@pytest.mark.asyncio
async def test_invalid_transition_rejection(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    with pytest.raises(InvalidLifecycleTransitionError):
        await lifecycle.start("planner")


@pytest.mark.asyncio
async def test_duplicate_initialize_protection(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    state = await lifecycle.initialize("planner")
    assert state is AgentLifecycleState.READY
    assert agent.initialize_calls == 1


@pytest.mark.asyncio
async def test_concurrent_start_protection(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    lifecycle._records["planner"].operation_in_progress = True
    with pytest.raises(LifecycleOperationConflictError):
        await lifecycle.start("planner")


@pytest.mark.asyncio
async def test_concurrent_stop_protection(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    await lifecycle.start("planner")

    async def slow_stop() -> None:
        lifecycle._records["planner"].operation_in_progress = True
        await asyncio.sleep(0.05)
        lifecycle._records["planner"].operation_in_progress = False

    stopper = asyncio.create_task(slow_stop())
    await asyncio.sleep(0.01)
    with pytest.raises(LifecycleOperationConflictError):
        await lifecycle.stop("planner")
    await stopper


@pytest.mark.asyncio
async def test_health_check_behavior(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    health = await lifecycle.health_check("planner")
    assert health is AgentLifecycleHealth.HEALTHY


@pytest.mark.asyncio
async def test_failed_health_check(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    agent.fail_health = True
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    await lifecycle.start("planner")
    health = await lifecycle.health_check("planner")
    assert health is AgentLifecycleHealth.DEGRADED
    assert lifecycle.get_state("planner") is AgentLifecycleState.DEGRADED


@pytest.mark.asyncio
async def test_agent_manager_compatibility(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")

    discovery = AgentDiscovery(registry, modules=[TEST_MODULE], lifecycle=lifecycle)
    manager = AgentManager()
    handle = await registry.get("planner")
    adapter = ManagerAgentAdapter(handle, capabilities=frozenset({"planning"}))
    await manager.register(adapter)

    class _Task:
        task_id = "t-1"
        task_type = "plan"
        required_capabilities = frozenset({"planning"})
        priority = __import__(
            "kodiak.db.models.task", fromlist=["TaskPriority"]
        ).TaskPriority.MEDIUM
        attempt = 1
        allow_fallback = True
        health_check_required = False

    selected = await manager.select_agent(_Task())
    assert selected.name == "planner"
    assert discovery is not None


@pytest.mark.asyncio
async def test_registry_compatibility(registry: AgentRegistry, lifecycle: AgentLifecycleManager) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    discovered = await lifecycle.sync_from_registry()
    assert "planner" in discovered
    assert await registry.exists("planner")


@pytest.mark.asyncio
async def test_agent_selector_compatibility(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    ready = _LifecycleAgent()
    stopped = _LifecycleAgent()
    await _register_agent(registry, "planner", ready)
    await _register_agent(registry, "coder", stopped)
    await lifecycle.sync_from_registry()
    await lifecycle.initialize("planner")
    await lifecycle.initialize("coder")
    await lifecycle.shutdown("coder")

    selector = AgentSelector(registry, lifecycle=lifecycle)
    selection = await selector.select(required_capabilities=frozenset({"planning"}))
    assert selection is not None
    assert selection.agent_id == "planner"

    blocked = await selector.select(agent_id="coder")
    assert blocked is None


@pytest.mark.asyncio
async def test_agent_discovery_compatibility(registry: AgentRegistry) -> None:
    lifecycle = AgentLifecycleManager(registry)
    discovery = AgentDiscovery(registry, modules=[TEST_MODULE], lifecycle=lifecycle)
    result = await discovery.discover_and_register()
    assert "planner" in result.registered
    assert lifecycle.get_state("planner") is AgentLifecycleState.DISCOVERED


@pytest.mark.asyncio
async def test_execution_engine_compatibility() -> None:
    from kodiak.orchestration.execution import ExecutionEngine

    assert ExecutionEngine is not None


@pytest.mark.asyncio
async def test_shutdown_idempotent(
    lifecycle: AgentLifecycleManager,
    registry: AgentRegistry,
) -> None:
    agent = _LifecycleAgent()
    await _register_agent(registry, "planner", agent)
    await lifecycle.sync_from_registry()
    await lifecycle.shutdown("planner")
    state = await lifecycle.shutdown("planner")
    assert state is AgentLifecycleState.STOPPED
    assert agent.shutdown_calls == 1
