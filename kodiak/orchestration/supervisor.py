import uuid
from typing import Any, Protocol, runtime_checkable

import structlog

from kodiak.orchestration.state import AgentState, AgentStatus, ExecutionState, TaskStatus

logger = structlog.get_logger(__name__)


@runtime_checkable
class ExecutableAgent(Protocol):
    name: str

    async def execute(self, input_data: Any) -> Any: ...

    async def stop(self) -> None: ...


class Supervisor:
    def __init__(
        self, state: ExecutionState, agents: dict[str, ExecutableAgent]
    ) -> None:
        self._state = state
        self._agents = agents
        self._running = False

    async def start(self) -> None:
        self._running = True
        for name in self._agents:
            self._state.agents[name] = AgentState(name=name, status=AgentStatus.IDLE)
        logger.info("supervisor_started", agents=list(self._agents.keys()))

    async def stop(self) -> None:
        self._running = False
        for agent in self._agents.values():
            await agent.stop()
        logger.info("supervisor_stopped")

    async def run_task(
        self, task_id: uuid.UUID, agent_type: str, input_data: Any
    ) -> Any:
        agent = self._agents.get(agent_type)
        if not agent:
            logger.error("agent_not_found", agent_type=agent_type)
            return None

        agent_state = self._state.agents[agent_type]
        self._state.agents[agent_type] = AgentState(
            name=agent_type,
            status=AgentStatus.WORKING,
            current_task_id=task_id,
            tasks_completed=agent_state.tasks_completed,
            tasks_failed=agent_state.tasks_failed,
        )

        try:
            result = await agent.execute(input_data)
            self._state.agents[agent_type] = AgentState(
                name=agent_type,
                status=AgentStatus.IDLE,
                tasks_completed=agent_state.tasks_completed + 1,
                tasks_failed=agent_state.tasks_failed,
            )
            return result
        except Exception:
            self._state.agents[agent_type] = AgentState(
                name=agent_type,
                status=AgentStatus.ERROR,
                tasks_completed=agent_state.tasks_completed,
                tasks_failed=agent_state.tasks_failed + 1,
            )
            logger.exception("task_execution_error", task_id=str(task_id))
            raise

    async def restart_agent(self, agent_type: str) -> bool:
        agent = self._agents.get(agent_type)
        if not agent:
            return False
        await agent.stop()
        self._state.agents[agent_type] = AgentState(
            name=agent_type, status=AgentStatus.IDLE
        )
        logger.info("agent_restarted", agent_type=agent_type)
        return True

    def get_statistics(self) -> dict[str, Any]:
        total_tasks = len(self._state.tasks)
        completed = sum(
            1 for t in self._state.tasks if t.status == TaskStatus.COMPLETED
        )
        failed = sum(1 for t in self._state.tasks if t.status == TaskStatus.FAILED)
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "agents": {
                name: state.status for name, state in self._state.agents.items()
            },
        }