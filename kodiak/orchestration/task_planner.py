import uuid
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ExecutableTask(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    agent_type: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[uuid.UUID] = Field(default_factory=list)


class TaskPlanner:
    async def plan(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> list[ExecutableTask]:
        ctx = context or {}

        if "implement" in goal.lower() or "create" in goal.lower():
            return self._plan_implementation(goal, ctx)
        if "fix" in goal.lower() or "debug" in goal.lower():
            return self._plan_debugging(goal, ctx)
        if "review" in goal.lower():
            return self._plan_review(goal, ctx)

        return self._plan_generic(goal, ctx)

    def _plan_implementation(
        self, goal: str, ctx: dict[str, Any]
    ) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            id=uuid.uuid4(), name="retrieve_context", agent_type="retrieval", input_data={"task": goal}
        )
        plan = ExecutableTask(
            id=uuid.uuid4(), name="create_plan", agent_type="planner", input_data={"task": goal}, dependencies=[retrieval.id]
        )
        code = ExecutableTask(
            id=uuid.uuid4(), name="write_code", agent_type="coder", input_data={"task": goal}, dependencies=[plan.id]
        )
        test = ExecutableTask(
            id=uuid.uuid4(), name="write_tests", agent_type="tester", input_data={"task": goal}, dependencies=[code.id]
        )
        review = ExecutableTask(
            id=uuid.uuid4(), name="review_code", agent_type="reviewer", input_data={"task": goal}, dependencies=[code.id]
        )
        logger.info("implementation_plan_created", tasks=5)
        return [retrieval, plan, code, test, review]

    def _plan_debugging(
        self, goal: str, ctx: dict[str, Any]
    ) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            id=uuid.uuid4(), name="retrieve_context", agent_type="retrieval", input_data={"task": goal}
        )
        debug = ExecutableTask(
            id=uuid.uuid4(), name="analyze_error", agent_type="debugger", input_data={"task": goal}, dependencies=[retrieval.id]
        )
        fix = ExecutableTask(
            id=uuid.uuid4(), name="apply_fix", agent_type="coder", input_data={"task": goal}, dependencies=[debug.id]
        )
        test = ExecutableTask(
            id=uuid.uuid4(), name="verify_fix", agent_type="tester", input_data={"task": goal}, dependencies=[fix.id]
        )
        logger.info("debugging_plan_created", tasks=4)
        return [retrieval, debug, fix, test]

    def _plan_review(
        self, goal: str, ctx: dict[str, Any]
    ) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            id=uuid.uuid4(), name="get_code", agent_type="retrieval", input_data={"task": goal}
        )
        review = ExecutableTask(
            id=uuid.uuid4(), name="execute_review", agent_type="reviewer", input_data={"task": goal}, dependencies=[retrieval.id]
        )
        logger.info("review_plan_created", tasks=2)
        return [retrieval, review]

    def _plan_generic(
        self, goal: str, ctx: dict[str, Any]
    ) -> list[ExecutableTask]:
        retrieval = ExecutableTask(
            id=uuid.uuid4(), name="retrieve_context", agent_type="retrieval", input_data={"task": goal}
        )
        plan = ExecutableTask(
            id=uuid.uuid4(), name="create_plan", agent_type="planner", input_data={"task": goal}, dependencies=[retrieval.id]
        )
        exec_task = ExecutableTask(
            id=uuid.uuid4(), name="execute_task", agent_type="coder", input_data={"task": goal}, dependencies=[plan.id]
        )
        logger.info("generic_plan_created", tasks=3)
        return [retrieval, plan, exec_task]