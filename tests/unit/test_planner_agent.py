from __future__ import annotations

import json

import pytest

from kodiak.agents.base import AgentInput
from kodiak.agents.planner import PlannerAgent


class FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def complete(self, **_: object) -> dict:
        return {
            "content": json.dumps(self.payload),
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }


@pytest.mark.asyncio
async def test_planner_agent_emits_machine_readable_execution_plan() -> None:
    payload = {
        "goal": "Add a planner agent.",
        "acceptance_criteria": ["Plan is structured before code changes."],
        "subtasks": [
            {
                "id": "st-1",
                "title": "Inspect orchestration",
                "description": "Find integration points.",
                "type": "inspection",
                "complexity": "low",
                "depends_on": [],
                "files_to_inspect": ["kodiak/orchestration/task_planner.py"],
                "tools": [
                    {
                        "name": "context_builder",
                        "purpose": "Collect orchestration context.",
                        "required_capability": "repository_context",
                        "inputs": {"query": "planner integration"},
                        "risk_level": "low",
                    }
                ],
                "parallel_group": "pg-1",
                "can_run_parallel": False,
                "likely_files": [],
            },
            {
                "id": "st-2",
                "title": "Implement planner",
                "description": "Extend planner schema.",
                "type": "implementation",
                "complexity": "medium",
                "depends_on": ["st-1"],
                "files_to_inspect": ["kodiak/agents/planner.py"],
                "tools": [
                    {
                        "name": "coder",
                        "purpose": "Generate code changes.",
                        "required_capability": "code_generation",
                        "inputs": {"target_files": ["kodiak/agents/planner.py"]},
                        "risk_level": "medium",
                    }
                ],
                "parallel_group": "pg-2",
                "can_run_parallel": False,
                "likely_files": ["kodiak/agents/planner.py"],
            },
        ],
        "estimated_total_complexity": "medium",
        "requires_architecture_review": True,
    }

    agent = PlannerAgent(FakeLLM(payload))
    output = await agent.run(
        AgentInput(
            task_id="task-1",
            project_id="project-1",
            instruction="Implement a planner agent.",
        )
    )

    assert output.success
    plan = output.result["plan"]
    assert plan["plan_version"] == "1.0"
    assert plan["repository_files"] == [
        "kodiak/agents/planner.py",
        "kodiak/orchestration/task_planner.py",
    ]
    assert plan["execution_order"] == ["st-1", "st-2"]
    assert plan["parallel_groups"] == [["st-1"], ["st-2"]]
    assert plan["estimated_dependencies"] == [
        {
            "from_task": "st-1",
            "to_task": "st-2",
            "reason": "Declared task dependency.",
        }
    ]
    assert plan["subtasks"][0]["tools"][0]["name"] == "context_builder"


@pytest.mark.asyncio
async def test_planner_agent_backfills_tools_and_inspection_files() -> None:
    payload = {
        "goal": "Update docs.",
        "acceptance_criteria": [],
        "subtasks": [
            {
                "id": "st-1",
                "title": "Change docs",
                "description": "Update docs.",
                "type": "documentation",
                "complexity": "low",
                "depends_on": [],
                "likely_files": ["docs/usage.md"],
            }
        ],
        "estimated_total_complexity": "low",
        "requires_architecture_review": False,
    }

    agent = PlannerAgent(FakeLLM(payload))
    output = await agent.run(
        AgentInput(
            task_id="task-2",
            project_id="project-1",
            instruction="Update docs.",
        )
    )

    task = output.result["plan"]["subtasks"][0]
    assert task["files_to_inspect"] == ["docs/usage.md"]
    assert task["tools"][0]["name"] == "coder"
