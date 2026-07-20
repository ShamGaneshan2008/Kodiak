from __future__ import annotations

import pytest

from kodiak.orchestration.task_planner import TaskPlanner


@pytest.mark.asyncio
async def test_task_planner_converts_planner_output_to_executable_plan() -> None:
    planner_output = {
        "plan_version": "1.0",
        "goal": "Add planner orchestration.",
        "acceptance_criteria": ["Execution plan is machine readable."],
        "repository_files": ["kodiak/agents/planner.py"],
        "execution_order": ["st-1", "st-2", "st-3"],
        "parallel_groups": [["st-1"], ["st-2", "st-3"]],
        "estimated_dependencies": [
            {"from_task": "st-1", "to_task": "st-2", "reason": "Context first."},
            {"from_task": "st-1", "to_task": "st-3", "reason": "Context first."},
        ],
        "subtasks": [
            {
                "id": "st-1",
                "title": "Inspect planner",
                "description": "Read planner code.",
                "type": "inspection",
                "complexity": "low",
                "depends_on": [],
                "files_to_inspect": ["kodiak/agents/planner.py"],
                "tools": [{"name": "context_builder"}],
                "parallel_group": "pg-1",
                "can_run_parallel": False,
                "likely_files": [],
            },
            {
                "id": "st-2",
                "title": "Implement schema",
                "description": "Add planner fields.",
                "type": "implementation",
                "complexity": "medium",
                "depends_on": ["st-1"],
                "files_to_inspect": ["kodiak/agents/planner.py"],
                "tools": [{"name": "coder"}],
                "parallel_group": "pg-2",
                "can_run_parallel": True,
                "likely_files": ["kodiak/agents/planner.py"],
            },
            {
                "id": "st-3",
                "title": "Review schema",
                "description": "Review the plan contract.",
                "type": "review",
                "complexity": "low",
                "depends_on": ["st-1"],
                "files_to_inspect": ["kodiak/agents/planner.py"],
                "tools": [{"name": "reviewer"}],
                "parallel_group": "pg-2",
                "can_run_parallel": True,
                "likely_files": [],
            },
        ],
        "estimated_total_complexity": "medium",
        "requires_architecture_review": False,
    }

    plan = await TaskPlanner().plan_execution(
        "Add planner orchestration.",
        {"planner_output": {"plan": planner_output}, "work_dir": "D:/Kodiak"},
    )

    by_step = {task.plan_step_id: task for task in plan.tasks}
    assert by_step["st-1"].agent_type == "research"
    assert by_step["st-2"].agent_type == "coder"
    assert by_step["st-3"].agent_type == "reviewer"
    assert by_step["st-2"].dependencies == [by_step["st-1"].id]
    assert by_step["st-2"].tool_names == ["coder"]
    assert by_step["st-2"].files_to_inspect == ["kodiak/agents/planner.py"]
    assert plan.parallel_groups == [
        [by_step["st-1"].id],
        [by_step["st-2"].id, by_step["st-3"].id],
    ]
    assert plan.metadata["source"] == "planner_agent"
    assert plan.machine_readable()["tasks"][1]["plan_step_id"] == "st-2"
