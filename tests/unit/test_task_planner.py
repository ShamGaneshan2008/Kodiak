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


@pytest.mark.asyncio
async def test_goal_decomposition_hierarchical() -> None:
    from kodiak.orchestration.planning import TaskDecomposer

    decomposer = TaskDecomposer()
    planner_plan = {
        "subtasks": [
            {
                "id": "parent-1",
                "title": "Parent task",
                "type": "implementation",
                "depends_on": [],
            },
            {
                "id": "child-1",
                "title": "Child task 1",
                "type": "test",
                "depends_on": ["parent-1"],
                "parent_id": "parent-1",
            },
        ]
    }

    tasks = decomposer.decompose("Build feature", {"plan": planner_plan})
    assert len(tasks) == 2
    by_step = {task.plan_step_id: task for task in tasks}
    assert by_step["child-1"].parent_id == by_step["parent-1"].id
    assert by_step["child-1"].id in by_step["parent-1"].subtask_ids


@pytest.mark.asyncio
async def test_dependency_cycle_detection() -> None:
    import uuid

    from kodiak.orchestration.planning import DependencyCycleError, DependencyGraph
    from kodiak.orchestration.task_planner import ExecutableTask

    t1_id = uuid.uuid4()
    t2_id = uuid.uuid4()

    t1 = ExecutableTask(id=t1_id, name="task1", agent_type="coder", dependencies=[t2_id])
    t2 = ExecutableTask(id=t2_id, name="task2", agent_type="tester", dependencies=[t1_id])

    graph = DependencyGraph([t1, t2])
    with pytest.raises(DependencyCycleError) as exc_info:
        graph.detect_cycles()

    assert str(t1_id) in exc_info.value.cycle_path or str(t2_id) in exc_info.value.cycle_path


@pytest.mark.asyncio
async def test_task_prioritization() -> None:
    from kodiak.orchestration.planning import DependencyGraph, TaskPrioritizer
    from kodiak.orchestration.task_planner import ExecutableTask

    t1 = ExecutableTask(name="t1", agent_type="retrieval", priority="high")
    t2 = ExecutableTask(name="t2", agent_type="coder", dependencies=[t1.id], priority="medium")
    t3 = ExecutableTask(name="t3", agent_type="tester", dependencies=[t2.id], priority="low")

    graph = DependencyGraph([t1, t2, t3])
    prioritizer = TaskPrioritizer()
    prioritized = prioritizer.prioritize([t1, t2, t3], graph)

    # t1 unblocks downstream tasks t2 and t3, so it should get highest priority_score
    t1_score = next(t.priority_score for t in prioritized if t.name == "t1")
    t3_score = next(t.priority_score for t in prioritized if t.name == "t3")
    assert t1_score > t3_score


@pytest.mark.asyncio
async def test_task_estimation() -> None:
    from kodiak.orchestration.planning import TaskEstimator
    from kodiak.orchestration.task_planner import ExecutableTask

    t1 = ExecutableTask(
        name="low_task",
        agent_type="retrieval",
        estimated_complexity="low",
        tool_names=["context_builder"],
        files_to_inspect=["main.py"],
    )
    t2 = ExecutableTask(
        name="high_task",
        agent_type="coder",
        estimated_complexity="high",
        tool_names=["coder"],
        files_to_inspect=["main.py"],
    )

    estimator = TaskEstimator()
    updated, total_comp, total_dur = estimator.estimate([t1, t2])

    assert total_comp == "high"
    assert total_dur == 135.0  # 15s + 120s
    assert updated[0].confidence_score == 1.0


@pytest.mark.asyncio
async def test_plan_validation() -> None:
    import uuid

    from kodiak.orchestration.planning import PlanValidator
    from kodiak.orchestration.task_planner import ExecutableTask

    valid_task = ExecutableTask(name="task1", agent_type="coder")
    invalid_task = ExecutableTask(name="task2", agent_type="tester", dependencies=[uuid.uuid4()])

    validator = PlanValidator()
    result = validator.validate(
        "Test Goal", [valid_task, invalid_task], [valid_task.id, invalid_task.id]
    )

    assert not result.is_valid
    assert any("non-existent" in err for err in result.errors)


@pytest.mark.asyncio
async def test_plan_optimization_and_transitive_reduction() -> None:
    from kodiak.orchestration.planning import DependencyGraph, PlanOptimizer
    from kodiak.orchestration.task_planner import ExecutableTask

    t1 = ExecutableTask(name="t1", agent_type="retrieval")
    t2 = ExecutableTask(name="t2", agent_type="coder", dependencies=[t1.id])
    # t3 depends on t2 AND directly on t1 (t1 -> t3 is redundant since t1 -> t2 -> t3)
    t3 = ExecutableTask(name="t3", agent_type="tester", dependencies=[t1.id, t2.id])

    tasks = [t1, t2, t3]
    graph = DependencyGraph(tasks)
    optimizer = PlanOptimizer()
    optimized_tasks, order, parallel_groups = optimizer.optimize(tasks, graph)

    t3_opt = next(t for t in optimized_tasks if t.name == "t3")
    assert t3_opt.dependencies == [t2.id]  # t1 dependency removed by transitive reduction
    assert len(parallel_groups) == 3


@pytest.mark.asyncio
async def test_dynamic_replanning() -> None:
    from kodiak.orchestration.planning import PlanReplanner
    from kodiak.orchestration.task_planner import ExecutableTask, ExecutionPlan

    t1 = ExecutableTask(name="t1", agent_type="retrieval", status="completed")
    t2 = ExecutableTask(name="t2", agent_type="coder", dependencies=[t1.id], status="failed")

    initial_plan = ExecutionPlan(
        goal="Feature goal",
        tasks=[t1, t2],
        execution_order=[t1.id, t2.id],
        parallel_groups=[[t1.id], [t2.id]],
    )

    replanner = PlanReplanner()
    updated_plan = replanner.replan(
        initial_plan,
        {"task_id": str(t2.id), "error": "Syntax error in file"},
    )

    task_names = [t.name for t in updated_plan.tasks]
    assert "t1" in task_names  # Preserves completed task
    assert "debug_t2" in task_names  # Injects debug recovery task
    assert "repair_t2" in task_names  # Injects repair recovery task
    assert updated_plan.metadata["replanned"] is True


@pytest.mark.asyncio
async def test_plan_serialization() -> None:
    from kodiak.orchestration.planning import PlanSerializer
    from kodiak.orchestration.task_planner import ExecutableTask, ExecutionPlan

    t1 = ExecutableTask(name="t1", agent_type="retrieval")
    t2 = ExecutableTask(name="t2", agent_type="coder", dependencies=[t1.id])

    original_plan = ExecutionPlan(
        goal="Serialize test",
        tasks=[t1, t2],
        execution_order=[t1.id, t2.id],
        parallel_groups=[[t1.id], [t2.id]],
        acceptance_criteria=["Criteria 1"],
    )

    json_str = PlanSerializer.to_json(original_plan)
    deserialized_plan = PlanSerializer.from_json(json_str)

    assert deserialized_plan.goal == original_plan.goal
    assert len(deserialized_plan.tasks) == 2
    assert deserialized_plan.tasks[0].id == t1.id
    assert deserialized_plan.tasks[1].dependencies == [t1.id]
    assert deserialized_plan.acceptance_criteria == ["Criteria 1"]


@pytest.mark.asyncio
async def test_planning_pipeline_e2e() -> None:
    from kodiak.orchestration.planning import PlanningPipeline

    pipeline = PlanningPipeline()
    plan = await pipeline.plan("Implement new authentication flow")

    assert plan.goal == "Implement new authentication flow"
    assert len(plan.tasks) >= 3
    assert plan.validation_result["is_valid"] is True
    assert plan.estimated_total_duration > 0.0
