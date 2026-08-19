# kodiak/orchestration/planning/estimator.py
"""Resource and cost estimation component for planning tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from .models import ResourceEstimate, TaskComplexity

logger = structlog.get_logger(__name__)

__all__ = ["ResourceEstimator"]

# Pricing table USD per 1M tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku": (0.25, 1.25),
    "claude-sonnet": (3.0, 15.0),
    "claude-opus-4-5": (15.0, 75.0),
}


class ResourceEstimator:
    """Estimates tokens, costs, durations, and sandbox requirements for tasks and plans."""

    def estimate_task(self, task: Any) -> ResourceEstimate:
        """Calculate ResourceEstimate for a single task object or dictionary.

        Args:
            task: Task model or dictionary.

        Returns:
            ResourceEstimate populated with tokens, cost, duration, model, and sandbox settings.
        """
        complexity_str = str(
            getattr(task, "complexity", "medium")
            if not isinstance(task, dict)
            else task.get("complexity", "medium")
        ).lower()

        task_type = str(
            getattr(task, "type", getattr(task, "task_type", "implementation"))
            if not isinstance(task, dict)
            else task.get("type", task.get("task_type", "implementation"))
        ).lower()

        files = (
            getattr(task, "files_to_inspect", [])
            if not isinstance(task, dict)
            else task.get("files_to_inspect", [])
        )
        file_count = len(files) if isinstance(files, list) else 0

        # Base token estimates
        if complexity_str in (TaskComplexity.LOW.value, "low"):
            input_tokens = 500 + file_count * 200
            output_tokens = 300
            model = "claude-haiku"
            duration = 10.0
        elif complexity_str in (TaskComplexity.HIGH.value, "high"):
            input_tokens = 2500 + file_count * 500
            output_tokens = 1500
            model = "claude-sonnet"
            duration = 45.0
        elif complexity_str in (TaskComplexity.CRITICAL.value, "critical"):
            input_tokens = 6000 + file_count * 1000
            output_tokens = 3000
            model = "claude-opus-4-5"
            duration = 90.0
        else:  # medium
            input_tokens = 1200 + file_count * 300
            output_tokens = 800
            model = "claude-sonnet"
            duration = 25.0

        # Compute cost
        price_in, price_out = _MODEL_PRICING.get(model, (3.0, 15.0))
        cost_usd = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

        sandbox_required = task_type in ("test", "implementation", "coder", "tester")

        # Collect required tools
        tools_list: list[str] = []
        raw_tools = (
            getattr(task, "tools", []) if not isinstance(task, dict) else task.get("tools", [])
        )
        if isinstance(raw_tools, list):
            for t in raw_tools:
                if isinstance(t, str):
                    tools_list.append(t)
                elif isinstance(t, dict) and t.get("name"):
                    tools_list.append(str(t["name"]))

        return ResourceEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=round(cost_usd, 6),
            estimated_duration_seconds=duration,
            required_tools=tools_list,
            sandbox_required=sandbox_required,
            recommended_model=model,
        )

    def estimate_plan(self, tasks: Sequence[Any]) -> ResourceEstimate:
        """Aggregate total ResourceEstimate across all tasks in a plan.

        Args:
            tasks: Sequence of task models or dicts.

        Returns:
            ResourceEstimate representing total aggregated plan requirements.
        """
        total_in = 0
        total_out = 0
        total_cost = 0.0
        max_duration = 0.0
        sandbox_needed = False
        all_tools: set[str] = set()

        for task in tasks:
            est = self.estimate_task(task)
            total_in += est.estimated_input_tokens
            total_out += est.estimated_output_tokens
            total_cost += est.estimated_cost_usd
            max_duration += est.estimated_duration_seconds
            if est.sandbox_required:
                sandbox_needed = True
            all_tools.update(est.required_tools)

        return ResourceEstimate(
            estimated_input_tokens=total_in,
            estimated_output_tokens=total_out,
            estimated_cost_usd=round(total_cost, 4),
            estimated_duration_seconds=max_duration,
            required_tools=sorted(all_tools),
            sandbox_required=sandbox_needed,
            recommended_model="claude-sonnet" if total_cost < 0.50 else "claude-opus-4-5",
        )
