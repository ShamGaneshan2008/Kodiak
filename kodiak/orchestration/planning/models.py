# kodiak/orchestration/planning/models.py
"""Domain models and schemas for Kodiak's advanced planning system."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from kodiak.db.models.task import TaskPriority

__all__ = [
    "TaskComplexity",
    "TaskDependencyType",
    "FailureRecoveryStrategy",
    "ResourceEstimate",
    "PlanValidationIssue",
    "PlanValidationResult",
    "PlanOptimizationConfig",
    "HierarchicalTaskNode",
    "HierarchicalPlan",
    "ReplanContext",
    "PlanDiff",
]


class TaskComplexity(StrEnum):
    """Estimated LLM & computational complexity for a planning task."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskDependencyType(StrEnum):
    """Semantics of a task dependency link."""

    HARD = "hard"  # Task cannot start until dependency completes successfully
    SOFT = "soft"  # Task prefers dependency output, but can proceed if dependency skipped
    CONDITIONAL = "conditional"  # Task runs only if dependency meets specific condition
    RECOVERY = "recovery"  # Task runs only if dependency fails


class FailureRecoveryStrategy(StrEnum):
    """Strategy for recovering from a failed task attempt."""

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SWAP_AGENT = "swap_agent"
    DEBUGGER_ANALYSIS = "debugger_analysis"
    REPLAN_SUBGRAPH = "replan_subgraph"
    SKIP_OPTIONAL = "skip_optional"


class ResourceEstimate(BaseModel):
    """Resource and cost estimations for a single task or full plan."""

    estimated_input_tokens: int = Field(default=500, ge=0)
    estimated_output_tokens: int = Field(default=500, ge=0)
    estimated_cost_usd: float = Field(default=0.01, ge=0.0)
    estimated_duration_seconds: float = Field(default=30.0, ge=0.0)
    required_tools: list[str] = Field(default_factory=list)
    sandbox_required: bool = False
    recommended_model: str = "claude-sonnet"

    @property
    def total_tokens(self) -> int:
        """Calculate total estimated token spend."""
        return self.estimated_input_tokens + self.estimated_output_tokens


class PlanValidationIssue(BaseModel):
    """Individual issue found during plan validation."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    task_id: str | None = None


class PlanValidationResult(BaseModel):
    """Complete validation report for a plan."""

    is_valid: bool
    issues: list[PlanValidationIssue] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def errors(self) -> list[PlanValidationIssue]:
        """Filter validation issues for errors only."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[PlanValidationIssue]:
        """Filter validation issues for warnings only."""
        return [issue for issue in self.issues if issue.severity == "warning"]


class PlanOptimizationConfig(BaseModel):
    """Configuration options for plan optimization passes."""

    merge_inspection_tasks: bool = True
    maximize_parallelism: bool = True
    shorten_critical_path: bool = True
    prune_redundant_files: bool = True
    max_parallel_width: int = 5


class HierarchicalTaskNode(BaseModel):
    """Single node in a hierarchical task decomposition tree."""

    id: str = Field(default_factory=lambda: f"st-{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    # inspection, research, implementation, test, documentation, review, debugging
    task_type: str = "implementation"
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    priority: TaskPriority = TaskPriority.MEDIUM
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    files_to_inspect: list[str] = Field(default_factory=list)
    likely_files: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    parallel_group: str | None = None
    can_run_parallel: bool = False
    resource_estimate: ResourceEstimate = Field(default_factory=ResourceEstimate)
    recovery_strategy: FailureRecoveryStrategy = FailureRecoveryStrategy.RETRY_WITH_BACKOFF
    metadata: dict[str, Any] = Field(default_factory=dict)


class HierarchicalPlan(BaseModel):
    """Complete hierarchical plan structure containing task tree and metadata."""

    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    goal: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    root_task_ids: list[str] = Field(default_factory=list)
    tasks: dict[str, HierarchicalTaskNode] = Field(default_factory=dict)
    execution_order: list[str] = Field(default_factory=list)
    parallel_groups: list[list[str]] = Field(default_factory=list)
    total_resource_estimate: ResourceEstimate = Field(default_factory=ResourceEstimate)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReplanContext(BaseModel):
    """Contextual information provided during dynamic replanning."""

    original_plan: dict[str, Any]
    completed_step_ids: list[str] = Field(default_factory=list)
    failed_step_id: str | None = None
    failure_error: str | None = None
    execution_logs: list[str] = Field(default_factory=list)
    memory_context: str | None = None


class PlanDiff(BaseModel):
    """Summary of modifications made during replanning or optimization."""

    added_task_ids: list[str] = Field(default_factory=list)
    removed_task_ids: list[str] = Field(default_factory=list)
    modified_task_ids: list[str] = Field(default_factory=list)
    reason: str = ""
