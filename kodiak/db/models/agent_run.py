"""
kodiak/db/models/agent_run.py

A single agent execution step within a Task.
One Task typically spawns multiple AgentRuns (planner → coder → reviewer …).
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kodiak.db.base import KodiakBase


class AgentType(str, enum.Enum):
    PLANNER = "planner"
    REPOSITORY = "repository"
    RETRIEVAL = "retrieval"
    RESEARCH = "research"
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DEBUGGER = "debugger"
    REFLECTION = "reflection"
    GIT = "git"
    MEMORY = "memory"
    LEARNING = "learning"
    EVALUATION = "evaluation"


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentRun(KodiakBase):
    __tablename__ = "agent_runs"

    # Parent task
    task_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Execution graph: parent run for nested / sequential chains
    parent_run_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Agent identity
    agent_type: Mapped[AgentType] = mapped_column(
        Enum(AgentType), nullable=False, index=True
    )
    agent_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # LLM details
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Execution
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), nullable=False, default=RunStatus.QUEUED, index=True
    )
    step_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Token / cost
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)

    # Payload
    input: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    output: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    tool_calls: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Reflection / scoring
    reflection_score: Mapped[float | None] = mapped_column(nullable=True)
    reflection_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    task: Mapped["Task"] = relationship(  # noqa: F821
        "Task", back_populates="agent_runs", lazy="raise"
    )
    parent_run: Mapped["AgentRun | None"] = relationship(
        "AgentRun", remote_side="AgentRun.id", lazy="raise"
    )

    def __repr__(self) -> str:
        return (
            f"<AgentRun id={self.id!r} agent={self.agent_type!r} "
            f"status={self.status!r}>"
        )