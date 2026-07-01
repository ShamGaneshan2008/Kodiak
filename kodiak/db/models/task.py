"""
kodiak/db/models/task.py

A top-level unit of work assigned to Kodiak (e.g. "fix GitHub issue #42").
One Task fans out into many AgentRuns.
"""

from __future__ import annotations

import enum

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kodiak.db.base import (
    KodiakBase,
    SoftDeleteMixin,
    UUIDMixin,
    TimestampMixin,
)


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskSource(str, enum.Enum):
    GITHUB_ISSUE = "github_issue"
    GITHUB_PR = "github_pr"
    API = "api"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"


class Task(KodiakBase, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "tasks"

    # Ownership
    repository_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), nullable=True, index=True
    )

    # Description
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    source: Mapped[TaskSource] = mapped_column(
        Enum(TaskSource),
        nullable=False,
        default=TaskSource.API,
    )

    source_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Status
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus),
        nullable=False,
        default=TaskStatus.PENDING,
        index=True,
    )

    priority: Mapped[TaskPriority] = mapped_column(
        Enum(TaskPriority),
        nullable=False,
        default=TaskPriority.MEDIUM,
    )

    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    # GitHub outputs
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pr_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    branch_name: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # State
    plan: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    context: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Cost tracking
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[float | None] = mapped_column(nullable=True)

    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationships
    repository: Mapped["Repository"] = relationship(
        "Repository", back_populates="tasks", lazy="raise"
    )

    agent_runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun",
        back_populates="task",
        lazy="raise",
        cascade="all, delete-orphan",
    )

    # feedbacks: Mapped[list["Feedback"]] = relationship(
    #     "Feedback",
    #     back_populates="task",
    #     lazy="raise"
    # )

    def __repr__(self) -> str:
        return f"<Task id={self.id!r} status={self.status!r} title={self.title[:40]!r}>"