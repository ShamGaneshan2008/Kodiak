"""
kodiak/db/models/project.py

A Project represents a connected repository/workspace that Kodiak operates
on. Tasks belong to a Project; ownership and access control are scoped here.
"""

from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kodiak.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDMixin


class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class Project(UUIDMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "projects"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    repo_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus), nullable=False, default=ProjectStatus.ACTIVE
    )

    tasks: Mapped[list["Task"]] = relationship(  # noqa: F821
        "Task", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r}>"