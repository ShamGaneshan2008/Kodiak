"""
GitHub integration models: Installation, Repository.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kodiak.db.base import Base, TimestampMixin, UUIDMixin


class GitHubInstallation(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "github_installations"

    installation_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, nullable=False, index=True
    )
    app_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    account_type: Mapped[str] = mapped_column(String(16), nullable=False)  # "User" | "Organization"
    account_login: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    account_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Owner user (nullable if the installer hasn't signed up yet)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    permissions: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    events: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)

    owner: Mapped[User] = relationship("User", back_populates="installations")  # noqa: F821
    repositories: Mapped[list[Repository]] = relationship(
        "Repository", back_populates="installation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<GitHubInstallation id={self.installation_id} account={self.account_login!r}>"


class Repository(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "repositories"

    installation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("github_installations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    github_repo_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)  # "owner/repo"
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main", nullable=False)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_private: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Kodiak configuration stored per-repo (mirrors .kodiak.yml)
    config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # RAG index metadata
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    index_commit_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)

    installation: Mapped[GitHubInstallation] = relationship(
        "GitHubInstallation", back_populates="repositories"
    )
    tasks: Mapped[list[Task]] = relationship(  # noqa: F821
        "Task", back_populates="repository"
    )

    def __repr__(self) -> str:
        return f"<Repository full_name={self.full_name!r}>"
