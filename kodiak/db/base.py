"""
Shared SQLAlchemy base and reusable mixins for all database models.

Every model in the project should inherit from KodiakBase and reuse
these mixins instead of redefining common fields like id, timestamps, etc.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ─────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────

class KodiakBase(DeclarativeBase):
    """Base class for all ORM models in Kodiak."""
    pass


# IMPORTANT: many files expect this name
Base = KodiakBase


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────
# Mixins
# ─────────────────────────────────────────────

class UUIDMixin:
    """Adds a UUID primary key column."""
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )


class TimestampMixin:
    """Adds created_at and updated_at timestamps."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Soft delete support using deleted_at field."""
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


class SlugMixin:
    """
    Adds a human-readable unique identifier (slug).

    Useful for URLs like:
    /projects/my-project-name
    """
    slug: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )