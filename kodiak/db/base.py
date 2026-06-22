"""
Shared SQLAlchemy declarative base and reusable column mixins.
All domain models inherit from ``Base`` and compose mixins as needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Project-wide declarative base. Import this into every model module."""
    pass


# ── Mixins ────────────────────────────────────────────────────────────────────

class UUIDMixin:
    """Primary key as UUID v4."""
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )


class TimestampMixin:
    """Automatic created_at / updated_at columns."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Logical delete — set deleted_at instead of hard-deleting rows."""
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


class SlugMixin:
    """Human-readable URL slug."""
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)