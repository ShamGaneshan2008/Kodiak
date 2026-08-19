# kodiak/memory/procedural.py
"""Procedural Memory component for managing reusable step-by-step workflows and procedures."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from .errors import ProcedureNotFoundError
from .models import Memory, MemoryType

logger = structlog.get_logger(__name__)

__all__ = [
    "ProcedureNotFoundError",
    "ProcedureStep",
    "Procedure",
    "ProcedureSearchResult",
    "ProcedureRepository",
    "ProceduralMemory",
]


class ProcedureStep(BaseModel):
    """Single step definition in a procedure."""

    step_number: int = Field(ge=1)
    action: str
    tool_name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None


class Procedure(BaseModel):
    """Procedural workflow model with execution success statistics."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    description: str
    steps: list[ProcedureStep]
    tags: list[str] = Field(default_factory=list)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def success_rate(self) -> float:
        """Calculate procedure execution success rate ratio."""
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.0


class ProcedureSearchResult(BaseModel):
    """Procedure search match with relevance score."""

    procedure: Procedure
    relevance_score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class ProcedureRepository(Protocol):
    """Protocol for procedural memory repository implementations."""

    async def create(self, procedure: Procedure) -> Procedure: ...

    async def get_by_id(self, procedure_id: uuid.UUID) -> Procedure | None: ...

    async def update(self, procedure: Procedure) -> Procedure: ...

    async def search(self, query: str, limit: int = 10) -> list[ProcedureSearchResult]: ...

    async def increment_success(self, procedure_id: uuid.UUID) -> Procedure: ...

    async def increment_failure(self, procedure_id: uuid.UUID) -> Procedure: ...

    async def delete(self, procedure_id: uuid.UUID) -> bool: ...

    async def list_procedures(self, limit: int = 100, offset: int = 0) -> list[Procedure]: ...


class ProceduralMemory:
    """Manager for procedural memory workflows and tool operation sequences."""

    def __init__(self, repository: ProcedureRepository | None = None) -> None:
        """Initialize procedural memory manager.

        Args:
            repository: Underlying procedure repository. Defaults to InMemoryProcedureRepository.
        """
        if repository is None:
            from .persistence import InMemoryProcedureRepository

            repository = InMemoryProcedureRepository()
        self._repo = repository

    async def create_procedure(
        self,
        name: str,
        description: str,
        steps: list[ProcedureStep],
        tags: list[str] | None = None,
    ) -> Procedure:
        """Create and store a procedural workflow.

        Args:
            name: Procedure name label.
            description: Procedure intent description.
            steps: List of ProcedureStep items.
            tags: Category or tool tags.

        Returns:
            Created Procedure instance.
        """
        procedure = Procedure(
            name=name,
            description=description,
            steps=steps,
            tags=tags or [],
        )
        created = await self._repo.create(procedure)
        logger.info(
            "procedure_created",
            procedure_id=str(created.id),
            name=name,
            step_count=len(steps),
        )
        return created

    async def get_procedure(self, procedure_id: uuid.UUID) -> Procedure:
        """Fetch procedure by UUID.

        Args:
            procedure_id: Procedure UUID.

        Returns:
            Procedure model.

        Raises:
            ProcedureNotFoundError: If procedure is not found.
        """
        procedure = await self._repo.get_by_id(procedure_id)
        if procedure is None:
            raise ProcedureNotFoundError(str(procedure_id))
        return procedure

    async def update_procedure(
        self,
        procedure_id: uuid.UUID,
        name: str | None = None,
        description: str | None = None,
        steps: list[ProcedureStep] | None = None,
        tags: list[str] | None = None,
    ) -> Procedure:
        """Update fields of an existing procedure.

        Args:
            procedure_id: Procedure UUID.
            name: Optional new name.
            description: Optional new description.
            steps: Optional new step list.
            tags: Optional new tag list.

        Returns:
            Updated Procedure instance.
        """
        existing = await self.get_procedure(procedure_id)
        update_data = existing.model_dump(
            exclude={"id", "success_count", "failure_count", "created_at"}
        )
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if steps is not None:
            update_data["steps"] = steps
        if tags is not None:
            update_data["tags"] = tags

        update_data["updated_at"] = datetime.now(UTC)
        updated_procedure = Procedure(**update_data)
        result = await self._repo.update(updated_procedure)
        logger.info("procedure_updated", procedure_id=str(procedure_id))
        return result

    async def search_procedures(self, query: str, limit: int = 10) -> list[ProcedureSearchResult]:
        """Search procedures matching a query string.

        Args:
            query: Natural language query.
            limit: Maximum items to return.

        Returns:
            List of ProcedureSearchResult matches.
        """
        return await self._repo.search(query, limit)

    async def record_success(self, procedure_id: uuid.UUID) -> Procedure:
        """Record a successful execution for a procedure.

        Args:
            procedure_id: Procedure UUID.

        Returns:
            Updated Procedure model.
        """
        result = await self._repo.increment_success(procedure_id)
        logger.info(
            "procedure_success_recorded",
            procedure_id=str(procedure_id),
            success_count=result.success_count,
            success_rate=result.success_rate,
        )
        return result

    async def record_failure(self, procedure_id: uuid.UUID) -> Procedure:
        """Record a failed execution for a procedure.

        Args:
            procedure_id: Procedure UUID.

        Returns:
            Updated Procedure model.
        """
        result = await self._repo.increment_failure(procedure_id)
        logger.info(
            "procedure_failure_recorded",
            procedure_id=str(procedure_id),
            failure_count=result.failure_count,
            success_rate=result.success_rate,
        )
        return result

    async def delete_procedure(self, procedure_id: uuid.UUID) -> bool:
        """Delete a procedure by ID.

        Args:
            procedure_id: Procedure UUID.

        Returns:
            True if deleted, else False.
        """
        deleted = await self._repo.delete(procedure_id)
        if deleted:
            logger.info("procedure_deleted", procedure_id=str(procedure_id))
        return deleted

    async def list_procedures(self, limit: int = 100, offset: int = 0) -> list[Procedure]:
        """List stored procedures.

        Args:
            limit: Maximum records to return.
            offset: Offset index for pagination.

        Returns:
            List of Procedure instances.
        """
        return await self._repo.list_procedures(limit=limit, offset=offset)

    @staticmethod
    def to_memory(procedure: Procedure) -> Memory:
        """Convert Procedure model to normalized Memory representation.

        Args:
            procedure: Procedure model instance.

        Returns:
            Normalized Memory instance.
        """
        return Memory(
            id=procedure.id,
            type=MemoryType.PROCEDURAL,
            title=procedure.name,
            content=procedure.description,
            tags=list(procedure.tags),
            metadata={
                "steps": [s.model_dump(mode="json") for s in procedure.steps],
                "success_count": procedure.success_count,
                "failure_count": procedure.failure_count,
                "success_rate": procedure.success_rate,
            },
            confidence=procedure.success_rate,
            created_at=procedure.created_at,
            updated_at=procedure.updated_at,
        )
