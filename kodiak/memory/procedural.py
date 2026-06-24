from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ProcedureNotFoundError(Exception):
    pass


class ProcedureStep(BaseModel):
    step_number: int = Field(ge=1)
    action: str
    tool_name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str | None = None


class Procedure(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    description: str
    steps: list[ProcedureStep]
    tags: list[str] = Field(default_factory=list)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.0


class ProcedureSearchResult(BaseModel):
    procedure: Procedure
    relevance_score: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class ProcedureRepository(Protocol):
    async def create(self, procedure: Procedure) -> Procedure: ...

    async def get_by_id(self, procedure_id: uuid.UUID) -> Procedure | None: ...

    async def update(self, procedure: Procedure) -> Procedure: ...

    async def search(
        self, query: str, limit: int = 10
    ) -> list[ProcedureSearchResult]: ...

    async def increment_success(self, procedure_id: uuid.UUID) -> Procedure: ...

    async def increment_failure(self, procedure_id: uuid.UUID) -> Procedure: ...


class ProceduralMemory:
    def __init__(self, repository: ProcedureRepository) -> None:
        self._repo = repository

    async def create_procedure(
        self,
        name: str,
        description: str,
        steps: list[ProcedureStep],
        tags: list[str] | None = None,
    ) -> Procedure:
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
        procedure = await self._repo.get_by_id(procedure_id)
        if procedure is None:
            raise ProcedureNotFoundError(f"Procedure {procedure_id} not found")
        return procedure

    async def update_procedure(
        self,
        procedure_id: uuid.UUID,
        name: str | None = None,
        description: str | None = None,
        steps: list[ProcedureStep] | None = None,
        tags: list[str] | None = None,
    ) -> Procedure:
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

        update_data["updated_at"] = datetime.now(timezone.utc)
        updated_procedure = Procedure(**update_data)
        result = await self._repo.update(updated_procedure)
        logger.info("procedure_updated", procedure_id=str(procedure_id))
        return result

    async def search_procedures(
        self, query: str, limit: int = 10
    ) -> list[ProcedureSearchResult]:
        return await self._repo.search(query, limit)

    async def record_success(self, procedure_id: uuid.UUID) -> Procedure:
        result = await self._repo.increment_success(procedure_id)
        logger.info(
            "procedure_success_recorded",
            procedure_id=str(procedure_id),
            success_count=result.success_count,
            success_rate=result.success_rate,
        )
        return result

    async def record_failure(self, procedure_id: uuid.UUID) -> Procedure:
        result = await self._repo.increment_failure(procedure_id)
        logger.info(
            "procedure_failure_recorded",
            procedure_id=str(procedure_id),
            failure_count=result.failure_count,
            success_rate=result.success_rate,
        )
        return result