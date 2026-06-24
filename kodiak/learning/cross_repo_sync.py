from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SyncStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConflictResolution(StrEnum):
    HIGHEST_VERSION = "highest_version"
    HIGHEST_SUCCESS_RATE = "highest_success_rate"
    MERGE = "merge"


class DuplicateGroup(BaseModel):
    pattern_ids: list[uuid.UUID]
    similarity_score: float = Field(ge=0.0, le=1.0)
    resolution: ConflictResolution = ConflictResolution.HIGHEST_SUCCESS_RATE


class SyncResult(BaseModel):
    repository: str
    status: SyncStatus
    imported: int = 0
    duplicates: int = 0
    merged: int = 0
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SyncReport(BaseModel):
    results: list[SyncResult] = Field(default_factory=list)
    total_imported: int = 0
    total_merged: int = 0


class SyncStatistics(BaseModel):
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    total_patterns_synced: int = 0


@runtime_checkable
class Pattern(Protocol):
    id: uuid.UUID
    name: str
    description: str
    tags: list[str]
    success_rate: float
    usage_count: int
    version: int
    status: str
    metadata: dict[str, Any]


@runtime_checkable
class PatternStore(Protocol):
    async def list_patterns_by_repo(self, repo: str) -> list[Pattern]: ...
    async def find_similar(self, name: str, tags: list[str], limit: int) -> list[Pattern]: ...
    async def get_pattern(self, pattern_id: uuid.UUID) -> Pattern | None: ...
    async def update_pattern(self, pattern_id: uuid.UUID, data: dict[str, Any]) -> Pattern: ...
    async def create_pattern(self, data: dict[str, Any]) -> Pattern: ...
    async def deprecate_pattern(self, pattern_id: uuid.UUID) -> bool: ...


class CrossRepoSyncService:
    def __init__(self, store: PatternStore) -> None:
        self._store = store
        self._history: list[SyncResult] = []

    async def sync_repository(self, repo: str) -> SyncResult:
        result = SyncResult(repository=repo, status=SyncStatus.RUNNING)
        try:
            patterns = await self._store.list_patterns_by_repo(repo)
            for pattern in patterns:
                dup_group = await self.detect_duplicates(pattern)
                if dup_group.pattern_ids:
                    result.duplicates += 1
                    await self.resolve_conflicts(dup_group)
                else:
                    await self._store.update_pattern(
                        pattern.id,
                        {"metadata": {**pattern.metadata, "synced": True}},
                    )
                    result.imported += 1
            result.status = SyncStatus.COMPLETED
        except Exception as e:
            logger.exception("sync_failed", repo=repo)
            result.status = SyncStatus.FAILED
            result.error = str(e)
        self._history.append(result)
        return result

    async def sync_all_repositories(self, repos: list[str]) -> SyncReport:
        report = SyncReport()
        for repo in repos:
            res = await self.sync_repository(repo)
            report.results.append(res)
            report.total_imported += res.imported
            report.total_merged += res.merged
        return report

    async def detect_duplicates(
        self, pattern: Pattern, threshold: float = 0.9
    ) -> DuplicateGroup:
        similar = await self._store.find_similar(pattern.name, pattern.tags, limit=5)
        ids = [p.id for p in similar if p.id != pattern.id]
        return DuplicateGroup(
            pattern_ids=ids,
            similarity_score=threshold if ids else 0.0,
        )

    async def resolve_conflicts(self, group: DuplicateGroup) -> uuid.UUID | None:
        if not group.pattern_ids:
            return None
        if group.resolution == ConflictResolution.MERGE:
            return await self.merge_patterns(group.pattern_ids)

        patterns: list[Pattern] = []
        for pid in group.pattern_ids:
            p = await self._store.get_pattern(pid)
            if p is not None:
                patterns.append(p)
        if not patterns:
            return None

        if group.resolution == ConflictResolution.HIGHEST_SUCCESS_RATE:
            winner = max(patterns, key=lambda p: p.success_rate)
        else:
            winner = max(patterns, key=lambda p: p.version)

        for p in patterns:
            if p.id != winner.id:
                await self.deprecate_pattern(p.id)
        return winner.id

    async def merge_patterns(self, pattern_ids: list[uuid.UUID]) -> uuid.UUID | None:
        patterns: list[Pattern] = []
        for pid in pattern_ids:
            p = await self._store.get_pattern(pid)
            if p is not None:
                patterns.append(p)
        if not patterns:
            return None

        best = max(patterns, key=lambda p: p.success_rate)
        all_tags: set[str] = set()
        for p in patterns:
            all_tags.update(p.tags)

        merged_data = {
            "name": best.name,
            "description": best.description,
            "tags": list(all_tags),
            "usage_count": sum(p.usage_count for p in patterns),
            "version": max(p.version for p in patterns) + 1,
            "metadata": {"merged_from": [str(pid) for pid in pattern_ids]},
        }

        new_pattern = await self._store.create_pattern(merged_data)
        for p in patterns:
            await self.deprecate_pattern(p.id)
        return new_pattern.id

    async def promote_pattern(self, pattern_id: uuid.UUID) -> bool:
        try:
            await self._store.update_pattern(pattern_id, {"status": "active"})
            return True
        except Exception:
            logger.exception("promote_failed", pattern_id=pattern_id)
            return False

    async def deprecate_pattern(self, pattern_id: uuid.UUID) -> bool:
        return await self._store.deprecate_pattern(pattern_id)

    def generate_sync_report(self) -> SyncReport:
        report = SyncReport(results=list(self._history))
        report.total_imported = sum(r.imported for r in self._history)
        report.total_merged = sum(r.merged for r in self._history)
        return report

    def get_sync_statistics(self) -> SyncStatistics:
        total = len(self._history)
        return SyncStatistics(
            total_syncs=total,
            successful_syncs=sum(
                1 for r in self._history if r.status == SyncStatus.COMPLETED
            ),
            failed_syncs=sum(
                1 for r in self._history if r.status == SyncStatus.FAILED
            ),
            total_patterns_synced=sum(r.imported for r in self._history),
        )