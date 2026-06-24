from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TraceStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceSpan(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: uuid.UUID | None = None
    trace_id: uuid.UUID
    operation_name: str
    status: TraceStatus = TraceStatus.STARTED
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: uuid.UUID
    spans: list[TraceSpan]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceManager:
    def __init__(self) -> None:
        self._traces: dict[uuid.UUID, dict[uuid.UUID, TraceSpan]] = {}
        self._lock = asyncio.Lock()

    async def start_trace(self) -> uuid.UUID:
        trace_id = uuid.uuid4()
        async with self._lock:
            self._traces[trace_id] = {}
        logger.debug("trace_started", trace_id=str(trace_id))
        return trace_id

    async def start_span(
        self,
        trace_id: uuid.UUID,
        operation_name: str,
        parent_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan | None:
        span = TraceSpan(
            trace_id=trace_id,
            operation_name=operation_name,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        async with self._lock:
            trace_spans = self._traces.get(trace_id)
            if trace_spans is None:
                logger.warning("span_started_for_missing_trace", trace_id=str(trace_id))
                return None
            trace_spans[span.id] = span
        logger.debug(
            "span_started",
            trace_id=str(trace_id),
            span_id=str(span.id),
            operation=operation_name,
        )
        return span

    async def end_span(
        self,
        trace_id: uuid.UUID,
        span_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan | None:
        return await self._finish_span(
            trace_id, span_id, TraceStatus.COMPLETED, metadata
        )

    async def fail_span(
        self,
        trace_id: uuid.UUID,
        span_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan | None:
        return await self._finish_span(
            trace_id, span_id, TraceStatus.FAILED, metadata
        )

    async def _finish_span(
        self,
        trace_id: uuid.UUID,
        span_id: uuid.UUID,
        status: TraceStatus,
        metadata: dict[str, Any] | None,
    ) -> TraceSpan | None:
        async with self._lock:
            trace_spans = self._traces.get(trace_id)
            if trace_spans is None:
                return None
            span = trace_spans.get(span_id)
            if span is None:
                return None

            now = datetime.now(timezone.utc)
            duration = (now - span.start_time).total_seconds() * 1000.0

            span.status = status
            span.end_time = now
            span.duration_ms = duration
            if metadata:
                span.metadata.update(metadata)

            updated_span = span.model_copy(deep=True)
            trace_spans[span_id] = updated_span

        logger.debug(
            "span_finished",
            trace_id=str(trace_id),
            span_id=str(span_id),
            status=status.value,
            duration_ms=duration,
        )
        return updated_span

    async def get_trace(self, trace_id: uuid.UUID) -> TraceRecord | None:
        async with self._lock:
            trace_spans = self._traces.get(trace_id)
            if trace_spans is None:
                return None
            spans = list(trace_spans.values())
        return TraceRecord(trace_id=trace_id, spans=spans)

    async def export_trace(self, trace_id: uuid.UUID) -> TraceRecord | None:
        async with self._lock:
            trace_spans = self._traces.pop(trace_id, None)
            if trace_spans is None:
                return None
            spans = list(trace_spans.values())
        logger.info("trace_exported", trace_id=str(trace_id), span_count=len(spans))
        return TraceRecord(trace_id=trace_id, spans=spans)