from __future__ import annotations

import threading
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class MetricType(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricRecord(BaseModel):
    name: str
    type: MetricType
    value: float | list[float]
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MetricsSnapshot(BaseModel):
    metrics: list[MetricRecord]
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _make_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    sorted_labels = sorted(labels.items())
    label_str = ",".join(f"{k}={v}" for k, v in sorted_labels)
    return f"{name}{{{label_str}}}"


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, MetricRecord] = {}
        self._registered_types: dict[str, MetricType] = {}
        self._lock = threading.Lock()

    def register_metric(self, name: str, metric_type: MetricType) -> None:
        with self._lock:
            if name not in self._registered_types:
                self._registered_types[name] = metric_type
                logger.debug("metric_registered", name=name, type=metric_type)

    def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        key = _make_key(name, labels or {})
        with self._lock:
            current = self._metrics.get(key)
            current_val = (
                current.value
                if isinstance(current, MetricRecord) and isinstance(current.value, float)
                else 0.0
            )
            self._metrics[key] = MetricRecord(
                name=name,
                type=MetricType.COUNTER,
                value=current_val + value,
                labels=labels or {},
            )

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = _make_key(name, labels or {})
        with self._lock:
            self._metrics[key] = MetricRecord(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                labels=labels or {},
            )

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        key = _make_key(name, labels or {})
        with self._lock:
            current = self._metrics.get(key)
            history: list[float] = (
                current.value
                if isinstance(current, MetricRecord) and isinstance(current.value, list)
                else []
            )
            history.append(value)
            self._metrics[key] = MetricRecord(
                name=name,
                type=MetricType.HISTOGRAM,
                value=history,
                labels=labels or {},
            )

    def record_timer(
        self, name: str, duration_secs: float, labels: dict[str, str] | None = None
    ) -> None:
        key = _make_key(name, labels or {})
        with self._lock:
            current = self._metrics.get(key)
            history: list[float] = (
                current.value
                if isinstance(current, MetricRecord) and isinstance(current.value, list)
                else []
            )
            history.append(duration_secs)
            self._metrics[key] = MetricRecord(
                name=name,
                type=MetricType.TIMER,
                value=history,
                labels=labels or {},
            )

    def get_metric(self, name: str, labels: dict[str, str] | None = None) -> MetricRecord | None:
        key = _make_key(name, labels or {})
        with self._lock:
            return self._metrics.get(key)

    def get_all_metrics(self) -> list[MetricRecord]:
        with self._lock:
            return list(self._metrics.values())

    def remove_metric(self, name: str, labels: dict[str, str] | None = None) -> bool:
        key = _make_key(name, labels or {})
        with self._lock:
            if key in self._metrics:
                del self._metrics[key]
                return True
            return False

    def export_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            metrics = [m.model_copy(deep=True) for m in self._metrics.values()]
        return MetricsSnapshot(metrics=metrics)

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._registered_types.clear()
            logger.info("metrics_registry_reset")
