import statistics
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Metric:
    name: str
    value: float
    unit: str = ""
    timestamp: float | None = None


class MetricsCollector:
    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.labels: dict[str, str] = {}
        self.snapshots: list[dict[str, float]] = []

    def record(self, name: str, value: float, unit: str = ""):
        self.metrics[name].append(value)
        if unit:
            self.labels[name] = unit

    def record_multiple(self, metrics_dict: dict[str, float]):
        for name, value in metrics_dict.items():
            self.record(name, value)

    def get_snapshot(self) -> dict[str, float]:
        snapshot = {}
        for name, values in self.metrics.items():
            if values:
                snapshot[f"{name}_mean"] = statistics.mean(values)
                snapshot[f"{name}_median"] = statistics.median(values)
                if len(values) > 1:
                    snapshot[f"{name}_stdev"] = statistics.stdev(values)
                snapshot[f"{name}_min"] = min(values)
                snapshot[f"{name}_max"] = max(values)
                snapshot[f"{name}_count"] = len(values)
        return snapshot

    def get_metric_stats(self, name: str) -> dict[str, float] | None:
        if name not in self.metrics or not self.metrics[name]:
            return None

        values = self.metrics[name]
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def get_all_metrics(self) -> dict[str, list[float]]:
        return dict(self.metrics)

    def reset(self):
        self.metrics.clear()
        self.labels.clear()
        self.snapshots.clear()

    def aggregate_by_percentile(self, name: str, percentile: int) -> float | None:
        if name not in self.metrics or not self.metrics[name]:
            return None

        sorted_values = sorted(self.metrics[name])
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_unit(self, name: str) -> str:
        return self.labels.get(name, "")
