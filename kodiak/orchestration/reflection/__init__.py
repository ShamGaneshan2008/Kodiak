"""Reflection and self-repair subsystem."""

from kodiak.orchestration.reflection.analyzer import FailureAnalyzer
from kodiak.orchestration.reflection.engine import ReflectionEngine
from kodiak.orchestration.reflection.models import (
    FailureCategory,
    ReflectionContext,
    ReflectionOutcome,
    ReflectionResult,
    RepairStrategy,
)

__all__ = [
    "FailureAnalyzer",
    "FailureCategory",
    "ReflectionContext",
    "ReflectionEngine",
    "ReflectionOutcome",
    "ReflectionResult",
    "RepairStrategy",
    "SelfRepairLoop",
]


def __getattr__(name: str) -> object:
    if name == "SelfRepairLoop":
        from kodiak.orchestration.reflection.loop import SelfRepairLoop

        return SelfRepairLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
