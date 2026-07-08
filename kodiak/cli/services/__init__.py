"""Kodiak CLI Services.

This package contains the service layer for the Kodiak CLI.

Architecture:
    CLI Commands
        ↓
    CLI Services
        ↓
    Agents
        ↓
    Database / GitHub / AI

The service layer is responsible for orchestration only. Services validate
inputs, coordinate one or more agents, handle domain-specific exceptions,
and return structured models. They do not implement business logic or
perform any CLI presentation.

Services included:
    - AnalyzeService
    - PlannerService
    - TaskService
    - ReviewService
    - ExplainService
"""

from __future__ import annotations

from kodiak.cli.services.analyze_service import AnalyzeService
from kodiak.cli.services.explain_service import ExplainService
from kodiak.cli.services.planner_service import PlannerService
from kodiak.cli.services.review_service import ReviewService
from kodiak.cli.services.task_service import TaskService

__all__ = [
    "AnalyzeService",
    "PlannerService",
    "TaskService",
    "ReviewService",
    "ExplainService",
]