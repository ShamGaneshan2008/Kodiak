"""
Application metrics via Prometheus.

Exposes a /metrics endpoint and provides typed helpers for
counters, histograms, and gauges used across Kodiak.
"""

from __future__ import annotations


try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Info,
        CollectorRegistry,
        generate_latest,
        multiprocess,
    )
except ImportError:
    Counter = None
    Gauge = None
    Histogram = None
    Info = None
    CollectorRegistry = None
    CONTENT_TYPE_LATEST = None
    REGISTRY = None
    generate_latest = None
    multiprocess = None

REQUESTS_TOTAL = (
    Counter("kodiak_requests_total", "Total API requests")
    if Counter is not None
    else None
)

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    multiprocess,
    CollectorRegistry,
)

# ── Application info ──────────────────────────────────────────────────────────
APP_INFO = Info("kodiak_app", "Kodiak application metadata")

# ── HTTP metrics ──────────────────────────────────────────────────────────────
HTTP_REQUESTS_TOTAL = Counter(
    "kodiak_http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "kodiak_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HTTP_REQUESTS_IN_FLIGHT = Gauge(
    "kodiak_http_requests_in_flight",
    "Number of HTTP requests currently being processed",
    ["method", "path"],
)

# ── LLM metrics ───────────────────────────────────────────────────────────────
LLM_REQUESTS_TOTAL = Counter(
    "kodiak_llm_requests_total",
    "Total LLM API calls",
    ["provider", "model", "status"],
)

LLM_TOKEN_USAGE = Counter(
    "kodiak_llm_tokens_total",
    "Total tokens consumed",
    ["provider", "model", "token_type"],  # token_type: prompt | completion
)

LLM_COST_USD = Counter(
    "kodiak_llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["provider", "model"],
)

LLM_LATENCY_SECONDS = Histogram(
    "kodiak_llm_latency_seconds",
    "Time to first token / full completion",
    ["provider", "model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# ── Agent task metrics ────────────────────────────────────────────────────────
AGENT_TASKS_TOTAL = Counter(
    "kodiak_agent_tasks_total",
    "Total agent tasks dispatched",
    ["agent_type", "status"],  # status: success | failure | timeout
)

AGENT_TASK_DURATION_SECONDS = Histogram(
    "kodiak_agent_task_duration_seconds",
    "End-to-end agent task duration",
    ["agent_type"],
    buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

ACTIVE_AGENT_TASKS = Gauge(
    "kodiak_active_agent_tasks",
    "Number of currently running agent tasks",
    ["agent_type"],
)

# ── RAG / retrieval metrics ───────────────────────────────────────────────────
RAG_RETRIEVALS_TOTAL = Counter(
    "kodiak_rag_retrievals_total",
    "Total RAG retrieval operations",
    ["status"],
)

RAG_CHUNKS_RETRIEVED = Histogram(
    "kodiak_rag_chunks_retrieved",
    "Number of chunks returned per retrieval",
    buckets=(1, 2, 5, 10, 20, 50),
)

RAG_RETRIEVAL_LATENCY_SECONDS = Histogram(
    "kodiak_rag_retrieval_latency_seconds",
    "Latency of vector store retrieval",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ── Worker / Celery metrics ───────────────────────────────────────────────────
CELERY_TASKS_TOTAL = Counter(
    "kodiak_celery_tasks_total",
    "Total Celery tasks processed",
    ["task_name", "status"],
)

CELERY_TASK_DURATION_SECONDS = Histogram(
    "kodiak_celery_task_duration_seconds",
    "Celery task processing duration",
    ["task_name"],
    buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 300.0),
)

# ── Sandbox metrics ───────────────────────────────────────────────────────────
SANDBOX_EXECUTIONS_TOTAL = Counter(
    "kodiak_sandbox_executions_total",
    "Total sandbox code execution attempts",
    ["status"],  # status: success | timeout | error | oom
)

SANDBOX_EXECUTION_DURATION_SECONDS = Histogram(
    "kodiak_sandbox_execution_duration_seconds",
    "Sandbox execution wall time",
    buckets=(0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
)

# ── GitHub integration metrics ────────────────────────────────────────────────
GITHUB_WEBHOOK_EVENTS_TOTAL = Counter(
    "kodiak_github_webhook_events_total",
    "Total GitHub webhook events received",
    ["event_type", "action"],
)

GITHUB_PR_CREATED_TOTAL = Counter(
    "kodiak_github_prs_created_total",
    "Total pull requests opened by Kodiak",
)

# ── DB / connection pool metrics ──────────────────────────────────────────────
DB_POOL_SIZE = Gauge(
    "kodiak_db_pool_size",
    "Current database connection pool size",
)

DB_POOL_CHECKED_OUT = Gauge(
    "kodiak_db_pool_checked_out",
    "Connections currently checked out from the pool",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def configure_metrics(settings: Any | None = None) -> None:
    """
    Initialise metrics metadata. Call once at startup.

    Args:
        settings: Optional Settings instance; auto-fetched if omitted.
    """
    if settings is None:
        from kodiak.config.settings import get_settings

        settings = get_settings()

    APP_INFO.info(
        {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
        }
    )


def metrics_response() -> tuple[bytes, str]:
    """
    Generate a Prometheus scrape response body and content-type header.
    Use in the /metrics FastAPI route.

    Returns:
        Tuple of (body_bytes, content_type_string).
    """
    import os

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        body = generate_latest(registry)
    else:
        body = generate_latest(REGISTRY)

    return body, CONTENT_TYPE_LATEST

# ============================================================================
# Compatibility aliases for BaseAgent
# ============================================================================

tasks_total = AGENT_TASKS_TOTAL
task_duration_seconds = AGENT_TASK_DURATION_SECONDS
active_tasks = ACTIVE_AGENT_TASKS