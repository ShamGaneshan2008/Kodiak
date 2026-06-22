"""
Distributed tracing via OpenTelemetry.
Exports spans to an OTLP collector (Jaeger, Tempo, etc.).
Auto-instruments FastAPI, SQLAlchemy, httpx, Redis, and Celery.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine

F = TypeVar("F", bound=Callable[..., Any])

_tracer_provider: TracerProvider | None = None


def configure_tracing(
    engine: AsyncEngine | None = None,
    settings: Any | None = None,
) -> TracerProvider:
    """
    Initialise the global OpenTelemetry TracerProvider.
    Call once at startup, before the FastAPI app is created.

    Args:
        engine: Optional AsyncEngine for SQLAlchemy auto-instrumentation.
        settings: Optional Settings instance; fetched automatically if omitted.

    Returns:
        The configured TracerProvider.
    """
    global _tracer_provider

    if settings is None:
        from kodiak.config.settings import get_settings

        settings = get_settings()

    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: settings.OTEL_SERVICE_NAME,
            ResourceAttributes.SERVICE_VERSION: settings.APP_VERSION,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: settings.ENVIRONMENT.value,
        }
    )

    provider = TracerProvider(resource=resource)

    # ── Exporters ─────────────────────────────────────────────────────────
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=not settings.is_production,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    elif settings.DEBUG:
        # Pretty-print spans to stdout in local dev when no collector is set
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    # ── Auto-instrumentation ──────────────────────────────────────────────
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    CeleryInstrumentor().instrument()

    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)

    return provider


def instrument_fastapi(app: FastAPI) -> None:
    """
    Attach OTel middleware to a FastAPI application.
    Call after ``configure_tracing`` and after the app object is created.
    """
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=_tracer_provider,
        excluded_urls="health,readyz,livez,metrics",
    )


def get_tracer(name: str) -> trace.Tracer:
    """
    Return a named tracer for manual instrumentation.

    Usage::

        from kodiak.config.tracing import get_tracer
        tracer = get_tracer(__name__)

        async def my_func():
            with tracer.start_as_current_span("my_func") as span:
                span.set_attribute("key", "value")
                ...
    """
    return trace.get_tracer(name)


def traced(span_name: str | None = None) -> Callable[[F], F]:
    """
    Decorator that wraps a function in an OTel span.
    Works with both sync and async functions.

    Usage::

        @traced("llm.complete")
        async def call_llm(prompt: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        name = span_name or func.__qualname__
        tracer = get_tracer(func.__module__)

        if _is_async(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(name):
                    return await func(*args, **kwargs)  # type: ignore[misc]

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(name):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _is_async(func: Callable[..., Any]) -> bool:
    import asyncio
    import inspect

    return asyncio.iscoroutinefunction(func) or inspect.iscoroutinefunction(func)


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider. Call on application shutdown."""
    if _tracer_provider is not None:
        _tracer_provider.shutdown()