"""
Regression tests for graceful optional-tracing startup (issue #17).

opentelemetry-instrumentation-fastapi (and the other instrumentor packages
kodiak/config/tracing.py imports) is not a declared dependency -- it's
genuinely absent in a normal install, which is exactly the condition these
tests exercise for real rather than mocking it.
"""

from __future__ import annotations

import structlog.testing

from kodiak.api.main import app, lifespan


async def test_missing_otel_instrumentation_logs_info_not_warning():
    with structlog.testing.capture_logs() as logs:
        async with lifespan(app):
            pass

    unavailable = [e for e in logs if e["event"] == "tracing_unavailable"]
    assert len(unavailable) == 1, f"expected one tracing_unavailable log, got: {logs}"
    assert unavailable[0]["log_level"] == "info"
    assert "opentelemetry" in unavailable[0]["reason"]

    # The old behavior (issue #17) logged this exact condition as a warning
    # under event name "tracing_initialization_failed" -- make sure that's
    # gone, not just that a new info log was added alongside it.
    assert not any(e["event"] == "tracing_initialization_failed" for e in logs)


async def test_unexpected_tracing_error_still_logs_a_warning(monkeypatch):
    import kodiak.api.main as main_module

    def _boom(name):
        raise RuntimeError("something unrelated to a missing package")

    monkeypatch.setattr(main_module.importlib, "import_module", _boom)

    with structlog.testing.capture_logs() as logs:
        async with lifespan(app):
            pass

    failed = [e for e in logs if e["event"] == "tracing_initialization_failed"]
    assert len(failed) == 1, f"expected one tracing_initialization_failed log, got: {logs}"
    assert failed[0]["log_level"] == "warning"


async def test_available_tracing_module_is_actually_invoked(monkeypatch):
    """Regression for the init_tracing/configure_tracing name mismatch:
    main.py checked hasattr(tracing, "init_tracing"), but tracing.py only
    ever defined configure_tracing, so tracing silently never started even
    when fully available. Confirm configure_tracing is the name looked up
    and called."""
    import kodiak.api.main as main_module

    calls = []

    class _FakeTracingModule:
        def configure_tracing(self):
            calls.append("configure_tracing")

    monkeypatch.setattr(main_module.importlib, "import_module", lambda name: _FakeTracingModule())

    with structlog.testing.capture_logs() as logs:
        async with lifespan(app):
            pass

    assert calls == ["configure_tracing"]
    assert any(e["event"] == "tracing_initialized" for e in logs)
