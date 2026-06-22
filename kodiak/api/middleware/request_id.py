"""
kodiak/api/middleware/request_id.py

Assigns a unique request ID to every inbound request and threads it through
the entire call stack: logs, traces, error responses, and outbound calls to
GitHub/LLM providers. This is the single value that lets you correlate a
user-reported bug with a specific agent run across Postgres, Redis, and
OpenTelemetry traces.

Must be the outermost middleware registered (see middleware/__init__.py) so
every other middleware and route handler can read `request.state.request_id`.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

# Exposed so logging/tracing setup (config/logging.py, config/tracing.py) can
# pull the current request id without needing direct access to the request.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, trust_incoming_header: bool = True) -> None:
        super().__init__(app)
        # In production, only trust an incoming X-Request-ID if it came through
        # a trusted ingress (e.g. an internal load balancer that strips/sets it).
        # Set to False for public-facing deployments to prevent log injection.
        self.trust_incoming_header = trust_incoming_header

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER) if self.trust_incoming_header else None
        request_id = incoming or str(uuid.uuid4())

        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)

        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def get_current_request_id() -> str | None:
    """Safe accessor for code that has no direct handle on the Request object."""
    return request_id_ctx.get()