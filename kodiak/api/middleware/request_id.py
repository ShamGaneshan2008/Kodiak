from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from kodiak.config.logging import bind_request_context, clear_request_context

logger = structlog.get_logger(__name__)

_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(_HEADER) or str(uuid.uuid4())
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
            response.headers[_HEADER] = request_id
            return response
        finally:
            clear_request_context()