from fastapi import FastAPI

from kodiak.api.middleware.audit import AuditMiddleware
from kodiak.api.middleware.cors import configure_cors
from kodiak.api.middleware.request_id import RequestIdMiddleware
from kodiak.api.routers import agents, approvals, auth, health, memory, plugins, repositories, tasks
from kodiak.api.routers.webhooks import github
from kodiak.config.logging import configure_logging
from kodiak.config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(AuditMiddleware)
    configure_cors(app)

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(repositories.router, prefix="/api/v1")
    app.include_router(agents.router, prefix="/api/v1")
    app.include_router(memory.router, prefix="/api/v1")
    app.include_router(plugins.router, prefix="/api/v1")
    app.include_router(approvals.router, prefix="/api/v1")
    app.include_router(github.router, prefix="/api/v1")
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "kodiak.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=not settings.is_test,
    )
