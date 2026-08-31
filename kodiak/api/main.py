from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("starting_kodiak_api")

    # Initialize tracing if available. OpenTelemetry's instrumentation
    # packages are optional extras -- ModuleNotFoundError here just means
    # they aren't installed, which is a normal, expected configuration,
    # not something worth a warning. Anything else (a real misconfiguration
    # inside configure_tracing) still warrants one.
    try:
        tracing = importlib.import_module("kodiak.config.tracing")
        if hasattr(tracing, "configure_tracing"):
            tracing.configure_tracing()
            logger.info("tracing_initialized")
    except ModuleNotFoundError as exc:
        logger.info("tracing_unavailable", reason=str(exc))
    except Exception as exc:
        logger.warning("tracing_initialization_failed", error=str(exc))

    yield

    logger.info("shutting_down_kodiak_api")


app = FastAPI(
    title="Kodiak API",
    description="Autonomous AI Coding Agent Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------------------------
# Register Middleware (if available)
# --------------------------------------------------

middleware_dir = Path(__file__).parent / "middleware"

if middleware_dir.exists():
    for middleware_file in middleware_dir.glob("*.py"):
        if middleware_file.name.startswith("_"):
            continue

        module_name = f"kodiak.api.middleware.{middleware_file.stem}"

        try:
            module = importlib.import_module(module_name)

            if hasattr(module, "setup"):
                module.setup(app)
                logger.info(
                    "middleware_registered",
                    module=module_name,
                )

        except Exception as exc:
            logger.warning(
                "middleware_registration_failed",
                module=module_name,
                error=str(exc),
            )

# --------------------------------------------------
# Register Routers
# --------------------------------------------------

routers_dir = Path(__file__).parent / "routers"

for router_file in routers_dir.glob("*.py"):
    if router_file.name.startswith("_"):
        continue

    module_name = f"kodiak.api.routers.{router_file.stem}"

    try:
        module = importlib.import_module(module_name)

        if hasattr(module, "router"):
            app.include_router(module.router)

            logger.info(
                "router_registered",
                router=router_file.stem,
            )

        else:
            logger.warning(
                "router_not_found",
                module=module_name,
            )

    except Exception as exc:
        logger.exception(
            "router_registration_failed",
            module=module_name,
            error=str(exc),
        )


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "name": "Kodiak API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }
