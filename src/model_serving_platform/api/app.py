"""FastAPI application factory."""

from datetime import UTC, datetime

from fastapi import FastAPI

from model_serving_platform.api.routes.health import health_router
from model_serving_platform.application.service_state import ServiceRuntimeState
from model_serving_platform.config.settings import ServiceSettings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    The factory pattern keeps startup wiring testable and explicit, which
    makes staged development easier as runtime dependencies are introduced.
    The app state includes startup metadata and a readiness placeholder.
    Parameters: none.
    """

    service_settings = ServiceSettings()
    application = FastAPI(
        title=service_settings.service_name,
        version=service_settings.service_version,
        description="Production-style GraphSAGE serving service",
    )

    # Stage 1 keeps readiness false until bundle and model startup exists.
    application.state.runtime_state = ServiceRuntimeState(
        is_ready=False,
        readiness_reason="startup dependencies are not initialised yet",
    )
    application.state.startup_timestamp = datetime.now(UTC).isoformat()

    application.include_router(health_router)
    return application
