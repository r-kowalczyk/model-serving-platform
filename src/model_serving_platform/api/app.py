"""FastAPI application factory."""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI

from model_serving_platform.api.routes.health import health_router
from model_serving_platform.api.routes.metadata import metadata_router
from model_serving_platform.application.service_state import ServiceRuntimeState
from model_serving_platform.config.settings import ServiceSettings
from model_serving_platform.infrastructure.bundles.loader import GraphSageBundleLoader


def create_app(
    service_settings: ServiceSettings | None = None,
    graph_sage_bundle_loader: GraphSageBundleLoader | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    The factory pattern keeps startup wiring testable and explicit, which
    makes staged development easier as runtime dependencies are introduced.
    Stage 2 performs fail-fast bundle validation before readiness is enabled.
    Parameters: optional settings and loader support test injection.
    """

    resolved_service_settings = service_settings or ServiceSettings()
    resolved_bundle_loader = graph_sage_bundle_loader or GraphSageBundleLoader()
    application = FastAPI(
        title=resolved_service_settings.service_name,
        version=resolved_service_settings.service_version,
        description="Production-style GraphSAGE serving service",
    )
    loaded_bundle_metadata = resolved_bundle_loader.load_and_validate_bundle(
        bundle_directory_path=Path(resolved_service_settings.bundle_path)
    )

    # Bundle validation is completed before readiness is marked true.
    application.state.runtime_state = ServiceRuntimeState(
        is_ready=True,
        readiness_reason="bundle and startup dependencies are initialised",
    )
    application.state.service_settings = resolved_service_settings
    application.state.loaded_bundle_metadata = loaded_bundle_metadata
    application.state.startup_timestamp = datetime.now(UTC).isoformat()

    application.include_router(health_router)
    application.include_router(metadata_router)
    return application
