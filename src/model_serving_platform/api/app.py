"""FastAPI application factory."""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI

from model_serving_platform.api.routes.health import health_router
from model_serving_platform.api.routes.metadata import metadata_router
from model_serving_platform.application.inference_runtime import InferenceRuntime
from model_serving_platform.application.service_state import ServiceRuntimeState
from model_serving_platform.config.settings import ServiceSettings
from model_serving_platform.infrastructure.bundles.loader import GraphSageBundleLoader
from model_serving_platform.infrastructure.graphsage.runtime import (
    GraphSageInferenceRuntime,
)


def create_app(
    service_settings: ServiceSettings | None = None,
    graph_sage_bundle_loader: GraphSageBundleLoader | None = None,
    inference_runtime: InferenceRuntime | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    The factory pattern keeps startup wiring testable and explicit, which
    makes staged development easier as runtime dependencies are introduced.
    Stage 3 also initialises an inference runtime and links readiness to it.
    Parameters: optional startup dependencies support deterministic tests.
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
    resolved_inference_runtime = (
        inference_runtime
        or GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    # Readiness depends on successful bundle validation and runtime initialisation.
    application.state.runtime_state = ServiceRuntimeState(
        is_ready=resolved_inference_runtime.initialisation_summary.is_ready,
        readiness_reason=resolved_inference_runtime.initialisation_summary.readiness_reason,
    )
    application.state.service_settings = resolved_service_settings
    application.state.loaded_bundle_metadata = loaded_bundle_metadata
    application.state.inference_runtime = resolved_inference_runtime
    application.state.runtime_initialisation_summary = (
        resolved_inference_runtime.initialisation_summary
    )
    application.state.startup_timestamp = datetime.now(UTC).isoformat()

    application.include_router(health_router)
    application.include_router(metadata_router)
    return application
