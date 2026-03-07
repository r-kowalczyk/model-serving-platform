"""FastAPI application factory."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI

from model_serving_platform.api.middleware.request_context import (
    RequestContextMiddleware,
)
from model_serving_platform.api.routes.health import health_router
from model_serving_platform.api.routes.metadata import metadata_router
from model_serving_platform.api.routes.predictions import prediction_router
from model_serving_platform.application.inference_runtime import InferenceRuntime
from model_serving_platform.application.prediction_service import PredictionService
from model_serving_platform.application.service_state import ServiceRuntimeState
from model_serving_platform.config.settings import ServiceSettings
from model_serving_platform.infrastructure.bundles.loader import GraphSageBundleLoader
from model_serving_platform.infrastructure.graphsage.runtime import (
    GraphSageInferenceRuntime,
)
from model_serving_platform.infrastructure.logging import configure_structured_logging

app_logger = logging.getLogger("model_serving_platform.app")


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
    configure_structured_logging(
        log_level=resolved_service_settings.log_level,
        service_name=resolved_service_settings.service_name,
        service_environment=resolved_service_settings.service_environment,
        service_version=resolved_service_settings.service_version,
    )
    resolved_bundle_loader = graph_sage_bundle_loader or GraphSageBundleLoader()
    application = FastAPI(
        title=resolved_service_settings.service_name,
        version=resolved_service_settings.service_version,
        description="Production-style GraphSAGE serving service",
    )
    loaded_bundle_metadata = resolved_bundle_loader.load_and_validate_bundle(
        bundle_directory_path=Path(resolved_service_settings.bundle_path)
    )
    app_logger.info(
        "bundle_loaded",
        extra={
            "bundle_path": loaded_bundle_metadata.bundle_path,
            "bundle_version": loaded_bundle_metadata.bundle_version,
            "feature_dimension": loaded_bundle_metadata.feature_dimension,
            "node_count": loaded_bundle_metadata.node_count,
            "edge_count": loaded_bundle_metadata.edge_count,
            "service_version": resolved_service_settings.service_version,
        },
    )
    resolved_inference_runtime = (
        inference_runtime
        or GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )
    app_logger.info(
        "runtime_initialised",
        extra={
            "runtime_name": resolved_inference_runtime.initialisation_summary.runtime_name,
            "runtime_model_num_layers": resolved_inference_runtime.initialisation_summary.model_num_layers,
            "runtime_base_embedding_count": resolved_inference_runtime.initialisation_summary.base_embedding_count,
            "bundle_version": loaded_bundle_metadata.bundle_version,
            "service_version": resolved_service_settings.service_version,
        },
    )

    # Readiness depends on successful bundle validation and runtime initialisation.
    application.state.runtime_state = ServiceRuntimeState(
        is_ready=resolved_inference_runtime.initialisation_summary.is_ready,
        readiness_reason=resolved_inference_runtime.initialisation_summary.readiness_reason,
    )
    application.state.service_settings = resolved_service_settings
    application.state.loaded_bundle_metadata = loaded_bundle_metadata
    application.state.inference_runtime = resolved_inference_runtime
    application.state.prediction_service = PredictionService(
        inference_runtime=resolved_inference_runtime,
        service_version=resolved_service_settings.service_version,
        bundle_version=loaded_bundle_metadata.bundle_version,
        max_top_k=resolved_service_settings.max_top_k,
        default_attachment_strategy=resolved_service_settings.default_attachment_strategy,
    )
    application.state.runtime_initialisation_summary = (
        resolved_inference_runtime.initialisation_summary
    )
    application.state.startup_timestamp = datetime.now(UTC).isoformat()

    application.add_middleware(RequestContextMiddleware)
    application.include_router(health_router)
    application.include_router(metadata_router)
    application.include_router(prediction_router)
    return application
