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
from model_serving_platform.infrastructure.cache import LocalFileCacheStore
from model_serving_platform.infrastructure.clients import (
    CachingExternalEnrichmentClient,
    HttpExternalEnrichmentClient,
)
from model_serving_platform.infrastructure.graphsage.runtime import (
    GraphSageInferenceRuntime,
)
from model_serving_platform.infrastructure.logging import configure_structured_logging
from model_serving_platform.infrastructure.metrics import ServiceMetrics
from model_serving_platform.api.routes.metrics import metrics_router

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
    service_metrics = ServiceMetrics(enabled=resolved_service_settings.metrics_enabled)
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
    local_file_cache_store = LocalFileCacheStore(
        cache_directory_path=Path(resolved_service_settings.cache_path),
        ttl_seconds=resolved_service_settings.cache_ttl_seconds,
    )
    http_external_enrichment_client = HttpExternalEnrichmentClient(
        description_lookup_url=resolved_service_settings.external_description_lookup_url,
        interaction_lookup_url=resolved_service_settings.external_interaction_lookup_url,
        timeout_seconds=resolved_service_settings.external_api_timeout_seconds,
        retry_count=resolved_service_settings.external_api_retry_count,
        retry_backoff_seconds=resolved_service_settings.external_api_retry_backoff_seconds,
        service_metrics=service_metrics,
    )
    external_enrichment_client = CachingExternalEnrichmentClient(
        wrapped_external_enrichment_client=http_external_enrichment_client,
        cache_store=local_file_cache_store,
        service_metrics=service_metrics,
    )
    resolved_inference_runtime = (
        inference_runtime
        or GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata,
            external_enrichment_client=external_enrichment_client,
            restricted_network_mode=resolved_service_settings.restricted_network_mode,
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
    application.state.service_metrics = service_metrics
    application.state.loaded_bundle_metadata = loaded_bundle_metadata
    application.state.inference_runtime = resolved_inference_runtime
    application.state.prediction_service = PredictionService(
        inference_runtime=resolved_inference_runtime,
        service_version=resolved_service_settings.service_version,
        bundle_version=loaded_bundle_metadata.bundle_version,
        max_top_k=resolved_service_settings.max_top_k,
        default_attachment_strategy=resolved_service_settings.default_attachment_strategy,
        restricted_network_mode=resolved_service_settings.restricted_network_mode,
        service_metrics=service_metrics,
    )
    application.state.runtime_initialisation_summary = (
        resolved_inference_runtime.initialisation_summary
    )
    application.state.startup_timestamp = datetime.now(UTC).isoformat()

    application.add_middleware(
        RequestContextMiddleware,
        service_metrics=service_metrics,
    )
    application.include_router(health_router)
    application.include_router(metadata_router)
    application.include_router(prediction_router)
    application.include_router(metrics_router)
    return application
