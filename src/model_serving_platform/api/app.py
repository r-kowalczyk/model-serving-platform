"""Application factory that builds the full API service process.

This module defines `create_app`, which initialises the FastAPI application and
connects all runtime dependencies before the service accepts any HTTP requests.
It resolves settings, configures structured logging, validates the model bundle,
and initialises the GraphSAGE inference runtime used by prediction endpoints.
It also creates shared long-lived objects such as metrics, cache-backed external
enrichment clients, and the prediction service, then stores them on `app.state`.
Those shared objects are reused by route handlers so they are not recreated per
request, which keeps behaviour consistent and avoids repeated startup overhead.
The factory then registers middleware and route groups for health, metadata,
predictions, and metrics, and returns the fully wired FastAPI application.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

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

    # Resolve service settings once so all startup components read one consistent
    # configuration source, whether values came from injection or defaults.
    resolved_service_settings = service_settings or ServiceSettings()
    # Convert configured paths to absolute paths for startup logs so operators
    # can see the exact filesystem locations used by this process.
    resolved_bundle_path = str(Path(resolved_service_settings.bundle_path).resolve())
    resolved_cache_path = str(Path(resolved_service_settings.cache_path).resolve())
    # Configure structured logging before startup work so early failures are
    # captured with service identity fields and consistent log structure.
    configure_structured_logging(
        log_level=resolved_service_settings.log_level,
        service_name=resolved_service_settings.service_name,
        service_environment=resolved_service_settings.service_environment,
        service_version=resolved_service_settings.service_version,
    )
    # Create a bundle loader dependency, with optional injection for tests.
    resolved_bundle_loader = graph_sage_bundle_loader or GraphSageBundleLoader()
    # Create the process-wide metrics collector used by middleware, clients,
    # and route handlers to report counters and latency measurements.
    service_metrics = ServiceMetrics(enabled=resolved_service_settings.metrics_enabled)

    # Create the FastAPI application object with service identity metadata that
    # appears in OpenAPI output and operational tooling.
    application = FastAPI(
        title=resolved_service_settings.service_name,
        version=resolved_service_settings.service_version,
        description="Production-style GraphSAGE serving service",
    )
    # This startup log is emitted before any bundle IO so deployment issues can
    # be traced from resolved paths and lifecycle stage markers.
    app_logger.info(
        "startup_bundle_validation_started",
        extra={
            "resolved_bundle_path": resolved_bundle_path,
            "resolved_cache_path": resolved_cache_path,
            "bundle_validation_starting": True,
            "runtime_initialisation_starting": False,
            "runtime_initialisation_finished": False,
            "service_version": resolved_service_settings.service_version,
        },
    )
    # Validate and load model artefacts before serving requests, so startup
    # fails immediately if required bundle files are missing or invalid.
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
    # This log marks the handover point from bundle checks to runtime wiring so
    # operators can separate bundle faults from runtime initialisation faults.
    app_logger.info(
        "startup_runtime_initialisation_started",
        extra={
            "resolved_bundle_path": resolved_bundle_path,
            "resolved_cache_path": resolved_cache_path,
            "bundle_validation_starting": False,
            "runtime_initialisation_starting": True,
            "runtime_initialisation_finished": False,
            "service_version": resolved_service_settings.service_version,
        },
    )
    # Start a runtime initialisation timer to quantify startup cost and to make
    # runtime initialisation duration visible in structured logs.
    runtime_initialisation_start_timestamp = perf_counter()
    # Create a local cache store used to reuse external enrichment responses and
    # reduce repeated external network calls across requests.
    local_file_cache_store = LocalFileCacheStore(
        cache_directory_path=Path(resolved_service_settings.cache_path),
        ttl_seconds=resolved_service_settings.cache_ttl_seconds,
    )
    # Create the direct HTTP enrichment client that talks to external services
    # for description and interaction lookups used by inference runtime logic.
    http_external_enrichment_client = HttpExternalEnrichmentClient(
        description_lookup_url=resolved_service_settings.external_description_lookup_url,
        interaction_lookup_url=resolved_service_settings.external_interaction_lookup_url,
        timeout_seconds=resolved_service_settings.external_api_timeout_seconds,
        retry_count=resolved_service_settings.external_api_retry_count,
        retry_backoff_seconds=resolved_service_settings.external_api_retry_backoff_seconds,
        service_metrics=service_metrics,
    )
    # Wrap the HTTP enrichment client with a cache layer so repeated lookups can
    # be served from local storage when cached entries are still valid.
    external_enrichment_client = CachingExternalEnrichmentClient(
        wrapped_external_enrichment_client=http_external_enrichment_client,
        cache_store=local_file_cache_store,
        service_metrics=service_metrics,
    )
    try:
        # Allow runtime injection for deterministic tests; otherwise build the
        # GraphSAGE runtime from validated bundle metadata and enrichment client.
        resolved_inference_runtime = (
            inference_runtime
            or GraphSageInferenceRuntime.from_loaded_bundle_metadata(
                loaded_bundle_metadata=loaded_bundle_metadata,
                external_enrichment_client=external_enrichment_client,
                restricted_network_mode=resolved_service_settings.restricted_network_mode,
            )
        )
    except Exception:
        # On runtime initialisation failure, capture elapsed time and resolved
        # paths to separate configuration faults from runtime construction faults.
        runtime_initialisation_elapsed_milliseconds = int(
            (perf_counter() - runtime_initialisation_start_timestamp) * 1000
        )
        app_logger.exception(
            "startup_runtime_initialisation_failed",
            extra={
                "resolved_bundle_path": resolved_bundle_path,
                "resolved_cache_path": resolved_cache_path,
                "runtime_initialisation_elapsed_milliseconds": runtime_initialisation_elapsed_milliseconds,
                "runtime_initialisation_starting": True,
                "runtime_initialisation_finished": False,
                "service_version": resolved_service_settings.service_version,
            },
        )
        raise
    # Record runtime initialisation duration for successful startup visibility.
    runtime_initialisation_elapsed_milliseconds = int(
        (perf_counter() - runtime_initialisation_start_timestamp) * 1000
    )
    app_logger.info(
        "startup_runtime_initialisation_finished",
        extra={
            "runtime_name": resolved_inference_runtime.initialisation_summary.runtime_name,
            "runtime_model_num_layers": resolved_inference_runtime.initialisation_summary.model_num_layers,
            "runtime_base_embedding_count": resolved_inference_runtime.initialisation_summary.base_embedding_count,
            "resolved_bundle_path": resolved_bundle_path,
            "resolved_cache_path": resolved_cache_path,
            "bundle_validation_starting": False,
            "runtime_initialisation_starting": False,
            "runtime_initialisation_finished": True,
            "runtime_initialisation_elapsed_milliseconds": runtime_initialisation_elapsed_milliseconds,
            "bundle_version": loaded_bundle_metadata.bundle_version,
            "service_version": resolved_service_settings.service_version,
        },
    )

    # Readiness depends on successful bundle validation and runtime initialisation.
    # Store readiness state on app state so health endpoints can report whether
    # the service is currently able to process inference requests.
    application.state.runtime_state = ServiceRuntimeState(
        is_ready=resolved_inference_runtime.initialisation_summary.is_ready,
        readiness_reason=resolved_inference_runtime.initialisation_summary.readiness_reason,
    )
    # Store settings in app state so routes can return service metadata and
    # access configuration-derived values without reloading configuration.
    application.state.service_settings = resolved_service_settings
    # Store the shared metrics collector for middleware, routes, and clients.
    application.state.service_metrics = service_metrics
    # Store loaded bundle metadata for metadata endpoints and diagnostics.
    application.state.loaded_bundle_metadata = loaded_bundle_metadata
    # Store inference runtime so prediction service and other components can
    # access the ready model runtime instance for scoring operations.
    application.state.inference_runtime = resolved_inference_runtime
    # Create and store the prediction service that enforces API-level prediction
    # rules while delegating low-level scoring to the inference runtime.
    application.state.prediction_service = PredictionService(
        inference_runtime=resolved_inference_runtime,
        service_version=resolved_service_settings.service_version,
        bundle_version=loaded_bundle_metadata.bundle_version,
        max_top_k=resolved_service_settings.max_top_k,
        default_attachment_strategy=resolved_service_settings.default_attachment_strategy,
        restricted_network_mode=resolved_service_settings.restricted_network_mode,
        service_metrics=service_metrics,
    )
    # Store runtime summary values for metadata route output and diagnostics.
    application.state.runtime_initialisation_summary = (
        resolved_inference_runtime.initialisation_summary
    )
    # Store startup timestamp once so all endpoints report one consistent
    # service boot time for this running process.
    application.state.startup_timestamp = datetime.now(UTC).isoformat()

    # Add request context middleware so every HTTP request gets correlation
    # identifiers, structured lifecycle logs, and request-level metrics updates.
    application.add_middleware(
        RequestContextMiddleware,
        service_metrics=service_metrics,
    )
    # Register route groups after dependencies are ready so handlers can safely
    # read required objects from app state during request processing.
    application.include_router(health_router)
    application.include_router(metadata_router)
    application.include_router(prediction_router)
    application.include_router(metrics_router)
    # Return the fully wired FastAPI application to the ASGI server.
    return application
