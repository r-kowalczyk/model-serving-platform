"""Metadata endpoint for startup and bundle visibility."""

from fastapi import APIRouter, Request

from model_serving_platform.application.inference_runtime import (
    RuntimeInitialisationSummary,
)
from model_serving_platform.domain.metadata import (
    GraphSageBundleMetadataResponse,
    ServiceMetadataResponse,
)
from model_serving_platform.infrastructure.bundles.loader import (
    LoadedGraphSageBundleMetadata,
)

metadata_router = APIRouter(prefix="/v1")


@metadata_router.get("/metadata", response_model=ServiceMetadataResponse)
def get_service_metadata(request: Request) -> ServiceMetadataResponse:
    """Return service and bundle metadata for operational verification.

    This endpoint confirms that startup validation completed and identifies
    the loaded bundle contract details used by the GraphSAGE backend.
    Parameters: request gives access to application startup state.
    """

    loaded_bundle_metadata: LoadedGraphSageBundleMetadata = (
        request.app.state.loaded_bundle_metadata
    )
    runtime_initialisation_summary: RuntimeInitialisationSummary = (
        request.app.state.runtime_initialisation_summary
    )
    return ServiceMetadataResponse(
        service_version=request.app.state.service_settings.service_version,
        model_backend="graphsage",
        startup_timestamp=request.app.state.startup_timestamp,
        supported_attachment_strategies=["interaction", "cosine"],
        runtime_name=runtime_initialisation_summary.runtime_name,
        runtime_model_num_layers=runtime_initialisation_summary.model_num_layers,
        runtime_base_embedding_count=runtime_initialisation_summary.base_embedding_count,
        bundle_metadata=GraphSageBundleMetadataResponse(
            bundle_path=loaded_bundle_metadata.bundle_path,
            manifest_path=loaded_bundle_metadata.manifest_path,
            model_state_path=loaded_bundle_metadata.model_state_path,
            node_features_path=loaded_bundle_metadata.node_features_path,
            edge_index_path=loaded_bundle_metadata.edge_index_path,
            node_count=loaded_bundle_metadata.node_count,
            feature_dimension=loaded_bundle_metadata.feature_dimension,
            edge_count=loaded_bundle_metadata.edge_count,
            semantic_model_name=loaded_bundle_metadata.semantic_model_name,
            semantic_max_length=loaded_bundle_metadata.semantic_max_length,
            is_undirected=loaded_bundle_metadata.is_undirected,
            attachment_seed=loaded_bundle_metadata.attachment_seed,
            attachment_top_k=loaded_bundle_metadata.attachment_top_k,
            model_architecture=loaded_bundle_metadata.model_architecture,
            bundle_version=loaded_bundle_metadata.bundle_version,
        ),
    )
