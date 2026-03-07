"""Domain response models for service metadata exposure."""

from pydantic import BaseModel


class GraphSageBundleMetadataResponse(BaseModel):
    """Represent validated GraphSAGE bundle information for API clients.

    The metadata endpoint exposes this model so operators can verify exactly
    which bundle and architecture values were loaded during startup.
    Parameters: all values are generated from validated startup state.
    """

    bundle_path: str
    manifest_path: str
    model_state_path: str
    node_features_path: str
    edge_index_path: str
    node_count: int
    feature_dimension: int
    edge_count: int
    semantic_model_name: str
    semantic_max_length: int
    is_undirected: bool
    attachment_seed: int
    attachment_top_k: int
    model_architecture: dict[str, int | float | str]
    bundle_version: str | None


class ServiceMetadataResponse(BaseModel):
    """Represent high-level service startup metadata for operations checks.

    Returning this typed response keeps version and backend identity explicit.
    It also provides runtime timestamps that are useful for deployment audits.
    Parameters: fields are read from settings and startup application state.
    """

    service_version: str
    model_backend: str
    startup_timestamp: str
    supported_attachment_strategies: list[str]
    runtime_name: str
    runtime_model_num_layers: int
    runtime_base_embedding_count: int
    bundle_metadata: GraphSageBundleMetadataResponse
