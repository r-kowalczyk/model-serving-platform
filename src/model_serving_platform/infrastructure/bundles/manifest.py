"""Pydantic models that define the Stage 2 GraphSAGE manifest contract."""

from typing import Literal

from pydantic import BaseModel


class GraphSageModelArchitecture(BaseModel):
    """Describe the GraphSAGE model architecture declared in the manifest.

    The serving runtime uses these exact values to reconstruct model layers.
    A typed schema is necessary so startup fails early when architecture keys
    are missing or malformed and model reconstruction would become unsafe.
    Parameters: each field is provided by manifest JSON.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    decoder_type: Literal["mlp", "bilinear", "dot_product"]
    decoder_hidden_dim: int
    num_layers: int


class GraphSageBundleManifest(BaseModel):
    """Define the authoritative Stage 2 serving-side bundle manifest schema.

    This schema mirrors the required upstream contract for GraphSAGE serving.
    It is intentionally strict so startup can fail fast when identity maps,
    semantic settings, or architecture metadata are inconsistent.
    Parameters: values are loaded from the bundle manifest file.
    """

    node_id_to_index: dict[str, int]
    index_to_node_id: list[str]
    node_name_to_id: dict[str, str]
    node_display_name_by_id: dict[str, str]
    semantic_model_name: str
    semantic_max_length: int
    is_undirected: bool
    attachment_seed: int
    attachment_top_k: int
    model: GraphSageModelArchitecture
    bundle_version: str | None = None
