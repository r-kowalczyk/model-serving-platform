"""Strict JSON shape for `manifest.json` inside a GraphSAGE bundle.

The bundle loader reads `manifest.json` as plain text, parses JSON, then passes
the result through these Pydantic models. Pydantic checks types, required keys,
and allowed values (for example decoder type) before any heavy runtime wiring.

If the file is wrong, validation fails immediately with a clear error instead
of failing later inside model code with a confusing stack trace.

`GraphSageBundleManifest` is the top-level object: node identity maps, text model
settings used for enrichment, attachment tuning fields, nested architecture
numbers, and an optional bundle version string for operators.
"""

from typing import Literal

from pydantic import BaseModel


class GraphSageModelArchitecture(BaseModel):
    """Numbers and choices needed to rebuild the trained GraphSAGE stack.

    These fields must match how the bundle was exported. The inference runtime
    reads them when constructing layers and decoders. `input_dim` is also cross
    checked against `node_features.npy` column count during bundle loading.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    decoder_type: Literal["mlp", "bilinear", "dot_product"]
    decoder_hidden_dim: int
    num_layers: int


class GraphSageBundleManifest(BaseModel):
    """Full contract for one bundle's `manifest.json` file.

    Identity fields link stable node ids, internal indices, human-readable names,
    and lookup keys used when API requests name entities as strings.

    Semantic fields name the text encoder and maximum sequence length used when
    descriptions are turned into vectors for unseen or enriched nodes.

    Graph and attachment fields describe undirectedness and parameters for how
    the runtime attaches new information to the graph.

    The nested `model` block holds architecture hyperparameters. `bundle_version`
    is optional metadata only; it does not change tensor maths.
    """

    # Maps between exported node identifiers and training-time row order.
    node_id_to_index: dict[str, int]
    index_to_node_id: list[str]
    node_name_to_id: dict[str, str]
    node_display_name_by_id: dict[str, str]
    # Text encoder configuration for description-based enrichment paths.
    semantic_model_name: str
    semantic_max_length: int
    # Graph semantics and attachment behaviour for runtime scoring.
    is_undirected: bool
    attachment_seed: int
    attachment_top_k: int
    model: GraphSageModelArchitecture
    bundle_version: str | None = None
