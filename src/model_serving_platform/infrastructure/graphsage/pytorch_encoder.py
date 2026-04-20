"""PyTorch GraphSAGE encoder aligned with graph-link-prediction training exports.

The class `TrainingMatchedGraphSageEncoder` duplicates the layout of
`graph_lp.graphsage.GraphSageEncoder` in the graph-link-prediction repository.
Keep both definitions logically in sync when changing architecture.

Checkpoint format from training (`torch.save(model.state_dict())` on
`GraphSageLinkPredictor`) uses keys prefixed with ``encoder.`` for these
weights. Standalone encoder-only exports use the same tensor names without that
prefix (used by unit test bundles).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
import torch.nn.functional as torch_functional
from torch_geometric.nn import SAGEConv


class TrainingMatchedGraphSageEncoder(nn.Module):
    """PyG SAGEConv stack with residuals; matches graph-link-prediction training.

    This mirrors `graph_lp.graphsage.GraphSageEncoder` so `model_state.pt` from
    `export_graphsage_bundle` loads without key translation beyond the optional
    ``encoder.`` prefix on full-model checkpoints.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout_rate: float,
        num_layers: int = 2,
    ) -> None:
        """Initialise layer stack from manifest hyperparameters."""

        super().__init__()
        self.dropout_rate = float(dropout_rate)
        self.convolutions = nn.ModuleList()

        if num_layers == 1:
            self.convolutions.append(SAGEConv(input_dimension, output_dimension))
        else:
            self.convolutions.append(SAGEConv(input_dimension, hidden_dimension))
            for _ in range(num_layers - 2):
                self.convolutions.append(SAGEConv(hidden_dimension, hidden_dimension))
            self.convolutions.append(SAGEConv(hidden_dimension, output_dimension))

        needs_final_projection = num_layers > 1 and hidden_dimension != output_dimension
        self.final_residual_projection: nn.Linear | None = (
            nn.Linear(hidden_dimension, output_dimension, bias=False)
            if needs_final_projection
            else None
        )

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        """Return node embeddings of shape `(num_nodes, output_dimension)`."""

        hidden = self.convolutions[0](node_features, edge_index)
        hidden = torch_functional.relu(hidden)
        hidden = torch_functional.dropout(
            hidden, p=self.dropout_rate, training=self.training
        )

        for convolution in self.convolutions[1:-1]:
            residual = hidden
            hidden = convolution(hidden, edge_index)
            hidden = hidden + residual
            hidden = torch_functional.relu(hidden)
            hidden = torch_functional.dropout(
                hidden, p=self.dropout_rate, training=self.training
            )

        if len(self.convolutions) > 1:
            residual = hidden
            hidden = self.convolutions[-1](hidden, edge_index)
            if self.final_residual_projection is not None:
                residual = self.final_residual_projection(residual)
            hidden = hidden + residual

        return cast(Tensor, hidden)


def load_raw_checkpoint_mapping(model_state_path: str) -> dict[str, Tensor]:
    """Load tensor mapping from `model_state.pt` on CPU."""

    payload = torch.load(
        model_state_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if isinstance(payload, dict) and "state_dict" in payload:
        raw_state = payload["state_dict"]
        if not isinstance(raw_state, dict):
            raise TypeError("Checkpoint state_dict entry must be a mapping.")
        return raw_state
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint must be a state dict mapping or wrap one.")


def select_encoder_weights(state_dictionary: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep only encoder tensors and strip `encoder.` prefix when present.

    Full link-predictor checkpoints from graph-link-prediction use keys such as
    ``encoder.convolutions.0.lin_l.weight``. Encoder-only bundles use
    ``convolutions.0.lin_l.weight`` directly.
    """

    encoder_keys = [key for key in state_dictionary if key.startswith("encoder.")]
    if encoder_keys:
        return {key[len("encoder.") :]: state_dictionary[key] for key in encoder_keys}
    return state_dictionary


def build_encoder_and_load_weights(
    *,
    model_state_path: str,
    input_dimension: int,
    hidden_dimension: int,
    output_dimension: int,
    num_layers: int,
    dropout_probability: float,
) -> TrainingMatchedGraphSageEncoder:
    """Construct an encoder and load weights strictly from the bundle checkpoint."""

    encoder_module = TrainingMatchedGraphSageEncoder(
        input_dimension=input_dimension,
        hidden_dimension=hidden_dimension,
        output_dimension=output_dimension,
        dropout_rate=dropout_probability,
        num_layers=num_layers,
    )
    raw_mapping = load_raw_checkpoint_mapping(model_state_path=model_state_path)
    encoder_state = select_encoder_weights(raw_mapping)
    encoder_module.load_state_dict(encoder_state, strict=True)
    encoder_module.eval()
    return encoder_module


@torch.no_grad()
def encode_all_nodes(
    encoder_module: TrainingMatchedGraphSageEncoder,
    node_feature_matrix: NDArray[np.float32] | NDArray[np.float64],
    edge_index_array: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Encode every graph node once and return a NumPy `(N, output_dim)` matrix."""

    node_features = torch.from_numpy(np.asarray(node_feature_matrix)).to(
        dtype=torch.float32
    )
    edge_index_tensor = torch.from_numpy(np.asarray(edge_index_array)).long()
    embeddings = encoder_module(node_features, edge_index_tensor)
    return np.asarray(embeddings.cpu().numpy(), dtype=np.float64)
