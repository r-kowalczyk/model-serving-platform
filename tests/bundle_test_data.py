"""Shared helpers for creating deterministic GraphSAGE bundle fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model_serving_platform.infrastructure.graphsage.pytorch_encoder import (
    TrainingMatchedGraphSageEncoder,
)


def write_valid_bundle(bundle_directory_path: Path) -> Path:
    """Create a minimal valid GraphSAGE bundle for service tests.

    This helper keeps tests deterministic and fast by writing a tiny bundle
    that satisfies the Stage 2 serving contract without real model runtime.
    Parameters: bundle_directory_path is created and populated in place.
    """

    bundle_directory_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    encoder_module = TrainingMatchedGraphSageEncoder(
        input_dimension=4,
        hidden_dimension=8,
        output_dimension=4,
        dropout_rate=0.1,
        num_layers=2,
    )
    torch.save(
        encoder_module.state_dict(),
        bundle_directory_path / "model_state.pt",
    )
    manifest_data = {
        "bundle_version": "test-bundle-v1",
        "node_id_to_index": {"N1": 0, "N2": 1, "N3": 2},
        "index_to_node_id": ["N1", "N2", "N3"],
        "node_name_to_id": {"Node One": "N1", "Node Two": "N2", "Node Three": "N3"},
        "node_display_name_by_id": {
            "N1": "Node One",
            "N2": "Node Two",
            "N3": "Node Three",
        },
        "semantic_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "semantic_max_length": 128,
        "is_undirected": True,
        "attachment_seed": 42,
        "attachment_top_k": 5,
        "model": {
            "input_dim": 4,
            "hidden_dim": 8,
            "output_dim": 4,
            "dropout": 0.1,
            "decoder_type": "mlp",
            "decoder_hidden_dim": 8,
            "num_layers": 2,
        },
    }
    (bundle_directory_path / "manifest.json").write_text(
        json.dumps(manifest_data),
        encoding="utf-8",
    )
    np.save(
        bundle_directory_path / "node_features.npy",
        np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.5, 0.5, 0.5, 0.5]],
            dtype=np.float32,
        ),
    )
    # Align with graph-link-prediction: undirected graphs store both directions for PyG message passing.
    base_edge_pairs = [(0, 1), (1, 2), (2, 0)]
    reversed_edge_pairs = [(target, source) for source, target in base_edge_pairs]
    edge_pair_array = np.asarray(base_edge_pairs + reversed_edge_pairs, dtype=np.int64)
    np.save(bundle_directory_path / "edge_index.npy", edge_pair_array.T)
    (bundle_directory_path / "resolver_cache.json").write_text("{}", encoding="utf-8")
    (bundle_directory_path / "interaction_cache.json").write_text(
        "{}", encoding="utf-8"
    )
    return bundle_directory_path
