"""Shared helpers for creating deterministic GraphSAGE bundle fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_valid_bundle(bundle_directory_path: Path) -> Path:
    """Create a minimal valid GraphSAGE bundle for service tests.

    This helper keeps tests deterministic and fast by writing a tiny bundle
    that satisfies the Stage 2 serving contract without real model runtime.
    Parameters: bundle_directory_path is created and populated in place.
    """

    bundle_directory_path.mkdir(parents=True, exist_ok=True)
    (bundle_directory_path / "model_state.pt").write_bytes(b"stage2-placeholder-state")
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
    np.save(
        bundle_directory_path / "edge_index.npy",
        np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
    )
    (bundle_directory_path / "resolver_cache.json").write_text("{}", encoding="utf-8")
    (bundle_directory_path / "interaction_cache.json").write_text(
        "{}", encoding="utf-8"
    )
    return bundle_directory_path
