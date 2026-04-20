"""Unit tests for `TrainingMatchedGraphSageEncoder` and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from model_serving_platform.infrastructure.graphsage.pytorch_encoder import (
    TrainingMatchedGraphSageEncoder,
    load_raw_checkpoint_mapping,
)


def test_single_layer_encoder_forward_matches_output_dimension() -> None:
    """One convolution maps input directly to output_dim when ``num_layers`` is 1."""

    encoder_module = TrainingMatchedGraphSageEncoder(
        input_dimension=4,
        hidden_dimension=8,
        output_dimension=6,
        dropout_rate=0.0,
        num_layers=1,
    )
    node_features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    embeddings = encoder_module(node_features, edge_index)
    assert embeddings.shape == (3, 6)


def test_three_layer_encoder_runs_middle_convolutions_and_residual_projection() -> None:
    """Three layers exercise the inner loop and optional final projection onto output_dim."""

    encoder_module = TrainingMatchedGraphSageEncoder(
        input_dimension=4,
        hidden_dimension=8,
        output_dimension=4,
        dropout_rate=0.0,
        num_layers=3,
    )
    assert encoder_module.final_residual_projection is not None
    node_features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    embeddings = encoder_module(node_features, edge_index)
    assert embeddings.shape == (3, 4)


def test_load_raw_checkpoint_mapping_unwraps_top_level_state_dict(
    tmp_path: Path,
) -> None:
    """Top-level ``{"state_dict": ...}`` checkpoints return the inner mapping."""

    checkpoint_path = tmp_path / "checkpoint.pt"
    inner_state = {"convolutions.0.lin_l.weight": torch.zeros(8, 4)}
    torch.save({"state_dict": inner_state}, checkpoint_path)
    loaded_mapping = load_raw_checkpoint_mapping(str(checkpoint_path))
    assert set(loaded_mapping) == set(inner_state)
    assert torch.equal(
        loaded_mapping["convolutions.0.lin_l.weight"],
        inner_state["convolutions.0.lin_l.weight"],
    )


def test_load_raw_checkpoint_mapping_rejects_non_mapping_state_dict_entry() -> None:
    """``state_dict`` must be a mapping when the wrapper key is present."""

    with patch(
        "model_serving_platform.infrastructure.graphsage.pytorch_encoder.torch.load",
        return_value={"state_dict": "not-a-mapping"},
    ):
        with pytest.raises(TypeError, match="mapping"):
            load_raw_checkpoint_mapping("/ignored/path.pt")


def test_load_raw_checkpoint_mapping_rejects_non_dict_checkpoint() -> None:
    """Checkpoint root must be a dict (either raw state or wrapper)."""

    with patch(
        "model_serving_platform.infrastructure.graphsage.pytorch_encoder.torch.load",
        return_value=["not", "a", "dict"],
    ):
        with pytest.raises(TypeError, match="state dict mapping"):
            load_raw_checkpoint_mapping("/ignored/path.pt")
