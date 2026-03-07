"""Unit tests for GraphSAGE bundle loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model_serving_platform.infrastructure.bundles.errors import (
    GraphSageBundleValidationError,
)
from model_serving_platform.infrastructure.bundles.loader import GraphSageBundleLoader
from tests.bundle_test_data import write_valid_bundle


def test_loader_returns_metadata_for_valid_bundle(tmp_path: Path) -> None:
    """Verify loader returns metadata when bundle files satisfy the contract.

    This test proves the happy path for startup validation and provides
    confidence that application readiness can be enabled after loading.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "valid-bundle"
    )
    graph_sage_bundle_loader = GraphSageBundleLoader()

    loaded_bundle_metadata = graph_sage_bundle_loader.load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )

    assert loaded_bundle_metadata.node_count == 3
    assert loaded_bundle_metadata.feature_dimension == 4
    assert loaded_bundle_metadata.edge_count == 3
    assert loaded_bundle_metadata.model_architecture["num_layers"] == 2


def test_loader_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    """Verify loader raises when one required core bundle file is absent.

    Startup must stop immediately when required files are missing because
    serving cannot proceed without a complete GraphSAGE bundle contract.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "missing-file-bundle"
    )
    (valid_bundle_directory_path / "model_state.pt").unlink()
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "missing_required_files"


def test_loader_fails_when_manifest_schema_is_invalid(tmp_path: Path) -> None:
    """Verify loader raises when required manifest keys are missing.

    Manifest schema validation is first-class because model reconstruction
    depends on architecture and mapping keys being present and correctly typed.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "invalid-manifest-bundle"
    )
    manifest_path = valid_bundle_directory_path / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data.pop("model")
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "invalid_manifest_schema"


def test_loader_fails_when_feature_dimension_does_not_match_manifest(
    tmp_path: Path,
) -> None:
    """Verify loader raises when feature width differs from model input dim.

    This check protects GraphSAGE reconstruction from dimension mismatch
    errors by rejecting the bundle before serving is marked as ready.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "dimension-mismatch-bundle"
    )
    np.save(
        valid_bundle_directory_path / "node_features.npy",
        np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
    )
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "feature_dimension_mismatch"


def test_loader_fails_when_edge_index_first_dimension_is_not_two(
    tmp_path: Path,
) -> None:
    """Verify loader raises when edge index does not follow COO row layout.

    The contract requires `edge_index` shape `(2, edge_count)` so the loader
    rejects any other layout before runtime graph logic can proceed.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "edge-index-shape-bundle"
    )
    np.save(
        valid_bundle_directory_path / "edge_index.npy",
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "invalid_edge_index_shape"


def test_loader_fails_when_node_features_rank_is_not_two(tmp_path: Path) -> None:
    """Verify loader raises when node features array rank is invalid.

    This test ensures startup validation rejects one dimensional features
    because GraphSAGE serving requires a two dimensional feature matrix.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "node-features-rank-bundle"
    )
    np.save(
        valid_bundle_directory_path / "node_features.npy",
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "invalid_node_features_rank"


def test_loader_fails_when_edge_index_rank_is_not_two(tmp_path: Path) -> None:
    """Verify loader raises when edge index array rank is invalid.

    This test ensures startup validation rejects one dimensional edge indexes
    because serving expects a two row COO structure for graph edges.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "edge-index-rank-bundle"
    )
    np.save(
        valid_bundle_directory_path / "edge_index.npy",
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "invalid_edge_index_rank"
