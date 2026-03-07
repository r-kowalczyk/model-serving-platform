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
    assert captured_error.value.details["missing_file_names"] == ["model_state.pt"]
    assert "model_state.pt" in str(captured_error.value)


def test_loader_fails_when_bundle_directory_is_missing(tmp_path: Path) -> None:
    """Verify loader raises explicit diagnostics for missing directory path.

    Mount failures can surface as empty or absent paths in container startup,
    so this test checks that directory visibility diagnostics are explicit.
    Parameters: tmp_path is provided by pytest.
    """

    missing_bundle_directory_path = tmp_path / "missing-bundle-directory"
    graph_sage_bundle_loader = GraphSageBundleLoader()

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=missing_bundle_directory_path
        )

    assert captured_error.value.error_code == "invalid_bundle_directory"
    assert captured_error.value.details["bundle_directory_exists"] is False
    assert "exists=False" in str(captured_error.value)


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


def test_loader_creates_missing_optional_cache_files(tmp_path: Path) -> None:
    """Verify loader initialises optional cache files when they are absent.

    Optional cache files are deployment convenience artefacts so startup must
    create empty JSON files rather than fail when bundles omit these caches.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "optional-cache-bundle"
    )
    resolver_cache_file_path = valid_bundle_directory_path / "resolver_cache.json"
    interaction_cache_file_path = valid_bundle_directory_path / "interaction_cache.json"
    resolver_cache_file_path.unlink()
    interaction_cache_file_path.unlink()
    graph_sage_bundle_loader = GraphSageBundleLoader()

    loaded_bundle_metadata = graph_sage_bundle_loader.load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )

    assert loaded_bundle_metadata.node_count == 3
    assert resolver_cache_file_path.read_text(encoding="utf-8") == "{}"
    assert interaction_cache_file_path.read_text(encoding="utf-8") == "{}"


def test_loader_reports_list_directory_error_when_listing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify loader reports directory listing failures in diagnostics.

    This test forces the directory listing path to fail so startup diagnostics
    include concrete list-directory error text used for deployment triage.
    Parameters: tmp_path and monkeypatch are provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "list-error-bundle"
    )
    graph_sage_bundle_loader = GraphSageBundleLoader()

    # This monkeypatch simulates a mount visibility issue where listdir fails,
    # which proves diagnostics capture the exact listing failure message.
    def _raise_list_directory_permission_error(self: Path) -> object:
        raise PermissionError("forced directory listing error")

    monkeypatch.setattr(
        type(valid_bundle_directory_path),
        "iterdir",
        _raise_list_directory_permission_error,
    )

    collected_bundle_directory_diagnostics = (
        graph_sage_bundle_loader._collect_bundle_directory_diagnostics(
            bundle_directory_path=valid_bundle_directory_path
        )
    )

    assert collected_bundle_directory_diagnostics["bundle_directory_exists"] is True
    assert collected_bundle_directory_diagnostics["bundle_directory_is_dir"] is True
    assert (
        collected_bundle_directory_diagnostics["bundle_directory_is_readable"] is True
    )
    assert (
        collected_bundle_directory_diagnostics["list_directory_error"]
        == "forced directory listing error"
    )


def test_loader_fails_when_optional_cache_file_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify loader raises explicit failure when cache file write fails.

    Optional cache files are created lazily at startup, so write failures must
    raise clear diagnostics instead of creating ambiguous startup behaviour.
    Parameters: tmp_path and monkeypatch are provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        bundle_directory_path=tmp_path / "cache-write-error-bundle"
    )
    interaction_cache_file_path = valid_bundle_directory_path / "interaction_cache.json"
    interaction_cache_file_path.unlink()
    graph_sage_bundle_loader = GraphSageBundleLoader()
    original_write_text = Path.write_text

    # This monkeypatch targets only interaction cache creation so the test
    # reaches the optional cache write branch and captures its failure details.
    def _raise_cache_write_error(
        self: Path, data: str, encoding: str | None = None
    ) -> int:
        if self.name == "interaction_cache.json":
            raise PermissionError("forced optional cache write failure")
        return original_write_text(self, data, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", _raise_cache_write_error)

    with pytest.raises(GraphSageBundleValidationError) as captured_error:
        graph_sage_bundle_loader.load_and_validate_bundle(
            bundle_directory_path=valid_bundle_directory_path
        )

    assert captured_error.value.error_code == "optional_cache_initialisation_failed"
    assert (
        captured_error.value.details["optional_cache_file_name"]
        == "interaction_cache.json"
    )
    assert (
        "forced optional cache write failure"
        in captured_error.value.details["cache_initialisation_error"]
    )


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
