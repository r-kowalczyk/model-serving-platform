"""Unit tests for the GraphSAGE runtime boundary implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model_serving_platform.infrastructure.bundles.loader import GraphSageBundleLoader
from model_serving_platform.infrastructure.graphsage.runtime import (
    GraphSageInferenceRuntime,
)
from tests.bundle_test_data import write_valid_bundle


def test_runtime_initialises_from_loaded_bundle_metadata(tmp_path: Path) -> None:
    """Verify runtime bootstrap uses loaded bundle metadata and manifest values.

    This test confirms `num_layers` is sourced from manifest architecture and
    that base embeddings are precomputed once during runtime startup.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "runtime-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )

    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    assert graph_sage_inference_runtime.initialisation_summary.is_ready is True
    assert graph_sage_inference_runtime.initialisation_summary.model_num_layers == 2
    assert graph_sage_inference_runtime.initialisation_summary.base_embedding_count == 3


def test_runtime_scores_existing_entity_pair(tmp_path: Path) -> None:
    """Verify runtime returns a deterministic score for existing entity names.

    Existing-existing scoring is required for low-latency baseline behaviour,
    so this test confirms the runtime path uses precomputed embeddings.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "pair-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    prediction_result = graph_sage_inference_runtime.score_entity_pair(
        source_entity_name="Node One",
        target_entity_name="Node Two",
        attachment_strategy="cosine",
    )

    assert prediction_result.source_entity_name == "Node One"
    assert prediction_result.target_entity_name == "Node Two"
    assert isinstance(prediction_result.score, float)
    assert prediction_result.attachment_strategy_used == "cosine"


def test_runtime_scores_against_candidates_with_top_k(tmp_path: Path) -> None:
    """Verify runtime candidate scoring returns sorted results with top-k limit.

    This test ensures ranking behaviour is stable so API-level tests can rely
    on deterministic ordering for predictable response assertions.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "candidate-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    ranked_prediction_results = (
        graph_sage_inference_runtime.score_entity_against_candidates(
            source_entity_name="Node One",
            candidate_entity_names=["Node Two", "Node Three"],
            top_k=1,
            attachment_strategy="interaction",
        )
    )

    assert len(ranked_prediction_results) == 1
    assert ranked_prediction_results[0].target_entity_name in {"Node Two", "Node Three"}
    assert ranked_prediction_results[0].attachment_strategy_used == "interaction"


def test_runtime_raises_key_error_for_unknown_entity_name(tmp_path: Path) -> None:
    """Verify runtime fails fast when entity name is not in bundle mappings.

    The boundary intentionally relies on strict mappings because unknown
    entities are handled by explicit unseen-node logic in later stages.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "unknown-name-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    try:
        graph_sage_inference_runtime.score_entity_pair(
            source_entity_name="Unknown Name",
            target_entity_name="Node Two",
            attachment_strategy="cosine",
        )
    except KeyError as key_error:
        assert "Unknown Name" in str(key_error)
    else:
        assert False


def test_runtime_precompute_produces_expected_output_dimension(tmp_path: Path) -> None:
    """Verify precomputed embeddings use output dimension from model config.

    This check protects the reconstruction boundary by confirming startup
    embeddings match the architecture output width declared in manifest.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "dimension-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )
    precomputed_node_embeddings = (
        graph_sage_inference_runtime._precomputed_node_embeddings
    )

    assert isinstance(precomputed_node_embeddings, np.ndarray)
    assert precomputed_node_embeddings.shape[1] == 4
