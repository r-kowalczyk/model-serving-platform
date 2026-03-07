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
    assert prediction_result.enrichment_status == "not_required"


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
    assert ranked_prediction_results[0].enrichment_status == "not_required"


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

    prediction_result = graph_sage_inference_runtime.score_entity_pair(
        source_entity_name="Unknown Name",
        target_entity_name="Node Two",
        attachment_strategy="cosine",
    )

    assert prediction_result.enrichment_status == "degraded_local_text"


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


def test_runtime_has_entity_name_and_lists_known_names(tmp_path: Path) -> None:
    """Verify runtime exposes entity existence checks and known name listing.

    The service layer depends on these methods for endpoint request rule
    enforcement and candidate pool generation in Stage 4 API handlers.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "known-names-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    assert graph_sage_inference_runtime.has_entity_name(entity_name="Node One") is True
    assert (
        graph_sage_inference_runtime.has_entity_name(entity_name="Missing Node")
        is False
    )
    assert graph_sage_inference_runtime.get_known_entity_names() == [
        "Node One",
        "Node Three",
        "Node Two",
    ]


def test_runtime_rejects_two_unseen_pair_request(tmp_path: Path) -> None:
    """Verify runtime rejects pair scoring requests with two unseen endpoints.

    The Stage 4 runtime keeps this explicit rule to align with v1 contract
    constraints for pair prediction behaviour at request time.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "two-unseen-bundle")
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
            source_entity_name="Unknown Name A",
            target_entity_name="Unknown Name B",
            attachment_strategy="cosine",
        )
    except ValueError as value_error:
        assert "Two unseen endpoints" in str(value_error)
    else:
        assert False


def test_runtime_marks_unseen_source_rank_requests_as_degraded(tmp_path: Path) -> None:
    """Verify unseen source ranking requests report degraded enrichment status.

    This behaviour signals that an unseen source entity was projected from
    local request text rather than known graph node embeddings.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "unseen-rank-bundle")
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
            source_entity_name="Unknown Source",
            candidate_entity_names=["Node One", "Node Two"],
            top_k=2,
            attachment_strategy="interaction",
            source_entity_description="Unknown source description",
        )
    )

    assert ranked_prediction_results[0].enrichment_status == "degraded_local_text"
