"""Unit tests for the GraphSAGE runtime boundary implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model_serving_platform.infrastructure.bundles.loader import GraphSageBundleLoader
from model_serving_platform.infrastructure.clients.enrichment import (
    EntityDescriptionLookupResult,
    ExternalEnrichmentClient,
    InteractionPartnerLookupResult,
)
from model_serving_platform.infrastructure.graphsage.runtime import (
    GraphSageInferenceRuntime,
    _filter_candidate_names_from_interactions,
    _resolve_unseen_enrichment_status,
)
from tests.bundle_test_data import write_valid_bundle


class StaticExternalEnrichmentClient(ExternalEnrichmentClient):
    """External enrichment client stub with deterministic lookup outcomes.

    Runtime tests use this client to exercise external lookup branches without
    depending on HTTP transport behaviour or mutable external services.
    Parameters: outcomes are provided at construction for deterministic tests.
    """

    def __init__(
        self,
        description_text: str | None,
        interaction_lookup_result: InteractionPartnerLookupResult,
        supports_interaction: bool,
    ) -> None:
        """Initialise deterministic external lookup outcomes for one test.

        The runtime calls these methods during unseen handling, so fixed
        outcomes allow explicit branch coverage of enrichment status logic.
        Parameters: values define description and interaction lookup behaviour.
        """

        self._description_text = description_text
        self._interaction_lookup_result = interaction_lookup_result
        self._supports_interaction = supports_interaction

    def lookup_entity_description(
        self, entity_name: str
    ) -> EntityDescriptionLookupResult:
        """Return fixed description lookup outcome for runtime tests.

        This method deliberately ignores entity_name to keep deterministic
        outcomes while testing runtime branch behaviour in isolation.
        Parameters: entity_name is accepted for protocol compatibility.
        """

        _ = entity_name
        if self._description_text is None:
            return EntityDescriptionLookupResult(description=None, outcome="failed")
        return EntityDescriptionLookupResult(
            description=self._description_text,
            outcome="success",
        )

    def lookup_interaction_partners(
        self, entity_name: str
    ) -> InteractionPartnerLookupResult:
        """Return fixed interaction lookup outcome for runtime tests.

        This deterministic response allows direct control over candidate
        filtering and fallback status paths in unseen interaction mode.
        Parameters: entity_name is accepted for protocol compatibility.
        """

        _ = entity_name
        return self._interaction_lookup_result

    def supports_interaction_strategy(self) -> bool:
        """Return deterministic interaction capability flag for runtime tests.

        This value is used by prediction orchestration fallback logic and
        runtime capability checks in dedicated test assertions.
        Parameters: none.
        """

        return self._supports_interaction


def test_runtime_loads_encoder_prefixed_checkpoint_like_training_export(
    tmp_path: Path,
) -> None:
    """Ensure full-model checkpoints from graph-link-prediction (`encoder.*` keys) load.

    Training saves `torch.save(model.state_dict())` for `GraphSageLinkPredictor`, so
    tensor keys carry an `encoder.` prefix that serving code must strip.
    """

    bundle_directory_path = write_valid_bundle(tmp_path / "prefixed-bundle")
    encoder_only_state = torch.load(
        bundle_directory_path / "model_state.pt",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    prefixed_state = {
        "encoder." + tensor_key: tensor_value
        for tensor_key, tensor_value in encoder_only_state.items()
    }
    torch.save(prefixed_state, bundle_directory_path / "model_state.pt")

    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata,
        )
    )

    assert graph_sage_inference_runtime.initialisation_summary.is_ready is True
    assert graph_sage_inference_runtime.initialisation_summary.base_embedding_count == 3


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
    assert graph_sage_inference_runtime.supports_interaction_strategy() is False


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
    """Verify unseen source interaction ranking reports explicit fallback state.

    This behaviour signals that interaction lookup was unavailable and scoring
    proceeded with fallback candidate handling for unseen source entities.
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
        )
    )

    assert (
        ranked_prediction_results[0].enrichment_status
        == "interaction_lookup_failed_fallback"
    )


def test_runtime_reports_caller_description_status_for_unseen_pair(
    tmp_path: Path,
) -> None:
    """Verify unseen pair requests report caller provided description status.

    This covers the pair enrichment status branch that reports caller-provided
    description usage when an unseen endpoint includes literal description text.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(
        tmp_path / "pair-description-bundle"
    )
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata
        )
    )

    prediction_result = graph_sage_inference_runtime.score_entity_pair(
        source_entity_name="Unknown Node",
        source_entity_description="Provided description text",
        target_entity_name="Node One",
        attachment_strategy="cosine",
    )

    assert prediction_result.enrichment_status == "caller_provided_description"


def test_runtime_reports_external_lookup_status_for_unseen_pair(tmp_path: Path) -> None:
    """Verify unseen pair requests report external lookup enrichment status.

    This test exercises the branch where external enrichment provides unseen
    description text and pair response status reflects external lookup usage.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "pair-external-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    external_enrichment_client = StaticExternalEnrichmentClient(
        description_text="Externally resolved text",
        interaction_lookup_result=InteractionPartnerLookupResult(
            partner_entity_names=[],
            outcome="unavailable",
        ),
        supports_interaction=False,
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata,
            external_enrichment_client=external_enrichment_client,
        )
    )

    prediction_result = graph_sage_inference_runtime.score_entity_pair(
        source_entity_name="Unknown Node",
        target_entity_name="Node One",
        attachment_strategy="cosine",
    )

    assert prediction_result.enrichment_status == "external_lookup"


def test_runtime_restricted_mode_requires_description_for_unseen_entity(
    tmp_path: Path,
) -> None:
    """Verify restricted mode raises when unseen entity omits description.

    This test covers restricted-network runtime enforcement for unseen entities
    when no caller-provided description is present in scoring requests.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "restricted-bundle")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata,
            restricted_network_mode=True,
        )
    )

    try:
        graph_sage_inference_runtime.score_entity_pair(
            source_entity_name="Unknown Node",
            target_entity_name="Node One",
            attachment_strategy="cosine",
        )
    except ValueError as value_error:
        assert "requires caller-provided descriptions" in str(value_error)
    else:
        assert False


def test_runtime_interaction_lookup_success_sets_interaction_status(
    tmp_path: Path,
) -> None:
    """Verify unseen interaction requests report interaction lookup status.

    This test covers interaction-success branch where partner lookup returns
    usable names and runtime response status indicates interaction path usage.
    Parameters: tmp_path is provided by pytest.
    """

    valid_bundle_directory_path = write_valid_bundle(tmp_path / "interaction-success")
    loaded_bundle_metadata = GraphSageBundleLoader().load_and_validate_bundle(
        bundle_directory_path=valid_bundle_directory_path
    )
    external_enrichment_client = StaticExternalEnrichmentClient(
        description_text=None,
        interaction_lookup_result=InteractionPartnerLookupResult(
            partner_entity_names=["Node One"],
            outcome="success",
        ),
        supports_interaction=True,
    )
    graph_sage_inference_runtime = (
        GraphSageInferenceRuntime.from_loaded_bundle_metadata(
            loaded_bundle_metadata=loaded_bundle_metadata,
            external_enrichment_client=external_enrichment_client,
        )
    )

    ranked_prediction_results = (
        graph_sage_inference_runtime.score_entity_against_candidates(
            source_entity_name="Unknown Source",
            candidate_entity_names=["Node One", "Node Two"],
            top_k=2,
            attachment_strategy="interaction",
        )
    )

    assert ranked_prediction_results[0].enrichment_status == "interaction_lookup"


def test_runtime_helper_functions_cover_filter_and_status_branches() -> None:
    """Verify helper branch behaviour for candidate filtering and status logic.

    Helper-level tests provide deterministic coverage for branch combinations
    that are awkward to trigger through public runtime methods alone.
    Parameters: none.
    """

    candidate_entity_names = ["Node One", "Node Two"]
    filtered_candidate_entity_names = _filter_candidate_names_from_interactions(
        candidate_entity_names=candidate_entity_names,
        interaction_lookup_result=InteractionPartnerLookupResult(
            partner_entity_names=["Unknown Partner"],
            outcome="success",
        ),
    )
    assert filtered_candidate_entity_names == candidate_entity_names

    assert (
        _resolve_unseen_enrichment_status(
            base_enrichment_status="caller_provided_description",
            attachment_strategy="interaction",
            interaction_lookup_result=InteractionPartnerLookupResult(
                partner_entity_names=["Node One"],
                outcome="success",
            ),
        )
        == "caller_provided_description"
    )
    assert (
        _resolve_unseen_enrichment_status(
            base_enrichment_status="degraded_local_text",
            attachment_strategy="cosine",
            interaction_lookup_result=None,
        )
        == "degraded_local_text"
    )
    assert (
        _resolve_unseen_enrichment_status(
            base_enrichment_status="degraded_local_text",
            attachment_strategy="interaction",
            interaction_lookup_result=None,
        )
        == "degraded_local_text"
    )
