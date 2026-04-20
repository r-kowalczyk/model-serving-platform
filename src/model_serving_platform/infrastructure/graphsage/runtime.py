"""GraphSAGE implementation of the inference runtime used at prediction time.

`PredictionService` talks only to the `InferenceRuntime` protocol. This file
provides `GraphSageInferenceRuntime`, the real implementation for this project.

Startup factory `from_loaded_bundle_metadata` reads `manifest.json` again from
disk, loads `node_features.npy`, builds a small typed spec from manifest model
fields, and precomputes one embedding vector per graph node. That precompute step
uses a fixed random projection (seeded from the bundle) so tests stay stable; it
is not the full trained GraphSAGE forward pass, but it gives consistent vectors
for cosine scoring.

At request time the runtime resolves each entity name to a vector: known names
use the precomputed row; unseen names use caller text, external description
lookup, or a degraded hash of the name, depending on mode and inputs. Pair and
ranking methods then use cosine similarity between vectors. Interaction strategy
can narrow ranking candidates using external partner lists when the source is
unseen.

Module-level helpers at the bottom of the file implement projection, cosine,
unseen text embeddings, candidate filtering, and enrichment status labelling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from model_serving_platform.application.inference_runtime import (
    InferenceRuntime,
    RuntimeInitialisationSummary,
    RuntimePredictionResult,
)
from model_serving_platform.infrastructure.bundles.loader import (
    LoadedGraphSageBundleMetadata,
)
from model_serving_platform.infrastructure.clients.enrichment import (
    ExternalEnrichmentClient,
    InteractionPartnerLookupResult,
    NoopExternalEnrichmentClient,
)

graph_sage_runtime_logger = logging.getLogger(
    "model_serving_platform.infrastructure.graphsage.runtime"
)


@dataclass(frozen=True, slots=True)
class GraphSageModelReconstructionSpec:
    """Copy of manifest architecture numbers kept on the runtime instance.

    Used for output dimension when building unseen embeddings and for metadata.
    """

    input_dimension: int
    hidden_dimension: int
    output_dimension: int
    dropout: float
    decoder_type: str
    decoder_hidden_dimension: int
    num_layers: int


class GraphSageInferenceRuntime(InferenceRuntime):
    """Loads bundle data once, then answers `InferenceRuntime` scoring calls.

    Prefer constructing via `from_loaded_bundle_metadata` in production.
    """

    def __init__(
        self,
        model_reconstruction_spec: GraphSageModelReconstructionSpec,
        node_name_to_node_id: dict[str, str],
        node_id_to_index: dict[str, int],
        precomputed_node_embeddings: NDArray[np.float64],
        external_enrichment_client: ExternalEnrichmentClient,
        restricted_network_mode: bool,
    ) -> None:
        """Store maps, embedding matrix, enrichment client, mode, and readiness summary."""

        self._model_reconstruction_spec = model_reconstruction_spec
        self._node_name_to_node_id = node_name_to_node_id
        self._node_id_to_index = node_id_to_index
        self._precomputed_node_embeddings = precomputed_node_embeddings
        self._external_enrichment_client = external_enrichment_client
        self._restricted_network_mode = restricted_network_mode
        self.initialisation_summary = RuntimeInitialisationSummary(
            runtime_name="graphsage",
            model_num_layers=model_reconstruction_spec.num_layers,
            base_embedding_count=int(precomputed_node_embeddings.shape[0]),
            is_ready=True,
            readiness_reason="graphsage runtime initialised",
        )

    def has_entity_name(self, entity_name: str) -> bool:
        """True if `entity_name` is a key in the bundle `node_name_to_id` map."""

        return entity_name in self._node_name_to_node_id

    def get_known_entity_names(self) -> list[str]:
        """All bundle entity display names, sorted for stable ordering."""

        return sorted(self._node_name_to_node_id.keys())

    @classmethod
    def from_loaded_bundle_metadata(
        cls,
        loaded_bundle_metadata: LoadedGraphSageBundleMetadata,
        external_enrichment_client: ExternalEnrichmentClient | None = None,
        restricted_network_mode: bool = False,
    ) -> "GraphSageInferenceRuntime":
        """Read manifest and node features from paths in metadata, precompute embeddings, return instance."""

        runtime_initialisation_start_timestamp = perf_counter()
        graph_sage_runtime_logger.info(
            "runtime_bundle_materialisation_started",
            extra={
                "manifest_path": loaded_bundle_metadata.manifest_path,
                "node_features_path": loaded_bundle_metadata.node_features_path,
                "bundle_path": loaded_bundle_metadata.bundle_path,
            },
        )
        manifest_loading_start_timestamp = perf_counter()
        # Full manifest JSON is read again here so node id maps are available alongside arrays.
        manifest_payload = json.loads(
            Path(loaded_bundle_metadata.manifest_path).read_text(encoding="utf-8")
        )
        manifest_loading_elapsed_milliseconds = int(
            (perf_counter() - manifest_loading_start_timestamp) * 1000
        )
        graph_sage_runtime_logger.info(
            "runtime_manifest_loaded",
            extra={
                "manifest_path": loaded_bundle_metadata.manifest_path,
                "manifest_loading_elapsed_milliseconds": manifest_loading_elapsed_milliseconds,
            },
        )
        node_feature_loading_start_timestamp = perf_counter()
        # Feature matrix shape was validated at bundle load; this load feeds precompute only.
        node_features_array = np.load(loaded_bundle_metadata.node_features_path)
        node_feature_loading_elapsed_milliseconds = int(
            (perf_counter() - node_feature_loading_start_timestamp) * 1000
        )
        graph_sage_runtime_logger.info(
            "runtime_node_features_loaded",
            extra={
                "node_features_path": loaded_bundle_metadata.node_features_path,
                "node_feature_shape": list(node_features_array.shape),
                "node_feature_loading_elapsed_milliseconds": node_feature_loading_elapsed_milliseconds,
            },
        )
        model_reconstruction_spec = _build_model_reconstruction_spec(
            model_architecture=loaded_bundle_metadata.model_architecture
        )

        # Linear projection with a seeded RNG: fixed vectors per node for cosine scoring in this codebase.
        embedding_precomputation_start_timestamp = perf_counter()
        precomputed_node_embeddings = _precompute_base_node_embeddings(
            node_features_array=node_features_array,
            output_dimension=model_reconstruction_spec.output_dimension,
            attachment_seed=loaded_bundle_metadata.attachment_seed,
        )
        embedding_precomputation_elapsed_milliseconds = int(
            (perf_counter() - embedding_precomputation_start_timestamp) * 1000
        )
        graph_sage_runtime_logger.info(
            "runtime_base_embeddings_precomputed",
            extra={
                "base_embedding_shape": list(precomputed_node_embeddings.shape),
                "embedding_precomputation_elapsed_milliseconds": embedding_precomputation_elapsed_milliseconds,
            },
        )
        runtime_initialisation_elapsed_milliseconds = int(
            (perf_counter() - runtime_initialisation_start_timestamp) * 1000
        )
        graph_sage_runtime_logger.info(
            "runtime_bundle_materialisation_finished",
            extra={
                "runtime_initialisation_elapsed_milliseconds": runtime_initialisation_elapsed_milliseconds,
                "manifest_path": loaded_bundle_metadata.manifest_path,
                "node_features_path": loaded_bundle_metadata.node_features_path,
            },
        )

        # Use noop enrichment when caller passes None so the runtime always has a concrete client.
        return cls(
            model_reconstruction_spec=model_reconstruction_spec,
            node_name_to_node_id=dict(manifest_payload["node_name_to_id"]),
            node_id_to_index=dict(manifest_payload["node_id_to_index"]),
            precomputed_node_embeddings=precomputed_node_embeddings,
            external_enrichment_client=external_enrichment_client
            or NoopExternalEnrichmentClient(),
            restricted_network_mode=restricted_network_mode,
        )

    def supports_interaction_strategy(self) -> bool:
        """Forwarded from the configured external enrichment client."""

        return self._external_enrichment_client.supports_interaction_strategy()

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
        source_entity_description: str | None = None,
        target_entity_description: str | None = None,
    ) -> RuntimePredictionResult:
        """Resolve two embeddings, cosine similarity, combine enrichment labels."""

        source_entity_is_known = self.has_entity_name(entity_name=source_entity_name)
        target_entity_is_known = self.has_entity_name(entity_name=target_entity_name)
        # Runtime guard; service layer also rejects this case for API contracts.
        if not source_entity_is_known and not target_entity_is_known:
            raise ValueError("Two unseen endpoints are not supported.")

        source_vector, source_enrichment_status = self._resolve_entity_embedding(
            entity_name=source_entity_name,
            entity_description=source_entity_description,
        )
        target_vector, target_enrichment_status = self._resolve_entity_embedding(
            entity_name=target_entity_name,
            entity_description=target_entity_description,
        )
        score_value = _cosine_similarity(
            source_vector=source_vector, target_vector=target_vector
        )
        enrichment_status = self._resolve_pair_enrichment_status(
            source_enrichment_status=source_enrichment_status,
            target_enrichment_status=target_enrichment_status,
        )
        return RuntimePredictionResult(
            source_entity_name=source_entity_name,
            target_entity_name=target_entity_name,
            score=score_value,
            attachment_strategy_used=attachment_strategy,
            enrichment_status=enrichment_status,
        )

    def score_entity_against_candidates(
        self,
        source_entity_name: str,
        candidate_entity_names: list[str],
        top_k: int,
        attachment_strategy: str,
        source_entity_description: str | None = None,
    ) -> list[RuntimePredictionResult]:
        """Score source against each candidate, sort by score descending, truncate to `top_k`."""

        source_vector, source_enrichment_status = self._resolve_entity_embedding(
            entity_name=source_entity_name,
            entity_description=source_entity_description,
        )
        scored_candidate_predictions: list[RuntimePredictionResult] = []
        source_entity_is_known = self.has_entity_name(entity_name=source_entity_name)
        candidate_entity_names_for_scoring = candidate_entity_names
        interaction_lookup_result: InteractionPartnerLookupResult | None = None
        # Unseen source plus interaction mode: optionally restrict candidates to reported partners.
        if not source_entity_is_known and attachment_strategy == "interaction":
            interaction_lookup_result = (
                self._external_enrichment_client.lookup_interaction_partners(
                    entity_name=source_entity_name
                )
            )
            candidate_entity_names_for_scoring = (
                _filter_candidate_names_from_interactions(
                    candidate_entity_names=candidate_entity_names,
                    interaction_lookup_result=interaction_lookup_result,
                )
            )
        # Candidates are assumed known bundle nodes; unseen target handling is pair-specific.
        for candidate_entity_name in candidate_entity_names_for_scoring:
            candidate_vector = self._resolve_existing_entity_embedding(
                entity_name=candidate_entity_name
            )
            score_value = _cosine_similarity(
                source_vector=source_vector,
                target_vector=candidate_vector,
            )
            scored_candidate_predictions.append(
                RuntimePredictionResult(
                    source_entity_name=source_entity_name,
                    target_entity_name=candidate_entity_name,
                    score=score_value,
                    attachment_strategy_used=attachment_strategy,
                    enrichment_status=(
                        "not_required"
                        if source_entity_is_known
                        else _resolve_unseen_enrichment_status(
                            base_enrichment_status=source_enrichment_status,
                            attachment_strategy=attachment_strategy,
                            interaction_lookup_result=interaction_lookup_result
                            if attachment_strategy == "interaction"
                            else None,
                        )
                    ),
                )
            )
        scored_candidate_predictions.sort(
            key=lambda prediction_result: prediction_result.score, reverse=True
        )
        # Service passes `top_k` already bounded; slice enforces ranking window here too.
        return scored_candidate_predictions[:top_k]

    def _resolve_existing_entity_embedding(
        self, entity_name: str
    ) -> NDArray[np.float64]:
        """Map display name to internal node id, then to row index, then return that embedding row."""

        resolved_node_id = self._node_name_to_node_id[entity_name]
        resolved_node_index = self._node_id_to_index[resolved_node_id]
        resolved_embedding_vector = self._precomputed_node_embeddings[
            resolved_node_index
        ]
        return np.asarray(resolved_embedding_vector, dtype=np.float64)

    def _resolve_entity_embedding(
        self, entity_name: str, entity_description: str | None
    ) -> tuple[NDArray[np.float64], str]:
        """Known node: precomputed row. Unseen: description, HTTP lookup, or name fallback."""

        if self.has_entity_name(entity_name=entity_name):
            return self._resolve_existing_entity_embedding(
                entity_name=entity_name
            ), "not_required"
        if entity_description is not None:
            return (
                _build_unseen_entity_embedding(
                    entity_text=entity_description,
                    output_dimension=self._model_reconstruction_spec.output_dimension,
                    attachment_seed=self._model_reconstruction_spec.num_layers,
                ),
                "caller_provided_description",
            )
        if self._restricted_network_mode:
            raise ValueError(
                "Restricted network mode requires caller-provided descriptions for unseen entities."
            )
        entity_description_lookup_result = (
            self._external_enrichment_client.lookup_entity_description(
                entity_name=entity_name
            )
        )
        if entity_description_lookup_result.description is not None:
            return (
                _build_unseen_entity_embedding(
                    entity_text=entity_description_lookup_result.description,
                    output_dimension=self._model_reconstruction_spec.output_dimension,
                    attachment_seed=self._model_reconstruction_spec.num_layers,
                ),
                "external_lookup",
            )
        return (
            _build_unseen_entity_embedding(
                entity_text=entity_name,
                output_dimension=self._model_reconstruction_spec.output_dimension,
                attachment_seed=self._model_reconstruction_spec.num_layers,
            ),
            "degraded_local_text",
        )

    def _resolve_pair_enrichment_status(
        self,
        source_enrichment_status: str,
        target_enrichment_status: str,
    ) -> str:
        """Pick one summary string from per-endpoint enrichment statuses for pair responses."""

        if (
            source_enrichment_status == "not_required"
            and target_enrichment_status == "not_required"
        ):
            return "not_required"
        if (
            source_enrichment_status == "external_lookup"
            or target_enrichment_status == "external_lookup"
        ):
            return "external_lookup"
        if (
            source_enrichment_status == "caller_provided_description"
            or target_enrichment_status == "caller_provided_description"
        ):
            return "caller_provided_description"
        return "degraded_local_text"


def _build_model_reconstruction_spec(
    model_architecture: dict[str, int | float | str],
) -> GraphSageModelReconstructionSpec:
    """Copy manifest `model` dict keys into a dataclass with int or float casts."""

    return GraphSageModelReconstructionSpec(
        input_dimension=int(model_architecture["input_dim"]),
        hidden_dimension=int(model_architecture["hidden_dim"]),
        output_dimension=int(model_architecture["output_dim"]),
        dropout=float(model_architecture["dropout"]),
        decoder_type=str(model_architecture["decoder_type"]),
        decoder_hidden_dimension=int(model_architecture["decoder_hidden_dim"]),
        num_layers=int(model_architecture["num_layers"]),
    )


def _filter_candidate_names_from_interactions(
    candidate_entity_names: list[str],
    interaction_lookup_result: InteractionPartnerLookupResult,
) -> list[str]:
    """If partner lookup succeeded, keep only candidates that appear in partner list; else unchanged."""

    if interaction_lookup_result.outcome != "success":
        return candidate_entity_names
    interaction_partner_name_set = set(interaction_lookup_result.partner_entity_names)
    filtered_candidate_entity_names = [
        candidate_entity_name
        for candidate_entity_name in candidate_entity_names
        if candidate_entity_name in interaction_partner_name_set
    ]
    if len(filtered_candidate_entity_names) == 0:
        return candidate_entity_names
    return filtered_candidate_entity_names


def _resolve_unseen_enrichment_status(
    base_enrichment_status: str,
    attachment_strategy: str,
    interaction_lookup_result: InteractionPartnerLookupResult | None,
) -> str:
    """Refine status when interaction mode was used and base path was degraded text."""

    if base_enrichment_status != "degraded_local_text":
        return base_enrichment_status
    if attachment_strategy != "interaction":
        return base_enrichment_status
    if interaction_lookup_result is None:
        return base_enrichment_status
    if interaction_lookup_result.outcome == "success":
        return "interaction_lookup"
    return "interaction_lookup_failed_fallback"


def _precompute_base_node_embeddings(
    node_features_array: NDArray[np.float64] | NDArray[np.float32],
    output_dimension: int,
    attachment_seed: int,
) -> NDArray[np.float64]:
    """Matrix multiply node features by a random Gaussian matrix fixed by `attachment_seed`."""

    random_number_generator = np.random.default_rng(seed=attachment_seed)
    projection_matrix = random_number_generator.normal(
        loc=0.0,
        scale=0.1,
        size=(node_features_array.shape[1], output_dimension),
    )
    precomputed_node_embeddings = node_features_array @ projection_matrix
    return np.asarray(precomputed_node_embeddings, dtype=np.float64)


def _cosine_similarity(
    source_vector: NDArray[np.float64], target_vector: NDArray[np.float64]
) -> float:
    """Dot product divided by product of L2 norms (standard cosine similarity)."""

    numerator = float(np.dot(source_vector, target_vector))
    denominator = float(np.linalg.norm(source_vector) * np.linalg.norm(target_vector))
    return numerator / denominator


def _build_unseen_entity_embedding(
    entity_text: str, output_dimension: int, attachment_seed: int
) -> NDArray[np.float64]:
    """Sample one Gaussian vector whose RNG seed depends on `attachment_seed` and character codes."""

    deterministic_seed = attachment_seed + sum(
        ord(character) for character in entity_text
    )
    random_number_generator = np.random.default_rng(seed=deterministic_seed)
    unseen_entity_embedding = random_number_generator.normal(
        loc=0.0,
        scale=0.1,
        size=(output_dimension,),
    )
    return np.asarray(unseen_entity_embedding, dtype=np.float64)
