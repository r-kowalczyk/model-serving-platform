"""Concrete GraphSAGE runtime boundary used by the service layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class GraphSageModelReconstructionSpec:
    """Describe model architecture values used during runtime reconstruction.

    This specification keeps architecture parsing explicit so startup can
    guarantee `num_layers` comes from manifest data and not code defaults.
    Parameters: every field is copied from validated manifest architecture.
    """

    input_dimension: int
    hidden_dimension: int
    output_dimension: int
    dropout: float
    decoder_type: str
    decoder_hidden_dimension: int
    num_layers: int


class GraphSageInferenceRuntime(InferenceRuntime):
    """Provide the GraphSAGE-specific runtime used by service orchestration.

    This runtime keeps GraphSAGE implementation details outside route logic.
    Stage 3 focuses on reconstruction boundary and startup embedding preload.
    Parameters: use `from_loaded_bundle_metadata` for deterministic startup.
    """

    def __init__(
        self,
        model_reconstruction_spec: GraphSageModelReconstructionSpec,
        node_name_to_node_id: dict[str, str],
        node_id_to_index: dict[str, int],
        precomputed_node_embeddings: NDArray[np.float64],
    ) -> None:
        """Initialise runtime with reconstructed model spec and embeddings.

        The service depends on this object for scoring operations, so startup
        precomputes embeddings once and stores them for repeated reuse.
        Parameters: values are produced by bundle-based startup initialisation.
        """

        self._model_reconstruction_spec = model_reconstruction_spec
        self._node_name_to_node_id = node_name_to_node_id
        self._node_id_to_index = node_id_to_index
        self._precomputed_node_embeddings = precomputed_node_embeddings
        self.initialisation_summary = RuntimeInitialisationSummary(
            runtime_name="graphsage",
            model_num_layers=model_reconstruction_spec.num_layers,
            base_embedding_count=int(precomputed_node_embeddings.shape[0]),
            is_ready=True,
            readiness_reason="graphsage runtime initialised",
        )

    def has_entity_name(self, entity_name: str) -> bool:
        """Return whether this entity name exists in bundle node mappings.

        Service-level request rules use this check to decide whether requests
        are existing-existing, one-unseen, or two-unseen endpoint scenarios.
        Parameters: entity_name is matched against bundle display name mapping.
        """

        return entity_name in self._node_name_to_node_id

    def get_known_entity_names(self) -> list[str]:
        """Return deterministic sorted known entity names from the bundle.

        Candidate ranking endpoints use this list as the known target pool and
        sorting keeps deterministic behaviour for stable API-level tests.
        Parameters: none.
        """

        return sorted(self._node_name_to_node_id.keys())

    @classmethod
    def from_loaded_bundle_metadata(
        cls, loaded_bundle_metadata: LoadedGraphSageBundleMetadata
    ) -> "GraphSageInferenceRuntime":
        """Construct runtime from bundle metadata and on-disk bundle artefacts.

        A classmethod keeps startup wiring concise while preserving a clear
        boundary where GraphSAGE reconstruction and preload behaviour live.
        Parameters: loaded_bundle_metadata references validated bundle files.
        """

        manifest_payload = json.loads(
            Path(loaded_bundle_metadata.manifest_path).read_text(encoding="utf-8")
        )
        node_features_array = np.load(loaded_bundle_metadata.node_features_path)
        model_reconstruction_spec = _build_model_reconstruction_spec(
            model_architecture=loaded_bundle_metadata.model_architecture
        )

        # A deterministic projection is used here to keep Stage 3 runtime simple while proving preload wiring.
        precomputed_node_embeddings = _precompute_base_node_embeddings(
            node_features_array=node_features_array,
            output_dimension=model_reconstruction_spec.output_dimension,
            attachment_seed=loaded_bundle_metadata.attachment_seed,
        )

        return cls(
            model_reconstruction_spec=model_reconstruction_spec,
            node_name_to_node_id=dict(manifest_payload["node_name_to_id"]),
            node_id_to_index=dict(manifest_payload["node_id_to_index"]),
            precomputed_node_embeddings=precomputed_node_embeddings,
        )

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
        source_entity_description: str | None = None,
        target_entity_description: str | None = None,
    ) -> RuntimePredictionResult:
        """Score one existing entity pair using precomputed node embeddings.

        Stage 3 keeps this path intentionally strict because existing-existing
        scoring is the lowest-latency baseline needed for later API stages.
        Parameters: source and target names identify one prediction request.
        """

        source_entity_is_known = self.has_entity_name(entity_name=source_entity_name)
        target_entity_is_known = self.has_entity_name(entity_name=target_entity_name)
        if not source_entity_is_known and not target_entity_is_known:
            raise ValueError("Two unseen endpoints are not supported.")

        source_vector = self._resolve_entity_embedding(
            entity_name=source_entity_name,
            entity_description=source_entity_description,
        )
        target_vector = self._resolve_entity_embedding(
            entity_name=target_entity_name,
            entity_description=target_entity_description,
        )
        score_value = _cosine_similarity(
            source_vector=source_vector, target_vector=target_vector
        )
        if source_entity_is_known and target_entity_is_known:
            enrichment_status = "not_required"
        else:
            enrichment_status = "degraded_local_text"
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
        """Score one entity against candidate entities using preload vectors.

        This method remains deterministic and avoids request-time graph work
        so service tests can validate boundary behaviour with minimal runtime.
        Parameters: entity names and top_k control ranked output entries.
        """

        source_vector = self._resolve_entity_embedding(
            entity_name=source_entity_name,
            entity_description=source_entity_description,
        )
        scored_candidate_predictions: list[RuntimePredictionResult] = []
        source_entity_is_known = self.has_entity_name(entity_name=source_entity_name)
        for candidate_entity_name in candidate_entity_names:
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
                        else "degraded_local_text"
                    ),
                )
            )
        scored_candidate_predictions.sort(
            key=lambda prediction_result: prediction_result.score, reverse=True
        )
        return scored_candidate_predictions[:top_k]

    def _resolve_existing_entity_embedding(
        self, entity_name: str
    ) -> NDArray[np.float64]:
        """Resolve one entity name into a precomputed embedding vector.

        Name resolution is explicit here because API-facing entity names differ
        from node identifiers used by graph arrays and model computations.
        Parameters: entity_name must map to a known node in the bundle.
        """

        resolved_node_id = self._node_name_to_node_id[entity_name]
        resolved_node_index = self._node_id_to_index[resolved_node_id]
        resolved_embedding_vector = self._precomputed_node_embeddings[
            resolved_node_index
        ]
        return np.asarray(resolved_embedding_vector, dtype=np.float64)

    def _resolve_entity_embedding(
        self, entity_name: str, entity_description: str | None
    ) -> NDArray[np.float64]:
        """Resolve an embedding for known and unseen entity request paths.

        This function uses precomputed vectors for known nodes and a local
        deterministic text projection for one unseen endpoint in Stage 4.
        Parameters: entity_name and description define unseen text embedding.
        """

        if self.has_entity_name(entity_name=entity_name):
            return self._resolve_existing_entity_embedding(entity_name=entity_name)
        return _build_unseen_entity_embedding(
            entity_text=entity_description or entity_name,
            output_dimension=self._model_reconstruction_spec.output_dimension,
            attachment_seed=self._model_reconstruction_spec.num_layers,
        )


def _build_model_reconstruction_spec(
    model_architecture: dict[str, int | float | str],
) -> GraphSageModelReconstructionSpec:
    """Build a strongly-typed reconstruction specification from manifest data.

    This conversion isolates parsing concerns so runtime classes receive a
    stable shape and can enforce exact layer count from startup metadata.
    Parameters: model_architecture comes from validated bundle metadata.
    """

    return GraphSageModelReconstructionSpec(
        input_dimension=int(model_architecture["input_dim"]),
        hidden_dimension=int(model_architecture["hidden_dim"]),
        output_dimension=int(model_architecture["output_dim"]),
        dropout=float(model_architecture["dropout"]),
        decoder_type=str(model_architecture["decoder_type"]),
        decoder_hidden_dimension=int(model_architecture["decoder_hidden_dim"]),
        num_layers=int(model_architecture["num_layers"]),
    )


def _precompute_base_node_embeddings(
    node_features_array: NDArray[np.float64] | NDArray[np.float32],
    output_dimension: int,
    attachment_seed: int,
) -> NDArray[np.float64]:
    """Precompute base node embeddings once during startup initialisation.

    A deterministic projection is used for Stage 3 so precompute wiring can be
    exercised without coupling tests to heavyweight training-side artefacts.
    Parameters: node_features_array and output_dimension define output matrix.
    """

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
    """Compute cosine similarity score for two embedding vectors.

    Cosine similarity is used because Stage 3 requires a deterministic scoring
    primitive for existing node comparisons before API prediction endpoints.
    Parameters: vectors are assumed non-empty and numeric.
    """

    numerator = float(np.dot(source_vector, target_vector))
    denominator = float(np.linalg.norm(source_vector) * np.linalg.norm(target_vector))
    return numerator / denominator


def _build_unseen_entity_embedding(
    entity_text: str, output_dimension: int, attachment_seed: int
) -> NDArray[np.float64]:
    """Build a deterministic embedding vector for one unseen entity text.

    Stage 4 uses this local deterministic vector because external enrichment
    clients are introduced later while API and runtime boundaries stabilise.
    Parameters: text content and dimensions define the embedding output.
    """

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
