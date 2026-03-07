"""Deterministic fake runtime used by service-layer tests."""

from model_serving_platform.application.inference_runtime import (
    InferenceRuntime,
    RuntimeInitialisationSummary,
    RuntimePredictionResult,
)


class FakeInferenceRuntime(InferenceRuntime):
    """Provide predictable runtime behaviour for API and app wiring tests.

    This fake keeps service tests fast and stable because it avoids bundle
    mathematics while still implementing the same boundary as real runtime.
    Parameters: readiness fields allow explicit startup-state assertions.
    """

    def __init__(
        self, is_ready: bool = True, readiness_reason: str = "fake runtime ready"
    ) -> None:
        """Initialise fake runtime with deterministic summary values.

        Service-level tests use this to force ready and not-ready branches
        without changing bundle fixtures or infrastructure code paths.
        Parameters: readiness values are copied to the summary.
        """

        self.initialisation_summary = RuntimeInitialisationSummary(
            runtime_name="fake-graphsage-runtime",
            model_num_layers=2,
            base_embedding_count=3,
            is_ready=is_ready,
            readiness_reason=readiness_reason,
        )

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
    ) -> RuntimePredictionResult:
        """Return one deterministic score result for a pair request.

        Fixed values are used so service tests can assert response mapping
        without depending on internal embedding implementation details.
        Parameters: names are mirrored into the returned prediction result.
        """

        return RuntimePredictionResult(
            source_entity_name=source_entity_name,
            target_entity_name=target_entity_name,
            score=0.75,
            attachment_strategy_used=attachment_strategy,
            enrichment_status="not_required",
        )

    def score_entity_against_candidates(
        self,
        source_entity_name: str,
        candidate_entity_names: list[str],
        top_k: int,
        attachment_strategy: str,
    ) -> list[RuntimePredictionResult]:
        """Return deterministic top-k candidate scores for ranking tests.

        Scores are deterministic and descending so service tests can verify
        ordering behaviour with simple assertions and no stochastic outputs.
        Parameters: candidate names define the returned result list.
        """

        ranked_prediction_results: list[RuntimePredictionResult] = []
        for candidate_index, candidate_entity_name in enumerate(candidate_entity_names):
            ranked_prediction_results.append(
                RuntimePredictionResult(
                    source_entity_name=source_entity_name,
                    target_entity_name=candidate_entity_name,
                    score=1.0 - (candidate_index * 0.1),
                    attachment_strategy_used=attachment_strategy,
                    enrichment_status="not_required",
                )
            )
        return ranked_prediction_results[:top_k]
