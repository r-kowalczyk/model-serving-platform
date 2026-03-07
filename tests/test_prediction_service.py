"""Unit tests for prediction service orchestration behaviour."""

from model_serving_platform.application.inference_runtime import (
    RuntimeInitialisationSummary,
    RuntimePredictionResult,
)
from model_serving_platform.application.prediction_service import PredictionService
from model_serving_platform.domain.prediction import (
    PredictLinkRequest,
    PredictLinksRequest,
)
from tests.fakes.fake_inference_runtime import FakeInferenceRuntime


class EmptyCandidateRuntime(FakeInferenceRuntime):
    """Return an empty known entity list for empty prediction branch coverage.

    This runtime keeps deterministic behaviour while forcing the service path
    where ranked results are empty and response defaults are applied.
    Parameters: this class uses inherited fake runtime initialisation.
    """

    def get_known_entity_names(self) -> list[str]:
        """Return no known entities to force empty ranked predictions branch.

        This override is used only in tests to exercise response defaults when
        candidate list construction yields no entries for ranking requests.
        Parameters: none.
        """

        return []


class UnsupportedStrategyRuntime(FakeInferenceRuntime):
    """Return unsupported strategy value to cover strict strategy validation.

    This runtime is used to verify that response typing enforcement rejects
    runtime outputs that do not match the public attachment strategy contract.
    Parameters: this class uses inherited fake runtime initialisation.
    """

    initialisation_summary = RuntimeInitialisationSummary(
        runtime_name="unsupported-strategy-runtime",
        model_num_layers=2,
        base_embedding_count=0,
        is_ready=True,
        readiness_reason="ready",
    )

    def score_entity_pair(
        self,
        source_entity_name: str,
        target_entity_name: str,
        attachment_strategy: str,
        source_entity_description: str | None = None,
        target_entity_description: str | None = None,
    ) -> RuntimePredictionResult:
        """Return an invalid strategy string for service validation testing.

        The value is intentionally outside the API contract so service-level
        normalisation logic can be tested for strict contract enforcement.
        Parameters: arguments are accepted to satisfy runtime protocol shape.
        """

        return RuntimePredictionResult(
            source_entity_name=source_entity_name,
            target_entity_name=target_entity_name,
            score=0.1,
            attachment_strategy_used="unsupported",
            enrichment_status="not_required",
        )


def test_predict_links_uses_empty_results_defaults() -> None:
    """Verify predict-links response defaults are used for empty results.

    This test covers the branch where runtime returns no ranked predictions,
    ensuring attachment strategy and enrichment metadata remain deterministic.
    Parameters: none.
    """

    prediction_service = PredictionService(
        inference_runtime=EmptyCandidateRuntime(),
        service_version="0.1.0",
        bundle_version="bundle-v1",
        max_top_k=25,
        default_attachment_strategy="interaction",
    )

    response = prediction_service.predict_links(
        predict_links_request=PredictLinksRequest(entity_name="Node One", top_k=3)
    )

    assert response.predictions == []
    assert response.attachment_strategy_used == "interaction"
    assert response.enrichment_status == "not_required"


def test_predict_link_rejects_unsupported_runtime_attachment_strategy() -> None:
    """Verify service rejects unsupported runtime strategy output values.

    Strict response typing is part of Stage 4 API stability, so this test
    confirms unsupported strategy values are rejected immediately.
    Parameters: none.
    """

    prediction_service = PredictionService(
        inference_runtime=UnsupportedStrategyRuntime(),
        service_version="0.1.0",
        bundle_version="bundle-v1",
        max_top_k=25,
        default_attachment_strategy="interaction",
    )

    try:
        prediction_service.predict_link(
            predict_link_request=PredictLinkRequest(
                entity_a_name="Node One",
                entity_b_name="Node Two",
            )
        )
    except ValueError as value_error:
        assert "unsupported attachment strategy" in str(value_error).lower()
    else:
        assert False
