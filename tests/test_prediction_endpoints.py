"""API tests for Stage 4 prediction endpoints."""

from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app
from model_serving_platform.config.settings import ServiceSettings
from tests.fakes.fake_inference_runtime import FakeInferenceRuntime


class NoInteractionStrategyRuntime(FakeInferenceRuntime):
    """Fake runtime that reports interaction strategy as unavailable.

    This runtime allows endpoint tests to verify explicit strategy fallback
    behaviour without introducing real external interaction dependencies.
    Parameters: this class uses inherited fake runtime initialisation.
    """

    def supports_interaction_strategy(self) -> bool:
        """Return false to trigger interaction strategy fallback logic.

        Service orchestration checks this capability and degrades unsupported
        interaction requests to cosine strategy for predictable behaviour.
        Parameters: none.
        """

        return False


def test_predict_link_returns_typed_response_for_existing_pair(
    configured_bundle_environment: None,
) -> None:
    """Verify predict-link returns expected fields for known endpoints.

    This test validates Stage 4 response contract for pair scoring requests
    and confirms the endpoint generates a request identifier when omitted.
    Parameters: none.
    """

    application = create_app(inference_runtime=FakeInferenceRuntime())
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Node One",
            "entity_b_name": "Node Two",
            "attachment_strategy": "interaction",
        },
    )

    response_payload = response.json()
    assert response.status_code == 200
    assert response_payload["score"] == 0.75
    assert response_payload["model_version"] == "0.1.0"
    assert response_payload["attachment_strategy_used"] == "interaction"
    assert response_payload["enrichment_status"] == "not_required"
    assert isinstance(response_payload["request_id"], str)


def test_predict_link_supports_one_unseen_endpoint(
    configured_bundle_environment: None,
) -> None:
    """Verify predict-link allows one unseen endpoint in Stage 4.

    Stage 4 permits one unseen endpoint while explicitly rejecting two unseen,
    so this test confirms the endpoint returns a valid degraded response.
    Parameters: none.
    """

    application = create_app(inference_runtime=FakeInferenceRuntime())
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Node One",
            "entity_b_name": "Unknown Node",
            "entity_b_description": "Unknown node description",
        },
    )

    response_payload = response.json()
    assert response.status_code == 200
    assert response_payload["enrichment_status"] == "degraded_local_text"
    assert response_payload["attachment_strategy_used"] == "interaction"


def test_predict_link_rejects_two_unseen_endpoints(
    configured_bundle_environment: None,
) -> None:
    """Verify predict-link rejects requests with two unseen endpoints.

    This enforces the explicit v1 constraint that pair prediction requests
    must include at least one endpoint that exists in the loaded graph.
    Parameters: none.
    """

    application = create_app(inference_runtime=FakeInferenceRuntime())
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Unknown Node A",
            "entity_b_name": "Unknown Node B",
        },
    )

    assert response.status_code == 422
    assert "two unseen endpoints" in response.json()["detail"].lower()


def test_predict_links_returns_ranked_predictions(
    configured_bundle_environment: None,
) -> None:
    """Verify predict-links returns ranked candidate predictions with top-k.

    This confirms Stage 4 multi-link response shape and metadata fields are
    stable and that top-k controls the returned ranked list size.
    Parameters: none.
    """

    application = create_app(inference_runtime=FakeInferenceRuntime())
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-links",
        json={
            "entity_name": "Node One",
            "top_k": 2,
            "attachment_strategy": "cosine",
        },
    )

    response_payload = response.json()
    assert response.status_code == 200
    assert len(response_payload["predictions"]) == 2
    assert response_payload["predictions"][0]["entity_name"] == "Node Two"
    assert response_payload["attachment_strategy_used"] == "cosine"
    assert response_payload["enrichment_status"] == "not_required"


def test_predict_links_rejects_top_k_above_configured_limit(
    configured_bundle_environment: None,
) -> None:
    """Verify predict-links rejects top-k values beyond configured max.

    This endpoint-level check protects request-time behaviour from oversized
    candidate ranking requests that violate configured service limits.
    Parameters: none.
    """

    application = create_app(inference_runtime=FakeInferenceRuntime())
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-links",
        json={
            "entity_name": "Node One",
            "top_k": 100,
        },
    )

    assert response.status_code == 422
    assert "max_top_k" in response.json()["detail"]


def test_predict_link_requires_description_in_restricted_network_mode(
    configured_bundle_environment: None,
) -> None:
    """Verify restricted mode rejects unseen endpoints without descriptions.

    This enforces explicit request-time behaviour for restricted environments
    where external description lookup is intentionally not available.
    Parameters: none.
    """

    restricted_network_service_settings = ServiceSettings(
        restricted_network_mode=True,
    )
    application = create_app(
        service_settings=restricted_network_service_settings,
        inference_runtime=FakeInferenceRuntime(),
    )
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Node One",
            "entity_b_name": "Unknown Node",
        },
    )

    assert response.status_code == 422
    assert (
        "restricted network mode requires entity_b_description"
        in response.json()["detail"].lower()
    )


def test_predict_links_falls_back_to_cosine_when_interaction_is_unavailable(
    configured_bundle_environment: None,
) -> None:
    """Verify interaction requests degrade to cosine when unavailable.

    This test proves service-level strategy resolution is explicit, producing
    fallback metadata instead of silently keeping unavailable interaction mode.
    Parameters: none.
    """

    application = create_app(inference_runtime=NoInteractionStrategyRuntime())
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-links",
        json={
            "entity_name": "Node One",
            "top_k": 2,
            "attachment_strategy": "interaction",
        },
    )

    response_payload = response.json()
    assert response.status_code == 200
    assert response_payload["attachment_strategy_used"] == "cosine"
    assert (
        response_payload["enrichment_status"]
        == "interaction_unavailable_fallback_to_cosine"
    )


def test_predict_links_requires_description_in_restricted_network_mode(
    configured_bundle_environment: None,
) -> None:
    """Verify restricted mode rejects unseen ranking source without description.

    This enforces explicit restricted-network behaviour for ranking requests
    when caller omits description for source entities not present in graph.
    Parameters: none.
    """

    restricted_network_service_settings = ServiceSettings(
        restricted_network_mode=True,
    )
    application = create_app(
        service_settings=restricted_network_service_settings,
        inference_runtime=FakeInferenceRuntime(),
    )
    test_client = TestClient(application)
    response = test_client.post(
        "/v1/predict-links",
        json={
            "entity_name": "Unknown Node",
            "top_k": 2,
        },
    )

    assert response.status_code == 422
    assert (
        "restricted network mode requires entity_description"
        in response.json()["detail"].lower()
    )
