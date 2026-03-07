"""End-to-end smoke coverage for application happy path."""

from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app


def test_smoke_boot_and_exercise_happy_path(
    configured_bundle_environment: None,
) -> None:
    """Boot app and execute a representative end-to-end happy path.

    This smoke test provides CI-friendly confidence that startup wiring,
    readiness, metadata, inference, and metrics exposure all work together.
    Parameters: configured_bundle_environment points startup to valid bundle.
    """

    test_client = TestClient(create_app())

    health_response = test_client.get("/healthz")
    ready_response = test_client.get("/readyz")
    metadata_response = test_client.get("/v1/metadata")
    pair_prediction_response = test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Node One",
            "entity_b_name": "Node Two",
        },
    )
    ranked_prediction_response = test_client.post(
        "/v1/predict-links",
        json={
            "entity_name": "Node One",
            "top_k": 2,
        },
    )
    metrics_response = test_client.get("/metrics")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert metadata_response.status_code == 200
    assert pair_prediction_response.status_code == 200
    assert ranked_prediction_response.status_code == 200
    assert metrics_response.status_code == 200
    assert metadata_response.json()["model_backend"] == "graphsage"
    assert pair_prediction_response.json()["attachment_strategy_used"] in {
        "interaction",
        "cosine",
    }
    assert len(ranked_prediction_response.json()["predictions"]) == 2
    assert "model_serving_http_request_total" in metrics_response.text
