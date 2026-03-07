"""API tests for Prometheus metrics endpoint behaviour."""

from fastapi.testclient import TestClient

from model_serving_platform.api.app import create_app
from model_serving_platform.config.settings import ServiceSettings


def test_metrics_endpoint_returns_prometheus_payload_when_enabled(
    configured_bundle_environment: None,
) -> None:
    """Verify metrics endpoint exposes Prometheus text when enabled.

    This test triggers representative requests so endpoint payload includes
    request, prediction, cache, external lookup, and fallback metrics.
    Parameters: none.
    """

    test_client = TestClient(create_app())
    test_client.get("/healthz")
    test_client.post(
        "/v1/predict-link",
        json={
            "entity_a_name": "Unknown Node",
            "entity_b_name": "Node One",
        },
    )
    response = test_client.get("/metrics")

    assert response.status_code == 200
    assert "model_serving_http_request_total" in response.text
    assert "model_serving_prediction_total" in response.text
    assert "model_serving_external_lookup_total" in response.text
    assert "model_serving_cache_event_total" in response.text
    assert "model_serving_fallback_total" in response.text


def test_metrics_endpoint_returns_not_found_when_disabled(
    configured_bundle_environment: None,
) -> None:
    """Verify metrics endpoint returns 404 when metrics are disabled.

    This explicit disabled behaviour keeps service configuration clear for
    environments where metrics exposure should not be available.
    Parameters: none.
    """

    disabled_metrics_service_settings = ServiceSettings(metrics_enabled=False)
    test_client = TestClient(
        create_app(service_settings=disabled_metrics_service_settings)
    )
    response = test_client.get("/metrics")

    assert response.status_code == 404
    assert "disabled" in response.json()["detail"].lower()
