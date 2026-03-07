"""Unit tests for service metrics collector behaviour."""

from model_serving_platform.infrastructure.metrics import ServiceMetrics


def test_service_metrics_enabled_records_and_renders_prometheus_payload() -> None:
    """Verify enabled metrics collector records values and renders text.

    This test covers core counter and histogram recording paths to ensure
    metrics endpoint payload includes expected metric names and labels.
    Parameters: none.
    """

    service_metrics = ServiceMetrics(enabled=True)
    service_metrics.observe_http_request(
        endpoint="/healthz",
        method="GET",
        status_code=200,
        latency_seconds=0.01,
    )
    service_metrics.increment_prediction_count(endpoint="/v1/predict-link")
    service_metrics.increment_external_lookup(
        operation="entity_description_lookup",
        outcome="success",
    )
    service_metrics.increment_cache_event(
        cache_name="entity_description_lookup",
        outcome="hit",
    )
    service_metrics.increment_fallback_usage(reason="interaction_strategy_unavailable")

    prometheus_payload = service_metrics.render_prometheus_text()

    assert "model_serving_http_request_total" in prometheus_payload
    assert "model_serving_http_request_latency_seconds" in prometheus_payload
    assert "model_serving_prediction_total" in prometheus_payload
    assert "model_serving_external_lookup_total" in prometheus_payload
    assert "model_serving_cache_event_total" in prometheus_payload
    assert "model_serving_fallback_total" in prometheus_payload


def test_service_metrics_disabled_is_noop_and_renders_empty_payload() -> None:
    """Verify disabled metrics collector does not record or render data.

    Disabled mode is required for constrained environments, so collector
    methods must no-op safely and produce an empty exposition payload.
    Parameters: none.
    """

    service_metrics = ServiceMetrics(enabled=False)
    service_metrics.observe_http_request(
        endpoint="/healthz",
        method="GET",
        status_code=200,
        latency_seconds=0.01,
    )
    service_metrics.increment_prediction_count(endpoint="/v1/predict-link")
    service_metrics.increment_external_lookup(
        operation="entity_description_lookup",
        outcome="success",
    )
    service_metrics.increment_cache_event(
        cache_name="entity_description_lookup",
        outcome="hit",
    )
    service_metrics.increment_fallback_usage(reason="interaction_strategy_unavailable")

    assert service_metrics.render_prometheus_text() == ""
