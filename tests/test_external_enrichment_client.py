"""Unit tests for HTTP external enrichment client behaviour."""

from __future__ import annotations

from pathlib import Path

import httpx

from model_serving_platform.infrastructure.cache.local_file_cache import (
    LocalFileCacheStore,
)
from model_serving_platform.infrastructure.clients.enrichment import (
    CachingExternalEnrichmentClient,
    HttpExternalEnrichmentClient,
    _build_description_cache_key,
    _build_interaction_cache_key,
)
from model_serving_platform.infrastructure.metrics import ServiceMetrics


def test_description_lookup_retries_and_succeeds_after_transient_failures() -> None:
    """Verify description lookup retries then succeeds with JSON response.

    This test covers bounded retry logic for transient transport failures so
    enrichment handling remains robust during short external outages.
    Parameters: none.
    """

    request_attempt_counter = {"count": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return transient failures before eventual success response.

        The response sequence simulates timeout-like external instability and
        validates that retry control flow reaches a successful response.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        request_attempt_counter["count"] += 1
        if request_attempt_counter["count"] < 3:
            raise httpx.ReadTimeout("temporary timeout")
        return httpx.Response(200, json={"description": "resolved external text"})

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url="https://example.invalid/description",
        interaction_lookup_url=None,
        timeout_seconds=0.01,
        retry_count=2,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_entity_description(entity_name="Node X")

    assert request_attempt_counter["count"] == 3
    assert lookup_result.outcome == "success"
    assert lookup_result.description == "resolved external text"


def test_description_lookup_returns_failed_after_retry_exhaustion() -> None:
    """Verify description lookup reports failed after all retries are used.

    Explicit failed outcomes are required so runtime logic can apply degraded
    fallback paths instead of surfacing transport exceptions to endpoints.
    Parameters: none.
    """

    request_attempt_counter = {"count": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Always raise timeout to force retry exhaustion in test.

        This handler guarantees request failure so retry count behaviour can
        be asserted deterministically without dependency on live network.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        request_attempt_counter["count"] += 1
        raise httpx.ConnectTimeout("unreachable")

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url="https://example.invalid/description",
        interaction_lookup_url=None,
        timeout_seconds=0.01,
        retry_count=1,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_entity_description(entity_name="Node X")

    assert request_attempt_counter["count"] == 2
    assert lookup_result.outcome == "failed"
    assert lookup_result.description is None


def test_interaction_lookup_returns_success_with_partner_names() -> None:
    """Verify interaction lookup parses partner names from response payload.

    This confirms HTTP client parsing behaviour for interaction attachment
    strategy and keeps partner filtering logic stable in runtime layer.
    Parameters: none.
    """

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return one successful interaction lookup response payload.

        The payload includes two partner names to validate list extraction
        and success outcome handling in client response conversion.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        return httpx.Response(200, json={"partners": ["Node One", "Node Two"]})

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_interaction_partners(entity_name="Node X")

    assert lookup_result.outcome == "success"
    assert lookup_result.partner_entity_names == ["Node One", "Node Two"]


def test_client_reports_unavailable_when_endpoints_are_not_configured() -> None:
    """Verify unavailable outcomes when external endpoint URLs are missing.

    This behaviour is required for restricted or offline environments where
    service must continue with explicit degraded enrichment signalling.
    Parameters: none.
    """

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url=None,
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
    )

    description_lookup_result = test_client.lookup_entity_description(
        entity_name="Node X"
    )
    interaction_lookup_result = test_client.lookup_interaction_partners(
        entity_name="Node X"
    )

    assert description_lookup_result.outcome == "unavailable"
    assert interaction_lookup_result.outcome == "unavailable"
    assert test_client.supports_interaction_strategy() is False


def test_description_lookup_returns_not_found_when_payload_has_no_description() -> None:
    """Verify description lookup reports not_found for empty payload values.

    This covers explicit not-found behaviour so runtime can distinguish lookup
    failures from valid responses that simply contain no description value.
    Parameters: none.
    """

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return successful response with missing description field.

        The payload shape is valid JSON yet intentionally lacks description so
        client code enters the explicit not_found branch for this operation.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        return httpx.Response(200, json={"other_field": "value"})

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url="https://example.invalid/description",
        interaction_lookup_url=None,
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_entity_description(entity_name="Node X")

    assert lookup_result.outcome == "not_found"
    assert lookup_result.description is None


def test_interaction_lookup_returns_failed_after_request_errors() -> None:
    """Verify interaction lookup returns failed outcome after retry failures.

    This test confirms interaction dependency errors are converted to explicit
    failed outcomes rather than leaked exceptions from transport layer.
    Parameters: none.
    """

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Always raise HTTP transport error for interaction request path.

        This deterministic failure ensures lookup code takes the failed branch
        after retries are exhausted in interaction partner operation.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        raise httpx.ConnectError("connection failed")

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_interaction_partners(entity_name="Node X")

    assert lookup_result.outcome == "failed"
    assert lookup_result.partner_entity_names == []


def test_interaction_lookup_returns_not_found_for_invalid_partner_payload() -> None:
    """Verify interaction lookup reports not_found for invalid partner data.

    Response payloads that omit list-form partners should not be treated as
    failures, but as not_found outcomes for explicit fallback handling.
    Parameters: none.
    """

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return response payload that has no valid partner list field.

        This payload shape exercises the not_found branch where partners field
        exists with unexpected type and cannot be used for candidate filtering.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        return httpx.Response(200, json={"partners": "invalid"})

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_interaction_partners(entity_name="Node X")

    assert lookup_result.outcome == "not_found"
    assert lookup_result.partner_entity_names == []


def test_interaction_lookup_returns_not_found_when_partner_list_has_no_strings() -> (
    None
):
    """Verify interaction lookup handles partner arrays without string values.

    This covers the branch where partner list exists but filtering removes all
    values, resulting in explicit not_found outcome for fallback handling.
    Parameters: none.
    """

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return partner list payload with non-string values only.

        Filtering should remove all values and trigger not_found outcome for
        lookup result so runtime can apply degraded interaction fallback.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        return httpx.Response(200, json={"partners": [1, 2, 3]})

    test_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )

    lookup_result = test_client.lookup_interaction_partners(entity_name="Node X")

    assert lookup_result.outcome == "not_found"
    assert lookup_result.partner_entity_names == []


def test_caching_wrapper_returns_hit_without_second_external_call(
    tmp_path: Path,
) -> None:
    """Verify repeated description lookups reuse cached value and skip HTTP.

    This test confirms cache wrapper behaviour for description lookups by
    asserting external transport call count does not increase on repeat calls.
    Parameters: tmp_path is provided by pytest.
    """

    request_attempt_counter = {"count": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return deterministic description payload and count HTTP attempts.

        Attempt counting proves cache hit behaviour because only first lookup
        should call transport while second lookup should use stored cache data.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        request_attempt_counter["count"] += 1
        return httpx.Response(200, json={"description": "cached-description"})

    wrapped_client = HttpExternalEnrichmentClient(
        description_lookup_url="https://example.invalid/description",
        interaction_lookup_url=None,
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )
    cache_store = LocalFileCacheStore(
        cache_directory_path=tmp_path / "description-cache",
        ttl_seconds=3600.0,
    )
    caching_client = CachingExternalEnrichmentClient(
        wrapped_external_enrichment_client=wrapped_client,
        cache_store=cache_store,
    )

    first_lookup_result = caching_client.lookup_entity_description(entity_name="Node A")
    second_lookup_result = caching_client.lookup_entity_description(
        entity_name="Node A"
    )

    assert request_attempt_counter["count"] == 1
    assert first_lookup_result.description == "cached-description"
    assert second_lookup_result.description == "cached-description"


def test_caching_wrapper_caches_interaction_lookup_results(tmp_path: Path) -> None:
    """Verify interaction lookups are cached for deterministic repeated calls.

    This test confirms cache wrapper stores interaction partner responses and
    prevents repeated external calls for identical entity name lookups.
    Parameters: tmp_path is provided by pytest.
    """

    request_attempt_counter = {"count": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return interaction payload while counting transport calls.

        Counting call attempts demonstrates cache reuse for repeated lookups
        because only the first request should reach this HTTP mock handler.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        _ = request
        request_attempt_counter["count"] += 1
        return httpx.Response(200, json={"partners": ["Node One"]})

    wrapped_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
    )
    cache_store = LocalFileCacheStore(
        cache_directory_path=tmp_path / "interaction-cache",
        ttl_seconds=3600.0,
    )
    caching_client = CachingExternalEnrichmentClient(
        wrapped_external_enrichment_client=wrapped_client,
        cache_store=cache_store,
    )

    first_lookup_result = caching_client.lookup_interaction_partners(
        entity_name="Node B"
    )
    second_lookup_result = caching_client.lookup_interaction_partners(
        entity_name="Node B"
    )

    assert request_attempt_counter["count"] == 1
    assert first_lookup_result.partner_entity_names == ["Node One"]
    assert second_lookup_result.partner_entity_names == ["Node One"]


def test_cache_key_builders_are_deterministic_for_normalised_inputs() -> None:
    """Verify cache keys are deterministic for normalised equivalent inputs.

    Deterministic keys are required so equivalent entity strings with case and
    whitespace differences reuse the same cache records and avoid duplicates.
    Parameters: none.
    """

    description_key_a = _build_description_cache_key(entity_name=" Node X ")
    description_key_b = _build_description_cache_key(entity_name="node x")
    interaction_key_a = _build_interaction_cache_key(entity_name=" Node X ")
    interaction_key_b = _build_interaction_cache_key(entity_name="node x")

    assert description_key_a == description_key_b
    assert interaction_key_a == interaction_key_b


def test_caching_wrapper_delegates_interaction_capability(tmp_path: Path) -> None:
    """Verify cache wrapper preserves wrapped client interaction capability.

    Capability checks drive strategy fallback logic, so cache wrapper must
    return wrapped-client support flags without changing their semantics.
    Parameters: none.
    """

    wrapped_client = HttpExternalEnrichmentClient(
        description_lookup_url=None,
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
    )
    cache_store = LocalFileCacheStore(
        cache_directory_path=tmp_path / "capability-cache",
        ttl_seconds=60.0,
    )
    caching_client = CachingExternalEnrichmentClient(
        wrapped_external_enrichment_client=wrapped_client,
        cache_store=cache_store,
    )

    assert caching_client.supports_interaction_strategy() is True


def test_clients_record_metrics_when_metrics_collector_is_provided(
    tmp_path: Path,
) -> None:
    """Verify enrichment and cache clients emit metrics when configured.

    This test exercises metrics-enabled paths in both HTTP and cache wrappers
    so external lookup and cache event counters are populated as expected.
    Parameters: tmp_path is provided by pytest.
    """

    service_metrics = ServiceMetrics(enabled=True)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        """Return deterministic payloads for description and interaction calls.

        Payload values enable one success path per operation while allowing
        cache wrapper to produce miss, write, and subsequent hit events.
        Parameters: request is provided by HTTPX mock transport framework.
        """

        if "description" in str(request.url):
            return httpx.Response(200, json={"description": "metrics-description"})
        return httpx.Response(200, json={"partners": ["Node One"]})

    wrapped_client = HttpExternalEnrichmentClient(
        description_lookup_url="https://example.invalid/description",
        interaction_lookup_url="https://example.invalid/partners",
        timeout_seconds=0.01,
        retry_count=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(mock_handler),
        service_metrics=service_metrics,
    )
    cache_store = LocalFileCacheStore(
        cache_directory_path=tmp_path / "metrics-cache",
        ttl_seconds=3600.0,
    )
    caching_client = CachingExternalEnrichmentClient(
        wrapped_external_enrichment_client=wrapped_client,
        cache_store=cache_store,
        service_metrics=service_metrics,
    )

    caching_client.lookup_entity_description(entity_name="Node Metrics")
    caching_client.lookup_entity_description(entity_name="Node Metrics")
    caching_client.lookup_interaction_partners(entity_name="Node Metrics")
    caching_client.lookup_interaction_partners(entity_name="Node Metrics")

    prometheus_payload = service_metrics.render_prometheus_text()
    assert "model_serving_external_lookup_total" in prometheus_payload
    assert "model_serving_cache_event_total" in prometheus_payload
