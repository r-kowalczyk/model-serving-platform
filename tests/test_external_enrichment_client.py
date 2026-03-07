"""Unit tests for HTTP external enrichment client behaviour."""

from __future__ import annotations

import httpx

from model_serving_platform.infrastructure.clients.enrichment import (
    HttpExternalEnrichmentClient,
)


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
